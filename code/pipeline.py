import os
import pandas as pd
import torch
import gc
from tqdm import tqdm

from src.utils.lm_modeling import load_text2embedding
from src.utils.load import load_parquet
from src.utils.lm_modeling import load_model as lm
from src.dataset.utils.retrieval import retrieval_via_pcst_2, concatenate_subgraphs_2
from src.model import load_model, llama_model_path
from src.config import parse_args_llama
from torch.utils.data import Subset
from src.utils.evaluate import eval_funcs
from src.utils.ckpt import _reload_model
from src.utils.load import get_indices

# Set the desired alpha value in the launch script

alpha = os.environ.get("ALPHA")
if alpha == None:
    print("Env variable not defined")


args = parse_args_llama()


model_name = 'sbert'
path = f'/home/project/preprocessed/{args.dataset}'
alpha_path = f'{path}/decomp_reasoning/{alpha}'
dataset_path = f"/home/project/decomp_datasets/{args.dataset}/dataset_chunk_*.parquet"
output_path = f'{args.output_dir}/{args.dataset}/{alpha}'
path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'
path_embs = f'{path}/embs'
cached_graph_sub = f'{alpha_path}/cached_graphs_sub'
cached_desc_sub = f'{alpha_path}/cached_desc_sub'
cached_graph = f'{alpha_path}/cached_graphs'
cached_desc = f'{alpha_path}/cached_desc'


# Building the prompt for final answer generation

def final_prompt(question, subs):
    s = ""
    for sub in subs:
        sub_q = sub["question"]
        sub_a = sub["answer"]
        s += sub_q + sub_a + '\n'
    return ("Use the given graph and question/answer pairs to answer the following question. \nQuestion:" + question + "\n Answer:")



# Pipeline for generation

def pipeline(args):

    # Creating Directories + Loading Dataset
    os.makedirs(path, exist_ok=True)
    os.makedirs(cached_graph, exist_ok=True)
    os.makedirs(cached_desc, exist_ok=True)
    os.makedirs(cached_graph_sub, exist_ok=True)
    os.makedirs(cached_desc_sub, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    torch.cuda.empty_cache()
    print("Cleared Memory Cache")
    print('Loading dataset...')
    dataset = load_parquet(dataset_path)
    test_indices = get_indices(f"{path}/split/test_indices.txt")
    dataset = Subset(dataset, test_indices)
    print(f"Size of the fed up dataset: {len(dataset)}")

    # Loading Encoder
    model_emb, tokenizer, device = lm[model_name]()
    text2embedding = load_text2embedding[model_name]

    # Loading LLM for subquestions
    args.llm_model_path = llama_model_path[args.llm_model_name]
    model = load_model[args.model_name](args=args, init_prompt = "Your role is to answer a question using a graph.")
    if args.llm_model_name == "13b" or args.llm_model_name == "13b_chat":
        checkpoint = "model_name_graph_llm_llm_model_name_13b_llm_frozen_True_max_txt_len_512_max_new_tokens_32_gnn_model_name_gt_patience_2_num_epochs_10_checkpoint_best.pth"
    if args.llm_model_name == "7b" or args.llm_model_name == "7b_chat":
        checkpoint = "model_name_graph_llm_llm_model_name_7b_llm_frozen_True_max_txt_len_512_max_new_tokens_32_gnn_model_name_gt_patience_2_num_epochs_10_checkpoint_best.pth"
    model = _reload_model(model, f"{args.output_dir}/{args.dataset}/{checkpoint}")
    model.eval()

    # Uncomment following lines for using different LLM for final question generation

    # model = load_model[args.model_name](args=args, init_prompt = "From the given elements, answer the following question.")
    # checkpoint_sub = "model_name_graph_llm_llm_model_name_7b_llm_frozen_True_max_txt_len_512_max_new_tokens_32_gnn_model_name_gt_patience_2_num_epochs_10_checkpoint_best.pth"
    # model_2 = _reload_model(model, f"{args.output_dir}/cwq_sub/{checkpoint_sub}") # set manually to choose wanted model
    # model_2.eval()

    # Pipeline
    save_path = f'{output_path}/model_name_{args.model_name}_llm_model_name_{args.llm_model_name}_llm_frozen_{args.llm_frozen}_max_txt_len_{args.max_txt_len}_max_new_tokens_{args.max_new_tokens}_gnn_model_name_{args.gnn_model_name}_patience_{args.patience}_num_epochs_{args.num_epochs}_alpha{alpha}_test_first1000.csv'
    print(f'save_path: {save_path}')
    first = test_indices[0]
    q_embs = torch.load(f'{path}/q_embs.pt')
    all_results = []

    # Iterate over questions
    for ind, line in enumerate(tqdm(dataset)):
        index = ind + first # if you generated subquestions for entire dataset
        index = ind # if you only generated subquestions for the test set
        subanswer_list = []
        q_emb = q_embs[index]
        nodes = pd.read_csv(f'{path_nodes}/{index}.csv')
        edges = pd.read_csv(f'{path_edges}/{index}.csv')
        graph = torch.load(f'{path_graphs}/{index}.pt')
        answer = None
        subquestions = line["subquestions"]
        os.makedirs(f"{cached_graph_sub}/{index}", exist_ok=True)
        os.makedirs(f"{cached_desc_sub}/{index}", exist_ok=True)

        # Iterate over subquestions
        for j,subquestion in enumerate(subquestions):
            if not answer :
                text = subquestion
            else:
                text = answer["pred"][0] + subquestion

            # For each subquestion: performing retrieval, generating answer, conditionning the next subquestion 
            sq_emb = text2embedding(model_emb, tokenizer, device, text)
            subg, desc = retrieval_via_pcst_2(graph, q_emb, sq_emb, nodes, edges, topk=3, topk_e=5, cost_e=0.5, alpha=float(alpha))
            torch.save(subg, f'{cached_graph_sub}/{index}/{j}.pt') 
            open(f'{cached_desc_sub}/{index}/{j}.txt', 'w').write(desc) 
            sample = {"id": line["id"], "label": line["a_entity"],"subquestion":text, "desc": desc, "graph": subg}
            with torch.no_grad():
                answer = model.inference_sub(sample)
                subanswer_list.append({"question":subquestion, "answer": answer["pred"][0]})

        # Merging graphs and textual descs
        subgraphs = []
        num_subquestions = len(subquestions)
        if num_subquestions == 0:
            print(f"No subquestions at index {index}")
            continue

        for j in range(num_subquestions):
            subgraph_path = f'{cached_graph_sub}/{index}/{j}.pt'
            desc_path = f'{cached_desc_sub}/{index}/{j}.txt'
            if not os.path.exists(subgraph_path) or not os.path.exists(desc_path):
                print(f'Missing files for question {index}, subquestion {j}')
                continue
            subgraph = torch.load(subgraph_path)
            subgraphs.append((subgraph, desc_path))


        if len(subgraphs) == 0:
            print(f"No subgraphs to concatenate at index {index}")
            continue
        try:
            merged_graph, merged_desc = concatenate_subgraphs_2(subgraphs)
        except:
            continue

        # Saving graphs
        torch.save(merged_graph, f'{path}/cached_graphs/{index}.pt')
        with open(f'{path}/cached_desc/{index}.txt', 'w') as k:
            k.write(merged_desc)

        # Generate final answer
        question = final_prompt(line["question"], subanswer_list)
        label = ('|').join(line['answer']).lower()
        sample = sample = {"id": line["id"], "label": label,"subquestion":question, "desc": merged_desc, "graph": merged_graph }
        with torch.no_grad():
                answer = model.inference_sub(sample)

        df = pd.DataFrame(answer)
        all_results.append(df)
             
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(save_path, index=False)

    print("Done with pipeline !")

    # Model evaluation ; bad calls show the samples where the generated answer is incorrect
    acc, bad_calls = eval_funcs[args.dataset](save_path)

    open(f'{output_path}/bad_calls.txt',"w").write(str(bad_calls))
    open(f'{output_path}/metrics.txt',"w").write(str(acc))
    print(f'Test Acc {acc}')


if __name__ == "__main__":

    pipeline(args)

    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()