import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data import Data
from datasets import load_dataset

from src.config import parse_args_llama
from src.utils.lm_modeling import load_model, load_text2embedding
from src.utils.load import load_parquet


args = parse_args_llama()

model_name = 'sbert'
path = f'home/project/preprocessed/{args.dataset}' # path to save the encoded graphs and questions + subquestions
dataset_path = f'home/project/decomp_datasets/{args.dataset}/dataset_chunk_*.parquet' # your local path to the dataset (with subquestions)
path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'
path_subquestions = f'{path}/embs'

def step_one():

    os.makedirs(path, exist_ok=True)

    dataset = load_parquet(dataset_path)

    os.makedirs(path_nodes, exist_ok=True)
    os.makedirs(path_edges, exist_ok=True)

    for i in tqdm(range(len(dataset))):
        nodes = {}
        edges = []
        for tri in dataset[i]['graph']:
            h, r, t = tri
            h = h.lower()
            t = t.lower()
            if h not in nodes:
                nodes[h] = len(nodes)
            if t not in nodes:
                nodes[t] = len(nodes)
            edges.append({'src': nodes[h], 'edge_attr': r, 'dst': nodes[t]})
        nodes = pd.DataFrame([{'node_id': v, 'node_attr': k} for k, v in nodes.items()], columns=['node_id', 'node_attr'])
        edges = pd.DataFrame(edges, columns=['src', 'edge_attr', 'dst'])

        nodes.to_csv(f'{path_nodes}/{i}.csv', index=False)
        edges.to_csv(f'{path_edges}/{i}.csv', index=False)


def generate_split(args):

    dataset = load_dataset(f"/home/project/datasets/RoG-{args.dataset}")

    train_indices = np.arange(len(dataset['train']))
    val_indices = np.arange(len(dataset['validation'])) + len(dataset['train'])
    if args.dataset == "cwq":
        test_indices = np.arange(1000) + len(dataset['train']) + len(dataset['validation'])
    else: 
        test_indices = np.arange(len(dataset['test'])) + len(dataset['train']) + len(dataset['validation'])

    print("# train samples: ", len(train_indices))
    print("# val samples: ", len(val_indices))
    print("# test samples: ", len(test_indices))

    # Create a folder for the split
    os.makedirs(f'{path}/split', exist_ok=True)

    # Save the indices to separate files
    with open(f'{path}/split/train_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, train_indices)))

    with open(f'{path}/split/val_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, val_indices)))

    with open(f'{path}/split/test_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, test_indices)))


def step_two():

    os.makedirs(path, exist_ok=True)

    print("Cleared Memory Cache")
    torch.cuda.empty_cache()

    print('Loading dataset...')
    dataset = load_parquet(dataset_path)
    questions = [i['question'] for i in dataset]
    subquestions = [i['subquestions'] for i in dataset]

    model, tokenizer, device = load_model[model_name]()
    text2embedding = load_text2embedding[model_name]

    # encode questions
    print('Encoding questions...')
    q_embs = text2embedding(model, tokenizer, device, questions)
    torch.save(q_embs, f'{path}/q_embs.pt')

    #encode subquestions
    print('Encoding subquestions...')
    os.makedirs(f'{path_subquestions}', exist_ok=True)
    i=0
    for sqs in subquestions:
        if os.path.exists(f'{path_subquestions}/sq_embs_{i}.pt'):
            continue
        sq_embs = text2embedding(model, tokenizer, device, sqs)
        torch.save(sq_embs, f'{path_subquestions}/sq_embs_{i}.pt') 
        i+=1
    print("Finished encoding all the subquestions")

    # encode graphs
    print('Encoding graphs...')
    os.makedirs(path_graphs, exist_ok=True)
    for index in tqdm(range(len(dataset))):

        # nodes
        nodes = pd.read_csv(f'{path_nodes}/{index}.csv')
        edges = pd.read_csv(f'{path_edges}/{index}.csv')
        nodes.node_attr.fillna("", inplace=True) # warning but no error
        if len(nodes) == 0:
            print(f'Empty graph at index {index}')
            continue
        x = text2embedding(model, tokenizer, device, nodes.node_attr.tolist()) # could be  error 

        # edges
        edge_attr = text2embedding(model, tokenizer, device, edges.edge_attr.tolist())
        edge_index = torch.LongTensor([edges.src.tolist(), edges.dst.tolist()])

        pyg_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(nodes))
        torch.save(pyg_graph, f'{path_graphs}/{index}.pt')



if __name__ == '__main__':

    args = parse_args_llama()
    step_one()
    step_two()
    generate_split()