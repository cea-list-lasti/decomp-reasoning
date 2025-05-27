
import re
import os
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Subset
import torch
import networkx as nx
from torch_geometric.utils import to_networkx

from src.config import parse_args_llama
from src.utils.load import load_parquet
from src.utils.lm_modeling import load_model, load_text2embedding
from src.dataset.utils.retrieval import retrieval_via_pcst_2, retrieval_filter_test


alpha = os.environ.get("ALPHA")
if alpha == None:
    print("Env variable not defined")

args = parse_args_llama()

dataset_path = f"/home/project/decomp_datasets/{args.dataset}/dataset_chunk_*.parquet"



def exact_matching(test_range, path):

    dataset = load_parquet(dataset_path)
    dataset = Subset(dataset, test_range)

    cached_desc = f'{path}/cached_desc'

    print(len(dataset))
    matches = 0

    for i in tqdm(test_range):
        graph_text_file = f"{cached_desc}/{i}.txt"
        node_attr_map = load_graph_text(graph_text_file)  # load node_id -> node_attr mapping

        a_entity_list = [entity.lower() for entity in dataset[i - test_range[0]]["answer"]]

        for a_entity in a_entity_list:
            matched_nodes = [node_id for node_id, node_attr in node_attr_map.items() if exact_match(a_entity,node_attr)]
            matched_nodes = [node_id for node_id, node_attr in node_attr_map.items() if a_entity in node_attr] # for flexible matching via a similarity threshold

            if matched_nodes:
                matches += 1
                print(f"Found match for '{a_entity}': Node IDs {matched_nodes}")

    print(f'Presence of answer entity for our method: {matches / len(test_range)}')


def load_graph_text(file_path):
    """Parses the textual graph file and returns a dictionary {node_id: node_attr}."""
    node_attr_map = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip() == "src,edge_attr,dst":
                break  # stop at the edges section
            
            if ',' in line:
                node_id, node_attr = line.strip().split(',', 1)  # split at first comma
                node_attr_map[node_id] = node_attr.lower()


    return node_attr_map

def exact_match(answer, node_attr):
    """Checks if the answer is a standalone word in node_attr."""
    words = re.split(r'\W+', node_attr)  # split by non-word characters
    return answer in words


# Check graph connectivity

def is_graph_connected(data):
    G = to_networkx(data, to_undirected=True) 
    return nx.is_connected(G)

def check_connectivity(path):
    cached_graph = f'{path}/cached_graphs'
    counter = 0
    for i in tqdm(range(31158,32157)):
        graph = torch.load(f'{cached_graph}/{i}.pt')
        if is_graph_connected(graph):
            counter += 1
        else:
            print("Graph is not connected")
    print(counter)
    print(counter/999)

def graph_density(data):
    num_nodes = data.num_nodes
    num_edges = data.edge_index.size(1)

    if num_nodes < 2:
        return 0  # A single node or empty graph has density 0

    return (2 * num_edges) / (num_nodes * (num_nodes - 1))

def check_density(path):
    cached_graph = f'{path}/cached_graphs'
    total_density = 0
    for i in tqdm(range(31158,32157)):
        graph = torch.load(f'{cached_graph}/{i}.pt')
        density = graph_density(graph)
        total_density += density
    print(total_density/999)



def check_size(test_range):

    dataset = load_parquet(dataset_path)
    dataset = Subset(dataset, test_range)

    path = '/home/data/edufraisse/vsix/DATA/dataset/subquestions/cond/pipeline_4/0.5'
    cached_graphs  =f'{path}/cached_graphs'

    total_size = 0

    for i in tqdm(test_range):
        graph = torch.load(f'{cached_graphs}/{i}.pt')
        size = graph.num_nodes
        total_size += size
    print(total_size/999)



def filter_centrality(test_range):
    dataset = load_parquet(dataset_path)
    dataset = Subset(dataset, test_range)

    index = 0
    cwq_path = '/home/data/edufraisse/vsix/DATA/dataset/cwq'
    path_nodes = f'{cwq_path}/nodes'
    path_edges = f'{cwq_path}/edges'
    path_graphs = f'{cwq_path}/graphs'

    nodes = pd.read_csv(f'{path_nodes}/{index}.csv')
    edges = pd.read_csv(f'{path_edges}/{index}.csv')
    graph = torch.load(f'{path_graphs}/{index}.pt')

    q_embs = torch.load(f'{cwq_path}/q_embs.pt')
    q_emb = q_embs[index]
    
    model_name = 'sbert'
    model, tokenizer, device = load_model[model_name]()
    text2embedding = load_text2embedding[model_name]

    subquestions = dataset[index]["subquestions"]
    sq_emb = text2embedding(model, tokenizer, device, subquestions[0])

    print("Ready for retrieval")

    subg_test, subd_test = retrieval_filter_test(graph, q_emb, sq_emb, nodes, edges, topk=3, topk_e=3, cost_e=0.5, alpha=0.5)

    print("Done with test")

    subg, subd = retrieval_via_pcst_2(graph, q_emb, sq_emb, nodes, edges, topk=3, topk_e=3, cost_e=0.5, alpha=0.5)

    print("Done with baseline")

    print(subd_test)
    print(subd)

