import argparse
import os

from dotenv import load_dotenv
from pathlib import Path


load_dotenv()  # Loads variables from .env into os.environ


HOME = os.getenv("HOME")
if HOME is None:
    raise RuntimeError("The HOME environment variable should exist.")

# 2. SETUP BASE DIRECTORIES
# Path(__file__).resolve().parent gives the directory of config.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 3. DEFINE PATHS WITH HIERARCHY
# Priority: 1. Actual Env Var -> 2. .env file -> 3. Fallback (Project Root)
DATASET_BASE_PATH = Path(os.getenv(
    "DATASET_PATH",
    PROJECT_ROOT / "dataset"
))

LLM_MODELS_PATH = Path(os.getenv(
    "LLM_MODELS_PATH ",
    "/home/data/dataset/huggingface/LLMs"
))

# 4. CONVENIENCE PATHS
LOG_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for directory in [DATASET_BASE_PATH, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def csv_list(string):
    return string.split(',')

def parse_args_llama():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default='graph_llm')

    parser.add_argument("--dataset", type=str, default='cwq')
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--patience", type=float, default=2)

    # Model Training
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_steps", type=int, default=2)

    # Learning Rate Scheduler
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--warmup_epochs", type=float, default=1)

    # Inference
    parser.add_argument("--eval_batch_size", type=int, default=16)

    # LLM related
    parser.add_argument("--llm_model_name", type=str, default='7b')
    parser.add_argument("--llm_model_path", type=str, default='')
    parser.add_argument("--llm_frozen", type=str, default='True')
    parser.add_argument("--llm_num_virtual_tokens", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default=f'{PROJECT_ROOT}/output')
    parser.add_argument("--max_txt_len", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_memory", type=csv_list, default=[80,80])

    # GNN related
    parser.add_argument("--gnn_model_name", type=str, default='gt')
    parser.add_argument("--gnn_num_layers", type=int, default=4)
    parser.add_argument("--gnn_in_dim", type=int, default=1024)
    parser.add_argument("--gnn_hidden_dim", type=int, default=1024)
    parser.add_argument("--gnn_num_heads", type=int, default=4)
    parser.add_argument("--gnn_dropout", type=float, default=0.0)

    args = parser.parse_args()
    return args
