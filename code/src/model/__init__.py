from src.model.llm import LLM
from src.model.pt_llm import PromptTuningLLM
from src.model.graph_llm import GraphLLM


load_model = {
    "llm": LLM,
    "inference_llm": LLM,
    "pt_llm": PromptTuningLLM,
    "graph_llm": GraphLLM
}

# Adapt the following with the model paths
llama_model_path = {
    "7b": "/home/data/dataset/huggingface/LLMs/meta-llama/Llama-2-7b-hf",
    "7b_chat": "/home/data/dataset/huggingface/LLMs/meta-llama/Llama-2-7b-chat-hf",
    "13b": "/home/data/dataset/huggingface/LLMs/meta-llama/Llama-2-13b-hf",
    "13b_chat": "/home/data/dataset/huggingface/LLMs/meta-llama/Llama-2-13b-chat-hf"
}
