"""
Nov-10, 2025
Johan C

(Archived)
Tried to augment preference dataset via Latent space synthesis.
cf. Tao et al., Limited Preference Data? Learning Better Reward Model with Latent Space Synthesis
"""

import sys
import logging
from itertools import takewhile
import argparse

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

from train import apply_chat_template

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

def get_all_layer_embeddings(texts, tokenizer, model, device="cuda"):
    """
    For each text, extract hidden states for ALL layers,
    do mean pooling per layer, and stack.
    Returns: (num_samples, num_layers, hidden_dim)
    """
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", max_length=8000, truncation=True).to(device)
            outputs = model(**inputs, output_hidden_states=True)
            # outputs.hidden_states: tuple of (layer# x (1, seq_len, hidden_dim))
            layer_means = []
            for layer_hid in outputs.hidden_states:
                # mean pool along sequence
                mean_emb = layer_hid[0].mean(dim=0)  # (hidden_dim,)
                mean_emb = mean_emb.detach().to(torch.float16).cpu().numpy()
                layer_means.append(mean_emb)
            all_embeddings.append(np.stack(layer_means, axis=0))
    return np.stack(all_embeddings, axis=0)  # shape: (num_samples, num_layers, hidden_dim)

def visualize_embeddings_from_last_layer(embs4d, ids, save_pic_path: str):
    """
    embs4d: (num_samples, 2, num_layers, hidden_dim)
    ids: example ids
    Visualize last layer (mean pooled) for chosen (pos=0) and rejected (pos=1).
    """
    num_samples = embs4d.shape[0]
    # convention: pos=0 is chosen, pos=1 is rejected
    chosen_embs = embs4d[:,0,-1,:]  # (num_samples, hidden_dim)
    rejected_embs = embs4d[:,1,-1,:]
    X = np.vstack([chosen_embs, rejected_embs])
    pca = PCA(n_components=2)
    X_emb2d = pca.fit_transform(X)
    chosen_2d = X_emb2d[:len(chosen_embs)]
    rejected_2d = X_emb2d[len(chosen_embs):]

    plt.figure(figsize=(8,8))
    for i in range(len(chosen_2d)):
        dx = chosen_2d[i,0] - rejected_2d[i,0]
        dy = chosen_2d[i,1] - rejected_2d[i,1]
        if dx == 0 and dy == 0:
            print(ids[i], "chosen embedding == rejected embedding")
        else:
            print(ids[i], "chosen embedding != rejected embedding")
            plt.arrow(
                rejected_2d[i,0], rejected_2d[i,1],
                dx, dy,
                color="blue", alpha=0.3, head_width=0.1, length_includes_head=True
            )
        plt.scatter([rejected_2d[i,0]], [rejected_2d[i,1]], color="red", marker="x")
        plt.scatter([chosen_2d[i,0]], [chosen_2d[i,1]], color="green", marker="o")
    plt.title("Preference pairs (arrow: rejected → chosen; o: positive, x: negative)")
    plt.xlabel("PCA component 1"); plt.ylabel("PCA component 2")
    plt.tight_layout()
    plt.savefig(save_pic_path)
    logger.info(f"Plot saved to {save_pic_path}")

def main(
    model_name_or_path,
    data_path,
    device="cuda" if torch.cuda.is_available() else "cpu",
    num_points=1000,
    npy_save_path="llm_embeddings.npy"
):
    logger.info(f"Loading model: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True
    ).to(device)

    if ".json" in data_path:
        raw_datasets = load_dataset( "json", data_files=data_path.split("||") )
    else:
        raw_datasets = load_dataset(data_path)
    logger.info(f"Loaded splits: {list(raw_datasets.keys())}")

    # We'll only use a subset if large
    ds = raw_datasets["train"]
    inds = np.arange(min(len(ds), num_points))
    ds = ds.select(inds.tolist())
    logger.info(f"Computing embeddings for {len(ds)} pairs.")

    id_inputs = []
    chosen_inputs = []
    rejected_inputs = []
    for example in ds:
        chosen = example["prompt"] + example["chosen"]
        rejected = example["prompt"] + example["rejected"]
        id_inputs.append(example["id"])
        chosen_inputs.append(chosen)
        rejected_inputs.append(rejected)
    num_samples = len(chosen_inputs)
    logger.info(f"Total samples for embedding: {num_samples}")

    # Compute all layer embeddings for both chosen & rejected responses
    logger.info("Extracting (chosen) embeddings...")
    chosen_embs_all = get_all_layer_embeddings(chosen_inputs, tokenizer, model, device)  # (n, num_layers, dim)
    logger.info("Extracting (rejected) embeddings...")
    rejected_embs_all = get_all_layer_embeddings(rejected_inputs, tokenizer, model, device)  # (n, num_layers, dim)

    # Compose result: (num_samples, 2, num_layers, dim)
    embs4d = np.stack([chosen_embs_all, rejected_embs_all], axis=1)
    logger.info(f"Final stacked embedding shape: {embs4d.shape} (num_samples, pos/neg, num_layers, dim)")
    np.save(npy_save_path, embs4d)
    logger.info(f"Saved 4D embeddings to {npy_save_path}")

    # Still enable PCA visualizing last layer's mean pooled emb
    visualize_embeddings_from_last_layer(embs4d, id_inputs, save_pic_path="data_embedding/llm_embedding_mean_pooling_lastlayer.png")

def print_npy_embedding_shape(npy_path):
    """
    Utility function to load a numpy .npy embedding file and print its shape.
    """
    arr = np.load(npy_path)
    print(f"Embedding file '{npy_path}' shape: {arr.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_points", type=int, default=100)
    parser.add_argument("--npy_save_path", type=str, default="data_embedding/llm_embeddings.npy")
    args = parser.parse_args()
    main(
        model_name_or_path=args.model_name_or_path,
        data_path=args.data_path,
        device=args.device,
        num_points=args.num_points,
        npy_save_path=args.npy_save_path
    )

    # print_npy_embedding_shape("llm_embeddings.npy")