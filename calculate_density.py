import torch
import json
import argparse
import os
import numpy as np
from transformers import AutoModelForCausalLM
from tqdm import tqdm

def calculate_density(model_name, output_path, k=20, device="cuda"):
    """
    Strict implementation of DYNTEXT density calculation logic.
    1. Calculate Euclidean distance matrix.
    2. Find K-th nearest neighbor distance for each token.
    3. Calculate Gamma (average of K-th distances).
    4. Calculate Density (count of neighbors within Gamma).
    5. Min-Max Normalize.
    """
    print(f"Loading model: {model_name}...")
    # Load model in FP16 to save memory (Pythia-6.9B takes ~14GB VRAM)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        device_map=device
    )
    
    # Extract Embedding Matrix [Vocab_Size, Hidden_Dim]
    # For Pythia (GPT-NeoX), the layer is gpt_neox.embed_in
    if hasattr(model, "gpt_neox"):
        embed_weight = model.gpt_neox.embed_in.weight.detach()
    elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"): # Llama/Mistral
        embed_weight = model.model.embed_tokens.weight.detach()
    else:
        # Fallback for other architectures (e.g. GPT2)
        embed_weight = model.get_input_embeddings().weight.detach()

    vocab_size = embed_weight.shape[0]
    hidden_dim = embed_weight.shape[1]
    print(f"Vocab size: {vocab_size}, Embedding dim: {hidden_dim}")

    # To avoid OOM, we calculate distances in chunks
    # DYNTEXT Step 1 & 2: Get d(t, t_K)
    dist_to_kth_neighbor = []
    chunk_size = 500  # Conservative chunk size for 24GB VRAM
    
    print(f"Phase 1: Calculating K-th ({k}) nearest neighbor distances...")
    
    # Ensure embeddings are on GPU and float32 for precision in distance calc
    embed_weight = embed_weight.to(device)
    
    for i in tqdm(range(0, vocab_size, chunk_size)):
        end = min(i + chunk_size, vocab_size)
        chunk_indices = slice(i, end)
        chunk_vecs = embed_weight[chunk_indices].float() # [batch, dim]
        
        # Compute pairwise Euclidean distances: |x-y|
        # torch.cdist creates a [batch, vocab_size] matrix
        dists = torch.cdist(chunk_vecs, embed_weight.float(), p=2)
        
        # Find k-th nearest neighbor
        # Note: Top-1 is the token itself (distance 0), so we need (k+1)-th smallest
        # topk returns largest, so we use largest=False (smallest)
        topk_vals = torch.topk(dists, k=k+1, dim=1, largest=False).values # [batch, k+1]
        
        # The K-th neighbor is at index k (0-based, index 0 is self)
        kth_dists = topk_vals[:, k]
        dist_to_kth_neighbor.append(kth_dists.cpu())
        
        # Cleanup to free VRAM
        del dists, topk_vals, chunk_vecs
        torch.cuda.empty_cache()

    dist_to_kth_neighbor = torch.cat(dist_to_kth_neighbor).numpy()
    
    # DYNTEXT Step 3: Calculate Gamma (Average Density Range)
    gamma = np.mean(dist_to_kth_neighbor)
    print(f"Calculated Threshold Gamma: {gamma:.6f}")
    
    # DYNTEXT Step 4: Calculate Density f(t)
    # f(t) = count of tokens where distance <= Gamma
    print("Phase 2: Calculating token densities...")
    raw_densities = []
    
    for i in tqdm(range(0, vocab_size, chunk_size)):
        end = min(i + chunk_size, vocab_size)
        chunk_indices = slice(i, end)
        chunk_vecs = embed_weight[chunk_indices].float()
        
        dists = torch.cdist(chunk_vecs, embed_weight.float(), p=2)
        
        # Count neighbors within gamma
        count = (dists <= gamma).sum(dim=1).cpu().numpy()
        raw_densities.append(count)
        
        del dists, chunk_vecs
        torch.cuda.empty_cache()
        
    raw_densities = np.concatenate(raw_densities)
    
    # DYNTEXT Step 5: Min-Max Normalization
    # F_hat(t) = (f(t) - min) / (max - min)
    f_min = raw_densities.min()
    f_max = raw_densities.max()
    print(f"Density Raw Range: min={f_min}, max={f_max}")
    
    normalized_densities = (raw_densities - f_min) / (f_max - f_min + 1e-8)
    
    # Invert for Sensitivity: 
    # High Density (1.0) -> Common Word -> Low Sensitivity (0.0)
    # Low Density (0.0) -> Rare Word -> High Sensitivity (1.0)
    # We save the DENSITY, logic will be handled in LLMEngine
    
    # Save as JSON list
    density_list = [round(float(x), 5) for x in normalized_densities]
    
    with open(output_path, "w") as f:
        json.dump(density_list, f)
        
    print(f"Success! Density map saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-6.9b")
    parser.add_argument("--output", type=str, default="data/pythia_6.9b_density.json")
    parser.add_argument("--k", type=int, default=20, help="Parameter K for density calculation (Default: 20)")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    calculate_density(args.model_name, args.output, k=args.k)