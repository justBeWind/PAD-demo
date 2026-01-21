import torch
import json
import argparse
import os
import numpy as np
from transformers import AutoModelForCausalLM
from tqdm import tqdm

def calculate_density_smoothed(model_name, output_path, k=20, device="cuda"):
    """
    DenPAD Implementation: Semantic Manifold Density (Local Smoothing).
    
    Difference from DYNTEXT:
    - DYNTEXT uses the distance to the K-th neighbor (Single Point).
    - DenPAD uses the AVERAGE distance of the top-K neighbors (Manifold Smoothing).
    
    This provides a more robust estimation of the local semantic sparsity.
    """
    print(f"Loading model: {model_name}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, 
            device_map=device
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    if hasattr(model, "gpt_neox"):
        embed_weight = model.gpt_neox.embed_in.weight.detach()
    elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        embed_weight = model.model.embed_tokens.weight.detach()
    else:
        embed_weight = model.get_input_embeddings().weight.detach()

    vocab_size = embed_weight.shape[0]
    print(f"Vocab size: {vocab_size}, Dim: {embed_weight.shape[1]}")

    # To avoid OOM
    chunk_size = 500
    
    # === Step 1: Calculate Smoothed Distances ===
    # 我们不再只存第 K 个距离，而是计算前 K 个的平均值
    avg_knn_distances = []
    
    print(f"Phase 1: Calculating Average K-NN Distances (Smoothing, K={k})...")
    
    embed_weight = embed_weight.to(device).float()
    
    for i in tqdm(range(0, vocab_size, chunk_size)):
        end = min(i + chunk_size, vocab_size)
        chunk_indices = slice(i, end)
        chunk_vecs = embed_weight[chunk_indices]
        
        # Euclidean Distance
        dists = torch.cdist(chunk_vecs, embed_weight, p=2)
        
        # Find K nearest neighbors (excluding self, so k+1)
        # topk returns largest values, so we use largest=False for smallest distances
        topk_res = torch.topk(dists, k=k+1, dim=1, largest=False)
        topk_vals = topk_res.values # [batch, k+1]
        
        # [核心修改] Innovation: Semantic Smoothing
        # DYNTEXT: kth_dists = topk_vals[:, k]
        # DenPAD:  avg_dists = mean(topk_vals[:, 1:])  (Index 0 is self, dist=0)
        
        # 取第 1 到 第 k 个邻居的距离 (排除 0)
        neighbor_dists = topk_vals[:, 1:] # [batch, k]
        avg_dists = neighbor_dists.mean(dim=1) # [batch]
        
        avg_knn_distances.append(avg_dists.cpu())
        
        del dists, topk_vals, chunk_vecs
        torch.cuda.empty_cache()

    avg_knn_distances = torch.cat(avg_knn_distances).numpy()
    
    # === Step 2: Calculate Threshold (Gamma) ===
    # Gamma 也是基于平均距离的平均值
    gamma = np.mean(avg_knn_distances)
    print(f"Calculated Manifold Threshold (Gamma): {gamma:.6f}")
    
    # === Step 3: Calculate Manifold Density ===
    # f(t) = count of tokens where Smoothed_Distance <= Gamma
    # 注意：为了保持计算一致性，这里我们不需要重新算 cdist
    # 因为我们已经有了 avg_knn_distances，我们可以直接用它来衡量“稀疏度”。
    # 但为了对齐 DYNTEXT 的“计数法”定义（密度=邻居多），我们还是得重新扫描一遍
    # 只要一个词的 "平均邻域半径" 小于 Gamma，说明它周围很挤 -> 密度高
    
    # 实际上，更科学的方法是直接用 avg_knn_distances 作为“稀疏度指标”
    # Distance 越小 -> 越 Dense
    # Distance 越大 -> 越 Sparse
    # 所以 Density = -Distance (或者 1/Distance)
    
    # 为了让代码改动最小且符合 "Density Map" 格式 (0~1, 1=Dense)：
    # 我们直接归一化反转 Distance。
    
    print("Phase 2: Converting Smoothed Distances to Density...")
    
    # Distance: Min (0.0, very dense) -> Max (large, very sparse)
    d_min = avg_knn_distances.min()
    d_max = avg_knn_distances.max()
    print(f"Smoothed Distance Range: min={d_min:.4f}, max={d_max:.4f}")
    
    # Normalize Distance to [0, 1]
    # norm_dist 0.0 = Dense (cluster center)
    # norm_dist 1.0 = Sparse (outlier)
    normalized_distances = (avg_knn_distances - d_min) / (d_max - d_min + 1e-8)
    
    # Density = 1 - Normalized_Distance
    # Density 1.0 = Dense
    # Density 0.0 = Sparse
    final_densities = 1.0 - normalized_distances
    
    # Save
    density_list = [round(float(x), 5) for x in final_densities]
    
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(output_path, "w") as f:
        json.dump(density_list, f)
        
    print(f"Success! Smoothed Local Density map saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-6.9b")
    # 为了区分，建议文件名加上 smoothed
    parser.add_argument("--output", type=str, default="data/pythia_6.9b_density_smoothed.json")
    parser.add_argument("--k", type=int, default=20)
    args = parser.parse_args()
    
    calculate_density_smoothed(args.model_name, args.output, k=args.k)