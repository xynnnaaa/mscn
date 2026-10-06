import argparse
import torch
import numpy as np

def compute_qerror(preds, trues):
    """计算标准的 Q-Error"""
    p = np.clip(preds, 1.0, None)
    t = np.clip(trues, 1.0, None)
    return np.maximum(p / t, t / p)

def print_metrics_table(title, qerrors):
    """格式化打印基数估计指标表格"""
    metrics = {
        "Median": np.median(qerrors),
        "80th%": np.percentile(qerrors, 80),
        "90th%": np.percentile(qerrors, 90),
        "95th%": np.percentile(qerrors, 95),
        "99th%": np.percentile(qerrors, 99),
        "Max": np.max(qerrors),
        "Mean": np.mean(qerrors)
    }
    
    print(f"\n📊 Scenario: {title}")
    print(f"{'Method/Metric':<20} | {'Median':<8} | {'80th%':<8} | {'90th%':<8} | {'95th%':<8} | {'99th%':<8} | {'Max':<8} | {'Mean':<8}")
    print("-" * 95)
    print(f"{'Result':<20} | " + " | ".join([f"{metrics[k]:<8.4f}" for k in metrics]))

def extract_embedding_matrix(pt_path):
    """从 { seq_id: {alias: tensor} } 结构中对齐提取高维矩阵，并处理多表平均"""
    print(f"Loading embeddings from: {pt_path}")
    emb_dict = torch.load(pt_path, map_location="cpu")
    
    tensors = []
    num_queries = len(emb_dict)
    
    for i in range(num_queries):
        alias_dict = emb_dict.get(i) or emb_dict.get(str(i))
        if alias_dict is None:
            raise KeyError(f"Error: seq_id {i} not found in {pt_path}. Please check alignment.")
        
        query_tensors = []
        for alias, tensor_val in alias_dict.items():
            if isinstance(tensor_val, torch.Tensor):
                tensor_val = tensor_val.detach().cpu().numpy()
            query_tensors.append(tensor_val)
        
        query_aggregated_tensor = np.mean(query_tensors, axis=0)
        tensors.append(query_aggregated_tensor)
        
    matrix = np.array(tensors, dtype=np.float32)
    print(f"Successfully extracted matrix shape: {matrix.shape}")
    return matrix

def main():
    parser = argparse.ArgumentParser(description="Fallback Mechanism Validation Script")
    parser.add_argument("--train_pt", required=True, help="Path to train_embeddings.pt")
    parser.add_argument("--test_pt", required=True, help="Path to test_embeddings.pt")
    parser.add_argument("--qa_csv", required=True, help="Path to QA-Embedding result (pred,true) CSV")
    parser.add_argument("--baseline_csv", required=True, help="Path to Baseline result (pred,true) CSV")
    parser.add_argument("--output_csv", default="merged_result_0.70.csv", help="Path to save the merged CSV for threshold 0.70")
    args = parser.parse_args()

    # 1. 读取高维特征矩阵
    X_train = extract_embedding_matrix(args.train_pt)
    X_test = extract_embedding_matrix(args.test_pt)

    # 2. 读取基数估计 CSV 数据
    qa_data = np.loadtxt(args.qa_csv, delimiter=',')
    base_data = np.loadtxt(args.baseline_csv, delimiter=',')
    
    qa_preds, qa_trues = qa_data[:, 0], qa_data[:, 1]
    base_preds, base_trues = base_data[:, 0], base_data[:, 1]
    qa_qerrors = compute_qerror(qa_preds, qa_trues)
    
    # 3. [核心修改] 剥离 Base 干扰，切片出纯粹的 PCA 向量段 (769维之后的所有维度)
    print("\n⚡ Slicing PCA components and calculating Pure PCA Cosine Similarities...")
    t_train_all = torch.from_numpy(X_train)
    t_test_all = torch.from_numpy(X_test)
    
    # 截取后 1536 维的 PCA 向量
    t_train_pca = t_train_all[:, 769:]
    t_test_pca = t_test_all[:, 769:]
    
    # 安全归一化，防止全零填充向量导致除以 0 产生 NaN
    train_denom = t_train_pca.norm(dim=1, keepdim=True)
    train_denom[train_denom == 0] = 1e-9
    t_train_norm = t_train_pca / train_denom
    
    test_denom = t_test_pca.norm(dim=1, keepdim=True)
    test_denom[test_denom == 0] = 1e-9
    t_test_norm = t_test_pca / test_denom
    
    max_sim_list = []
    max_idx_list = []
    batch_size = 500
    
    for i in range(0, len(t_test_norm), batch_size):
        batch_test = t_test_norm[i:i+batch_size]
        sim_matrix = torch.mm(batch_test, t_train_norm.t())
        max_vals, max_idxs = torch.max(sim_matrix, dim=1)
        max_sim_list.append(max_vals.numpy())
        max_idx_list.append(max_idxs.numpy())
        
    max_similarities = np.concatenate(max_sim_list, axis=0)
    max_sim_indices = np.concatenate(max_idx_list, axis=0)

    # 如果测试集本身就是全零 PCA（即样本不够的查询），其相似度会趋近 0，自然触发回退，逻辑完美闭环

    # 4. 特征深度分析：精准检索并定位所有 Q-Error > 50000 的坏点
    print("\n" + "="*75)
    print("🎯 DETAILED ANALYSIS FOR HIGH Q-ERROR OUTLIERS (Q-Error > 50000)")
    print("="*75)
    
    high_error_mask = qa_qerrors > 50000
    high_error_val_indices = np.where(high_error_mask)[0]
    
    print(f"Total outliers found with Q-Error > 50000: {len(high_error_val_indices)}")
    print("-" * 75)
    
    for rank, val_idx in enumerate(high_error_val_indices, 1):
        train_idx = max_sim_indices[val_idx]
        sim_val = max_similarities[val_idx]
        
        print(f"Outlier #{rank}:")
        print(f"  -> [Test Set] Query Index (Row ID) : {val_idx}")
        print(f"  -> [Test Set] True Cardinality     : {qa_trues[val_idx]}")
        print(f"  -> [Test Set] QA Model Predicted   : {qa_preds[val_idx]}")
        print(f"  -> [Test Set] QA Model Q-Error     : {qa_qerrors[val_idx]:.4f}")
        print(f"  -> [Test Set] Baseline Q-Error     : {compute_qerror(base_preds[val_idx], qa_trues[val_idx]):.4f}")
        print(f"  -> 🔍 [Train Set Connection] Nearest PCA Query Index : {train_idx}")
        print(f"  -> 🤝 [Train Set Connection] Max PCA Cosine Similarity: {sim_val:.6f}")
        print("-" * 75)

    print_metrics_table("Original QA-Embedding", qa_qerrors)
    print_metrics_table("Original Baseline Method", compute_qerror(base_preds, base_trues))

    # 5. [核心修改] 调整模拟阈值：由于去除了Base的高分基底，PCA相似度分布在较宽区间，调整为更合理的探查范围
    print("\n" + "="*75)
    print("🚀 SIMULATING FALLBACK UNDER DIFFERENT THRESHOLDS")
    print("="*75)
    print("Fallback Logic: If Max_PCA_Similarity < Threshold -> Use Baseline Predict")
    
    test_thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    for thresh in test_thresholds:
        merged_preds = np.copy(qa_preds)
        
        fallback_mask = max_similarities < thresh
        merged_preds[fallback_mask] = base_preds[fallback_mask]
        
        merged_qerrors = compute_qerror(merged_preds, qa_trues)
        num_triggered = np.sum(fallback_mask)
        pct_triggered = (num_triggered / len(qa_preds)) * 100
        
        print_metrics_table(f"Merged Result (Threshold={thresh:.2f} | Triggered: {num_triggered}/{len(qa_preds)} - {pct_triggered:.2f}%)", merged_qerrors)

        # if np.isclose(thresh, 0.40):  # 修改默认保存阈值为 0.40，以便测试
        #     output_matrix = np.column_stack((merged_preds, qa_trues))
        #     np.savetxt(args.output_csv, output_matrix, delimiter=',', fmt='%.4f')
        #     print(f"\n💾 [SAVE SUCCESS] Threshold=0.40 的融合数据已成功写入到: {args.output_csv}")

if __name__ == "__main__":
    main()

# import argparse
# import torch
# import numpy as np

# def analyze_pt_file(pt_path):
#     print(f"\n" + "="*60)
#     print(f"🔍 DIAGNOSING FILE: {pt_path}")
#     print("="*60)
    
#     data = torch.load(pt_path, map_location='cpu')
    
#     all_vectors = []
#     for seq_id, alias_dict in data.items():
#         for alias, tensor in alias_dict.items():
#             if isinstance(tensor, torch.Tensor):
#                 tensor = tensor.detach().cpu().numpy()
#             all_vectors.append(tensor)
            
#     matrix = np.array(all_vectors, dtype=np.float32)
#     num_vectors, dim = matrix.shape
#     print(f"📈 Total Extracted Table Vectors : {num_vectors}")
#     print(f"📐 Vector Dimension               : {dim}")
    
#     if np.isnan(matrix).any():
#         print("⚠️ [WARNING] Found NaN values in the embedding file!")
    
#     # 动态识别单表 matched (2305维) 还是包含 unmatched (4610维)
#     if dim == 2305 or dim == 4610:
#         is_double = (dim == 4610)
#         stride = 2305 if is_double else 0
        
#         # 无论是单还是双，我们只切出一组标准 2305 结构进行诊断
#         base_emb = matrix[:, stride : stride + 768]
#         log_cnt  = matrix[:, stride + 768]
#         pc1      = matrix[:, stride + 769 : stride + 1537]
#         pc2      = matrix[:, stride + 1537 : stride + 2305]
        
#         # 1. 检查数值区间 (Scale)
#         print("\n💡 [1. Scale Analysis] - Checking if PCA is dominated by Base Embedding:")
#         print(f"  -> Base Embedding : Mean Abs = {np.mean(np.abs(base_emb)):.6f} | Std = {np.std(base_emb):.6f}")
#         print(f"  -> Log Count      : Mean Abs = {np.mean(np.abs(log_cnt)):.6f} | Std = {np.std(log_cnt):.6f}")
#         print(f"  -> PCA - PC1      : Mean Abs = {np.mean(np.abs(pc1)):.6f} | Std = {np.std(pc1):.6f}")
#         print(f"  -> PCA - PC2      : Mean Abs = {np.mean(np.abs(pc2)):.6f} | Std = {np.std(pc2):.6f}")
        
#         # 2. 检查零值填充比例 (Zero-Padding Check)
#         pc1_zeros = np.all(np.abs(pc1) < 1e-7, axis=1)
#         pc2_zeros = np.all(np.abs(pc2) < 1e-7, axis=1)
#         pure_zero_pca = np.sum(pc1_zeros & pc2_zeros)
#         print("\n💡 [2. Zero-Padding Check] - Checking how many queries lacked enough tuples:")
#         print(f"  -> Vectors with completely ZERO PCA: {pure_zero_pca} / {num_vectors} ({pure_zero_pca/num_vectors*100:.2f}%)")
        
#         # 3. 独立计算各部分的内在相似度 (Anisotropy Check)
#         print("\n💡 [3. Intrinsic Cosine Similarity Check] - Inside this file:")
#         # 随机抽取500对向量算彼此之间的相似度
#         np.random.seed(42)
#         idx1 = np.random.choice(num_vectors, min(500, num_vectors), replace=False)
#         idx2 = np.random.choice(num_vectors, min(500, num_vectors), replace=False)
        
#         def cosine_sim(arr1, arr2):
#             n1 = np.linalg.norm(arr1, axis=1, keepdims=True)
#             n2 = np.linalg.norm(arr2, axis=1, keepdims=True)
#             n1[n1 == 0], n2[n2 == 0] = 1e-9, 1e-9
#             return np.sum((arr1 / n1) * (arr2 / n2), axis=1)
            
#         sim_base = cosine_sim(base_emb[idx1], base_emb[idx2])
#         print(f"  -> Base Embedding Pairwise Similarity : Median = {np.median(sim_base):.4f} | Mean = {np.mean(sim_base):.4f}")
        
#         valid_pca_mask = (~pc1_zeros[idx1]) & (~pc1_zeros[idx2])
#         if np.sum(valid_pca_mask) > 10:
#             sim_pc1 = cosine_sim(pc1[idx1][valid_pca_mask], pc1[idx2][valid_pca_mask])
#             sim_pc2 = cosine_sim(pc2[idx1][valid_pca_mask], pc2[idx2][valid_pca_mask])
#             print(f"  -> PC1 Pairwise Similarity (Valid only): Median = {np.median(sim_pc1):.4f} | Mean = {np.mean(sim_pc1):.4f}")
#             print(f"  -> PC2 Pairwise Similarity (Valid only): Median = {np.median(sim_pc2):.4f} | Mean = {np.mean(sim_pc2):.4f}")
#         else:
#             print("  -> PCA Pairwise Similarity            : Too many zero vectors to calculate reliably.")
            
#     else:
#         print(f"❌ Unknown dimension {dim}. Expected 2305 or 4610.")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--pt", required=True, help="Path to the .pt embedding file to diagnose")
#     args = parser.parse_args()
#     analyze_pt_file(args.pt)