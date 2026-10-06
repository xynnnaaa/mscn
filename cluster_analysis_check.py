# import os
# import argparse
# import json
# import torch
# import torch.nn.functional as F
# import numpy as np
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE

# from mscn.data import load_and_encode_all_data
# from mscn.model import SetConv

# extracted_features = []
# current_s_mask = None

# def embedding_hook_fn(module, input_tensor, output_tensor):
#     global current_s_mask
#     activated = F.leaky_relu(output_tensor, negative_slope=0.01)
    
#     if current_s_mask is not None:
#         # 1. 用 mask 把无效表的偏置向量彻底归零
#         masked_activated = activated * current_s_mask
#         # 2. 对有效表求和
#         summed = torch.sum(masked_activated, dim=1)
#         # 3. 统计每个查询真正有多少张有效的表，避免除以 0
#         counts = torch.sum(current_s_mask, dim=1).clamp(min=1.0)
#         # 4. 精准平均池化
#         pooled = summed / counts
#     else:
#         pooled = torch.mean(activated, dim=1)
        
#     extracted_features.append(pooled.cpu().numpy())

# def run_kmeans_analysis(eval_config_path, qerror_csv_path):
#     global extracted_features, current_s_mask
#     extracted_features = []

#     with open(eval_config_path, 'r') as f:
#         eval_cfg = json.load(f)

#     base_cfg_path = eval_cfg["base_model"]["train_config_path"]
#     base_model_path = eval_cfg["base_model"]["model_path"]
#     base_epoch = eval_cfg["base_model"]["epoch"]
    
#     print("\n--- Phase 1: Loading Configuration & Test Dataset ---")
#     with open(base_cfg_path, 'r') as f:
#         train_config = json.load(f)
        
#     cuda = train_config.get("cuda", True)
#     hid_units = train_config.get("hid", 256)
#     num_buckets = train_config.get("num_buckets", 16)
#     use_join_embedding = train_config.get("use_join_embedding", 0)
#     use_single_embedding = train_config.get("use_single_embedding", 0)

#     if use_single_embedding != 1:
#         raise ValueError("Analysis aborted: Model does not use single embedding.")

#     # 加载测试集数据
#     dicts, _, _, _, _, _, test_data, _, test_label_raw = load_and_encode_all_data(train_config)
    
#     table2vec, column2vec, op2vec, join2vec = dicts
#     join_sample_feats = test_data[0][3].shape[0] if use_join_embedding == 1 else 0
#     table_vec_size = len(table2vec)
#     total_sample_feats = test_data[0][0].shape[1]
#     sample_vec_size = total_sample_feats - table_vec_size 
#     predicate_feats = len(column2vec) + len(op2vec) + 1 + num_buckets
#     join_feats = len(join2vec)

#     print("\n--- Phase 2: Initializing Model & Registering Hook ---")
#     model = SetConv(table_vec_size, sample_vec_size, predicate_feats, join_feats, join_sample_feats, hid_units, use_single_embedding)
#     all_epochs_dict = torch.load(base_model_path, map_location="cpu")
#     model.load_state_dict(all_epochs_dict[base_epoch])
    
#     if cuda:
#         model.cuda()
#     model.eval()

#     hook_handle = model.single_emb_mlp1.register_forward_hook(embedding_hook_fn)

#     print("\n--- Phase 3: Extracting Embeddings ---")
#     test_data_loader = DataLoader(test_data, batch_size=1024, shuffle=False)
#     with torch.no_grad():
#         for data_batch in test_data_loader:
#             samples, predicates, joins, join_samples, _, s_mask, p_mask, j_mask = data_batch
#             if cuda:
#                 samples, predicates, joins, join_samples = samples.cuda(), predicates.cuda(), joins.cuda(), join_samples.cuda()
#                 s_mask, p_mask, j_mask = s_mask.cuda(), p_mask.cuda(), j_mask.cuda()
#             current_s_mask = s_mask
#             _ = model(samples, predicates, joins, join_samples, s_mask, p_mask, j_mask)

#     hook_handle.remove()

#     X_embeddings = np.concatenate(extracted_features, axis=0)
#     Y_true_cards = np.array(test_label_raw).astype(np.float64).flatten()

#     print(f"\nLoading baseline prediction metrics from: {qerror_csv_path}")
#     qerror_data = np.loadtxt(qerror_csv_path, delimiter=',')
#     preds_raw = qerror_data[:, 0].astype(np.float64)
#     trues_raw = qerror_data[:, 1].astype(np.float64)
    
#     # 严格对齐校验
#     if len(Y_true_cards) != len(trues_raw):
#         raise ValueError(f"❌ 长度不匹配！DataLoader 样本数 ({len(Y_true_cards)}) != CSV 行数 ({len(trues_raw)})")
    
#     mismatch_mask = np.abs(Y_true_cards - trues_raw) > 1e-2
#     mismatch_count = np.sum(mismatch_mask)
#     if mismatch_count > 0:
#         print(f"\n❌ [CRITICAL ERROR] 数据顺序严重错位！")
#         sys.exit(1)
#     else:
#         print("✅ [SUCCESS] 数据完美对齐！CSV 顺序与 DataLoader 完全一致。")

#     preds_clipped = np.clip(preds_raw, 1.0, None)
#     trues_clipped = np.clip(trues_raw, 1.0, None)
#     all_qerrors = np.maximum(preds_clipped / trues_clipped, trues_clipped / preds_clipped)

#     print("\n--- Phase 4: Filtering Target Density Regions ---")
#     true_labels = np.full_like(Y_true_cards, -1, dtype=int)
#     mask_low = (Y_true_cards >= 1) & (Y_true_cards <= 10)
#     mask_med = (Y_true_cards >= 50) & (Y_true_cards <= 100)
#     mask_high = (Y_true_cards >= 300)
    
#     true_labels[mask_low] = 0   
#     true_labels[mask_med] = 1   
#     true_labels[mask_high] = 2  

#     valid_mask = true_labels != -1
#     X_valid = X_embeddings[valid_mask]
#     Y_valid_true = true_labels[valid_mask]
#     Y_valid_cards = Y_true_cards[valid_mask]
#     Y_valid_qerror = all_qerrors[valid_mask]

#     counts = np.bincount(Y_valid_true)
#     print(f"Target samples filtered -> Total: {len(Y_valid_true)} (Low: {counts[0]}, Med: {counts[1]}, High: {counts[2]})")

#     print("\n--- Phase 7: Generating Dual t-SNE Visualizations with Single Max Q-Error Marked ---")
#     max_plot_points = 3000
#     if len(X_valid) > max_plot_points:
#         print(f"Downsampling visualization data while forcing Max Q-Error outlier inclusion...")
        
#         # 🔥 保底策略：精确定位全局 valid 集合中 Q-Error 绝对值最大的那单个点的索引
#         abs_max_qerror_idx = np.argmax(Y_valid_qerror)
        
#         all_indices = np.arange(len(X_valid))
#         remaining_indices = np.setdiff1d(all_indices, [abs_max_qerror_idx])
#         sampled_remaining = np.random.choice(remaining_indices, max_plot_points - 1, replace=False)
        
#         # 拼接索引，确保这个最大变态点 100% 进入画图池
#         sample_indices = np.append(sampled_remaining, abs_max_qerror_idx)
        
#         X_plot = X_valid[sample_indices]
#         cards_plot = Y_valid_cards[sample_indices]
#         labels_true_plot = Y_valid_true[sample_indices]
#         qerrors_plot = Y_valid_qerror[sample_indices]
#     else:
#         X_plot = X_valid
#         cards_plot = Y_valid_cards
#         labels_true_plot = Y_valid_true
#         qerrors_plot = Y_valid_qerror

#     print("Running t-SNE projection...")
#     tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
#     X_tsne = tsne.fit_transform(X_plot)

#     # 🎯 寻找画图池中唯一的最大值点坐标和数值
#     max_qerror_idx_in_plot = np.argmax(qerrors_plot)
#     max_x = X_tsne[max_qerror_idx_in_plot, 0]
#     max_y = X_tsne[max_qerror_idx_in_plot, 1]
#     max_qerror_value = qerrors_plot[max_qerror_idx_in_plot]
#     print(f"📢 [🎯 锁定目标] 整个画图池中最大 Q-Error 点的原始数值为: {max_qerror_value:.2f}")

#     # ------------------ 图 1：连续真实基数对数热力图 + 唯一样本标记 ------------------
#     plt.figure(figsize=(10, 8), dpi=150)
#     color_values = np.log1p(cards_plot)
#     sc = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=color_values, cmap='viridis', alpha=0.6, edgecolors='none', s=25)
#     cb = plt.colorbar(sc)
#     cb.set_label('Log(True Cardinality + 1)', fontsize=11, fontweight='bold', labelpad=10)
    
#     # 🔴 针对这唯一的一个最大点画大红圈并强制置顶
#     plt.scatter(max_x, max_y, s=180, facecolors='none', edgecolors='red', linewidths=2.5, zorder=5)
#     plt.scatter(max_x, max_y, c=color_values[max_qerror_idx_in_plot], cmap='viridis', s=40, edgecolors='black', linewidths=1.0, zorder=6)
    
#     plt.annotate(f"Max Q-Error: {max_qerror_value:.1f}", (max_x, max_y),
#                  textcoords="offset points", xytext=(0, 15), ha='center',
#                  fontweight='bold', color='red', fontsize=10,
#                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", lw=1.5, alpha=0.9))

#     plt.title('t-SNE Visualization (Continuous True Cardinality)', fontsize=12, fontweight='bold', pad=15)
#     plt.xlabel('t-SNE Dimension 1', fontsize=11)
#     plt.ylabel('t-SNE Dimension 2', fontsize=11)
#     plt.grid(True, linestyle='--', alpha=0.3)
    
#     plot_out_heatmap = "/data2/xuyining/learnedcardinalities/data/stats/results/embedding_true_card_heatmap.png"
#     os.makedirs(os.path.dirname(plot_out_heatmap), exist_ok=True)
#     plt.savefig(plot_out_heatmap, bbox_inches='tight')
#     print(f"Plot 1 (Heatmap with Max Marker) saved as: {plot_out_heatmap}")
#     plt.close()

#     # # ------------------ 图 2：离散真实频率三分类图 + 唯一样本标记 ------------------
#     # plt.figure(figsize=(10, 8), dpi=150)
#     # discrete_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
#     # class_names = ['Low Frequency (1 <= Card <= 10)', 'Medium Frequency (50 <= Card <= 100)', 'High Frequency (Card >= 300)']
    
#     # for class_id in [0, 1, 2]:
#     #     mask_id = (labels_true_plot == class_id)
#     #     plt.scatter(X_tsne[mask_id, 0], X_tsne[mask_id, 1], c=discrete_colors[class_id], label=class_names[class_id], alpha=0.6, edgecolors='none', s=25)

#     # # 🔴 针对同一个最大点在此分类图中画大红圈并依据其真实频段颜色强制置顶
#     # plt.scatter(max_x, max_y, s=180, facecolors='none', edgecolors='red', linewidths=2.5, zorder=5)
    
#     # max_point_class = labels_true_plot[max_qerror_idx_in_plot]
#     # plt.scatter(max_x, max_y, c=discrete_colors[max_point_class], s=40, edgecolors='black', linewidths=1.0, zorder=6)
        
#     # plt.annotate(f"Max Q-Error: {max_qerror_value:.1f}", (max_x, max_y),
#     #              textcoords="offset points", xytext=(0, 15), ha='center',
#     #              fontweight='bold', color='red', fontsize=10,
#     #              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", lw=1.5, alpha=0.9))

#     # plt.title('t-SNE Visualization (Ground Truth Frequency Bounds)', fontsize=12, fontweight='bold', pad=15)
#     # plt.xlabel('t-SNE Dimension 1', fontsize=11)
#     # plt.ylabel('t-SNE Dimension 2', fontsize=11)
#     # plt.legend(loc='best', frameon=True, shadow=True, fontsize=10)
#     # plt.grid(True, linestyle='--', alpha=0.3)
    
#     # plot_out_discrete = "/data2/xuyining/learnedcardinalities/data/stats/results/embedding_ground_truth_3classes.png"
#     # plt.savefig(plot_out_discrete, bbox_inches='tight')
#     # print(f"Plot 2 (Discrete Classes with Max Marker) saved as: {plot_out_discrete}")
#     # plt.close()
#     print("\n============= Analysis Complete =============")

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--eval_config", help="Path to evaluation config JSON file", required=True)
#     parser.add_argument("--qerror_csv", help="Path to the model prediction pred,true CSV file", required=True)
#     args = parser.parse_args()
#     run_kmeans_analysis(args.eval_config, args.qerror_csv)

# if __name__ == "__main__":
#     main()




import os
import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from mscn.data import load_and_encode_all_data
from mscn.model import SetConv

# 存储 hook 捕获的数据
extracted_inputs = []   # 存放输入向量的平均池化结果（全维度）
extracted_outputs = []  # 存放输出向量的平均池化结果（全维度）
current_s_mask = None

# ---------- 输入 hook ----------
def input_hook(module, input_tensor, output_tensor):
    global current_s_mask
    inp = input_tensor[0]  # 输入张量，shape: (batch, num_tables, feature_dim)
    if current_s_mask is not None:
        masked = inp * current_s_mask
        summed = torch.sum(masked, dim=1)               # (batch, feature_dim)
        counts = torch.sum(current_s_mask, dim=1).clamp(min=1.0)  # (batch, 1)
        pooled = summed / counts
    else:
        pooled = torch.mean(inp, dim=1)
    extracted_inputs.append(pooled.cpu().numpy())

# ---------- 输出 hook ----------
def output_hook(module, input_tensor, output_tensor):
    global current_s_mask
    # 对输出应用 LeakyReLU，模拟 MLP 后的激活
    activated = F.leaky_relu(output_tensor, negative_slope=0.01)
    if current_s_mask is not None:
        masked = activated * current_s_mask
        summed = torch.sum(masked, dim=1)
        counts = torch.sum(current_s_mask, dim=1).clamp(min=1.0)
        pooled = summed / counts
    else:
        pooled = torch.mean(activated, dim=1)
    extracted_outputs.append(pooled.cpu().numpy())

def run_tsne_analysis(eval_config_path, qerror_csv_path):
    global current_s_mask, extracted_inputs, extracted_outputs
    extracted_inputs = []
    extracted_outputs = []

    # ---------- 1. 加载配置 ----------
    with open(eval_config_path, 'r') as f:
        eval_cfg = json.load(f)
    base_cfg_path = eval_cfg["base_model"]["train_config_path"]
    base_model_path = eval_cfg["base_model"]["model_path"]
    base_epoch = eval_cfg["base_model"]["epoch"]

    with open(base_cfg_path, 'r') as f:
        train_config = json.load(f)

    cuda = train_config.get("cuda", True)
    hid_units = train_config.get("hid", 256)
    num_buckets = train_config.get("num_buckets", 16)
    use_join_embedding = train_config.get("use_join_embedding", 0)
    use_single_embedding = train_config.get("use_single_embedding", 0)

    if use_single_embedding != 1:
        raise ValueError("Analysis requires use_single_embedding=1")

    # ---------- 2. 加载测试数据 ----------
    dicts, _, _, _, _, _, test_data, _, test_label_raw = load_and_encode_all_data(train_config)
    table2vec, column2vec, op2vec, join2vec = dicts
    join_sample_feats = test_data[0][3].shape[0] if use_join_embedding == 1 else 0
    table_vec_size = len(table2vec)
    total_sample_feats = test_data[0][0].shape[1]
    sample_vec_size = total_sample_feats - table_vec_size
    predicate_feats = len(column2vec) + len(op2vec) + 1 + num_buckets
    join_feats = len(join2vec)

    # ---------- 3. 初始化模型并注册 hook ----------
    model = SetConv(table_vec_size, sample_vec_size, predicate_feats,
                    join_feats, join_sample_feats, hid_units, use_single_embedding)
    all_epochs_dict = torch.load(base_model_path, map_location="cpu")
    model.load_state_dict(all_epochs_dict[base_epoch])
    if cuda:
        model.cuda()
    model.eval()

    hook_in = model.single_emb_mlp1.register_forward_hook(input_hook)
    hook_out = model.single_emb_mlp1.register_forward_hook(output_hook)

    # ---------- 4. 前向传播收集特征 ----------
    print("Extracting input and output embeddings...")
    test_loader = DataLoader(test_data, batch_size=1024, shuffle=False)
    with torch.no_grad():
        for batch in test_loader:
            samples, predicates, joins, join_samples, _, s_mask, p_mask, j_mask = batch
            if cuda:
                samples = samples.cuda()
                predicates = predicates.cuda()
                joins = joins.cuda()
                join_samples = join_samples.cuda()
                s_mask = s_mask.cuda()
                p_mask = p_mask.cuda()
                j_mask = j_mask.cuda()
            current_s_mask = s_mask
            _ = model(samples, predicates, joins, join_samples, s_mask, p_mask, j_mask)

    hook_in.remove()
    hook_out.remove()

    # 特征矩阵 (N, D_full)
    X_in_full = np.concatenate(extracted_inputs, axis=0)
    X_out_full = np.concatenate(extracted_outputs, axis=0)
    # 前1000维位图特征
    X_in_1000 = X_in_full[:, :1000]   # 假设输入维度 >= 1000
    # 第1000维之后的特征（即 Embedding + PCA 部分）
    if X_in_full.shape[1] > 1000:
        X_in_after1000 = X_in_full[:, 1000:]
    else:
        print("Warning: Feature dimension <= 1000, skipping after1000 plot.")
        X_in_after1000 = None

    # 新增切片：Base Embedding (1000~1767) 和 PCA (最后1536维)
    # 根据模型设计，总维度 = 1000 (位图) + 768 (Base) + 1536 (PCA) = 3304
    # 但我们更通用地提取：若维度足够则切片
    if X_in_full.shape[1] >= 1000 + 768:
        X_in_emb = X_in_full[:, 1000:1000+768]
    else:
        X_in_emb = None

    if X_in_full.shape[1] >= 1000 + 768 + 768:
        X_in_pca = X_in_full[:, -768:]   # 取最后768维，等同于从1000+768开始
    else:
        X_in_pca = None

    print(f"X_in_full.shape[1] = {X_in_full.shape[1]}")

    Y_true = np.array(test_label_raw).astype(np.float64).flatten()

    # ---------- 5. 读取 Q-Error CSV ----------
    print(f"Loading Q-Error CSV: {qerror_csv_path}")
    qerror_data = np.loadtxt(qerror_csv_path, delimiter=',')
    preds_raw = qerror_data[:, 0].astype(np.float64)
    trues_raw = qerror_data[:, 1].astype(np.float64)

    if len(Y_true) != len(trues_raw):
        raise ValueError("Length mismatch between DataLoader and CSV")
    if not np.allclose(Y_true, trues_raw, atol=1e-2):
        raise ValueError("Data misalignment!")

    preds = np.clip(preds_raw, 1.0, None)
    trues = np.clip(trues_raw, 1.0, None)
    all_qerrors = np.maximum(preds / trues, trues / preds)

    # 全局最大 Q-Error 索引
    global_max_idx = np.argmax(all_qerrors)
    global_max_qerror = all_qerrors[global_max_idx]
    print(f"Global max Q-Error: {global_max_qerror:.2f} (sample index: {global_max_idx})")

    # ---------- 6. 下采样并固定索引（强制包含最大点） ----------
    max_plot_points = 3000
    N = len(X_in_full)
    if N > max_plot_points:
        all_indices = np.arange(N)
        remaining = np.setdiff1d(all_indices, [global_max_idx])
        sampled = np.random.choice(remaining, max_plot_points - 1, replace=False)
        plot_idx = np.append(sampled, global_max_idx)
    else:
        plot_idx = np.arange(N)

    # ---------- 7. 通用绘图函数 ----------
    def plot_tsne(X_features, cards, qerrors, title, save_path,
                  plot_idx=None, print_neighbors=False):
        """
        X_features: 待降维的特征矩阵 (N_sample, feature_dim)
        cards, qerrors: 对应的基数和Q-Error
        plot_idx: 如果提供，则用于输出全局样本索引
        print_neighbors: 是否打印距离最大点最近的5个点的全局索引
        """
        # t-SNE 降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        X_2d = tsne.fit_transform(X_features)

        color_vals = np.log1p(cards)
        plt.figure(figsize=(10, 8), dpi=150)
        sc = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=color_vals,
                         cmap='viridis', alpha=0.6, edgecolors='none', s=25)
        cb = plt.colorbar(sc)
        cb.set_label('Log(True Cardinality + 1)', fontsize=11, fontweight='bold')

        # 标记最大 Q-Error 点
        max_local = np.argmax(qerrors)
        xm, ym = X_2d[max_local, 0], X_2d[max_local, 1]
        plt.scatter(xm, ym, s=200, facecolors='none', edgecolors='red',
                    linewidths=2.5, zorder=5)
        plt.scatter(xm, ym, c=color_vals[max_local], cmap='viridis',
                    s=40, edgecolors='black', linewidths=1.0, zorder=6)
        plt.annotate(f"Max Q-Error: {qerrors[max_local]:.1f}\nCard: {cards[max_local]:.0f}",
                     (xm, ym), textcoords="offset points", xytext=(0, 18),
                     ha='center', fontweight='bold', color='red', fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", lw=1.5, alpha=0.9))

        plt.title(title, fontsize=12, fontweight='bold')
        plt.xlabel('t-SNE dim 1')
        plt.ylabel('t-SNE dim 2')
        plt.grid(True, linestyle='--', alpha=0.3)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()

        # ---------- 最近邻打印（仅当启用） ----------
        if print_neighbors and plot_idx is not None:
            max_point_2d = X_2d[max_local].reshape(1, -1)
            distances = np.linalg.norm(X_2d - max_point_2d, axis=1)
            sorted_idx = np.argsort(distances)
            neighbor_local_indices = sorted_idx[1:6]  # 前5个邻居（排除自身）
            neighbor_global_indices = plot_idx[neighbor_local_indices]
            neighbor_distances = distances[neighbor_local_indices]
            print(f"\n--- For {os.path.basename(save_path)} ---")
            print(f"Nearest 5 neighbor indices of max Q-Error point (global sample indices):")
            print(neighbor_global_indices.tolist())
            print("Corresponding 2D Euclidean distances:", neighbor_distances.round(4))

    # ---------- 8. 生成六张图 ----------
    # 图1：输入全维度
    print("Running t-SNE on full input...")
    plot_tsne(X_in_full[plot_idx],
              cards=Y_true[plot_idx],
              qerrors=all_qerrors[plot_idx],
              title="t-SNE of MLP1 Input (Full Features, Pooled)",
              save_path="/data2/xuyining/learnedcardinalities/data/stats/results/embedding_mlp1_input_full.png")

    # 图2：输入前1000维（位图部分）
    print("Running t-SNE on first 1000 dims of input...")
    plot_tsne(X_in_1000[plot_idx],
              cards=Y_true[plot_idx],
              qerrors=all_qerrors[plot_idx],
              title="t-SNE of MLP1 Input (First 1000 Dims, Pooled)",
              save_path="/data2/xuyining/learnedcardinalities/data/stats/results/embedding_mlp1_input_1000.png")

    # 图3：输出全维度
    print("Running t-SNE on output...")
    plot_tsne(X_out_full[plot_idx],
              cards=Y_true[plot_idx],
              qerrors=all_qerrors[plot_idx],
              title="t-SNE of MLP1 Output (LeakyReLU, Pooled)",
              save_path="/data2/xuyining/learnedcardinalities/data/stats/results/embedding_mlp1_output.png")

    # 图4：第1000维之后的部分（Embedding + PCA）—— 开启最近邻输出
    if X_in_after1000 is not None and X_in_after1000.shape[1] > 0:
        print("Running t-SNE on after-1000 dims of input (embedding part)...")
        plot_tsne(X_in_after1000[plot_idx],
                  cards=Y_true[plot_idx],
                  qerrors=all_qerrors[plot_idx],
                  title="t-SNE of MLP1 Input (After First 1000 Dims, Pooled)",
                  save_path="/data2/xuyining/learnedcardinalities/data/stats/results/embedding_mlp1_input_after1000.png",
                  plot_idx=plot_idx,
                  print_neighbors=True)   # 仅此图输出最近邻
    else:
        print("Skipped after-1000 plot due to insufficient dimensions.")

    # 图5：第1000维之后的768维（Base Embedding）
    if X_in_emb is not None:
        print("Running t-SNE on embedding part (dims 1000-1767)...")
        plot_tsne(X_in_emb[plot_idx],
                  cards=Y_true[plot_idx],
                  qerrors=all_qerrors[plot_idx],
                  title="t-SNE of MLP1 Input (Base Embedding: dims 1000-1767, Pooled)",
                  save_path="/data2/xuyining/learnedcardinalities/data/stats/results/embedding_mlp1_input_emb.png")
    else:
        print("Skipped embedding part plot due to insufficient dimensions.")

    # 图6：最后1536维（PCA Projection）
    if X_in_pca is not None:
        print("Running t-SNE on PCA part (last 768 dims)...")
        plot_tsne(X_in_pca[plot_idx],
                  cards=Y_true[plot_idx],
                  qerrors=all_qerrors[plot_idx],
                  title="t-SNE of MLP1 Input (PCA Projection: last 768 dims, Pooled)",
                  save_path="/data2/xuyining/learnedcardinalities/data/stats/results/embedding_mlp1_input_pca.png")
    else:
        print("Skipped PCA part plot due to insufficient dimensions.")

    print("\nAll six plots generated successfully.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_config", required=True, help="Path to eval config JSON")
    parser.add_argument("--qerror_csv", required=True, help="CSV with pred,true columns")
    args = parser.parse_args()
    run_tsne_analysis(args.eval_config, args.qerror_csv)

if __name__ == "__main__":
    main()