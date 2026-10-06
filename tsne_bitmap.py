# import os
# import argparse
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE

# # =============== ⚙️ 请在此处配置文件路径 (可被命令行参数覆盖) ===============
# BITMAP_PATH = "/data2/xuyining/learnedcardinalities/data/zipf/test.bitmaps"
# TEST_CSV_PATH = "/data2/xuyining/learnedcardinalities/data/zipf/test.csv"
# NUM_MATERIALIZED_SAMPLES = 1000  # 采样样本数

# # 统一规划输出路径
# PLOT_HEATMAP_OUT = "/data2/xuyining/learnedcardinalities/data/zipf/plots/bitmap_true_card_heatmap.png"
# PLOT_DISCRETE_OUT = "/data2/xuyining/learnedcardinalities/data/zipf/plots/bitmap_ground_truth_3classes.png"
# # =========================================================================

# def load_bitmaps(filepath, num_materialized_samples):
#     """
#     从 .bitmaps 文件加载 bitmap，假定每个查询只有一张表（num_bitmaps_curr_query == 1）。
#     返回 numpy 数组，形状 (num_queries, bitmap_dim)。
#     """
#     num_bytes_per_bitmap = (num_materialized_samples + 7) // 8
#     bitmap_dim = num_bytes_per_bitmap * 8
#     bitmaps_all = []

#     print(f"Opening binary bitmap file: {filepath}")
#     with open(filepath, 'rb') as f:
#         while True:
#             four_bytes = f.read(4)
#             if not four_bytes:
#                 break
#             num_bitmaps_curr = int.from_bytes(four_bytes, byteorder='little')
            
#             bitmaps = np.empty((num_bitmaps_curr, bitmap_dim), dtype=np.uint8)
#             for j in range(num_bitmaps_curr):
#                 bitmap_bytes = f.read(num_bytes_per_bitmap)
#                 if not bitmap_bytes:
#                     raise RuntimeError(f"Unexpected end of file while reading bitmap {j}")
#                 bitmaps[j] = np.unpackbits(np.frombuffer(bitmap_bytes, dtype=np.uint8))
            
#             # 单表场景，默认取第一个 table 的位图
#             bitmaps_all.append(bitmaps[0])
            
#     return np.array(bitmaps_all, dtype=np.float32)


# def load_labels_from_csv(filepath):
#     """
#     从 CSV 文件加载真实基数。
#     格式：无表头，每行以 '#' 分割，最后一部分为基数数值。
#     """
#     labels = []
#     with open(filepath, 'r') as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             card_str = line.split('#')[-1].strip()
#             try:
#                 card = float(card_str)
#             except ValueError:
#                 raise ValueError(f"无法解析基数: '{card_str}'，来自行: {line}")
#             labels.append(card)
#     return np.array(labels, dtype=np.float64)


# def run_bitmap_analysis(bitmap_path, csv_path, num_samples, out_heatmap, out_discrete):
#     print("\n--- Phase 1: Loading Bitmaps and Labels ---")
#     X_bitmaps = load_bitmaps(bitmap_path, num_samples)
#     print(f"Successfully loaded bitmap matrix. Shape: {X_bitmaps.shape}")

#     Y_true_cards = load_labels_from_csv(csv_path)
#     print(f"Successfully loaded true cardinalities. Total count: {len(Y_true_cards)}")

#     if len(X_bitmaps) != len(Y_true_cards):
#         raise ValueError(
#             f"❌ 严重对齐错误：Bitmap 样本数 ({len(X_bitmaps)}) 与标签数 ({len(Y_true_cards)}) 不一致！"
#         )

#     # ---------- Phase 2: 过滤目标密度区域（与 embedding 分析严格保持一致）----------
#     print("\n--- Phase 2: Filtering Target Density Regions ---")
#     true_labels = np.full_like(Y_true_cards, -1, dtype=int)
#     mask_low = (Y_true_cards >= 1) & (Y_true_cards <= 10)
#     mask_med = (Y_true_cards >= 50) & (Y_true_cards <= 100)
#     mask_high = (Y_true_cards >= 300)
    
#     true_labels[mask_low] = 0   # Low
#     true_labels[mask_med] = 1   # Med
#     true_labels[mask_high] = 2  # High

#     valid_mask = true_labels != -1
#     X_valid = X_bitmaps[valid_mask]
#     Y_valid_true = true_labels[valid_mask]
#     Y_valid_cards = Y_true_cards[valid_mask]

#     print(f"Target samples filtered -> Total valid: {len(Y_valid_true)}")
#     print(f"  -> Low Frequency (0): {np.sum(Y_valid_true == 0)}")
#     print(f"  -> Med Frequency (1): {np.sum(Y_valid_true == 1)}")
#     print(f"  -> High Frequency (2): {np.sum(Y_valid_true == 2)}")

#     # ---------- Phase 3: t-SNE 降维 ----------
#     print("\n--- Phase 3: t-SNE Projection ---")
#     max_plot_points = 3000
#     if len(X_valid) > max_plot_points:
#         print(f"Downsampling visualization data to {max_plot_points} points (Fixing seed=42 for consistency)...")
#         np.random.seed(42)  # 固定随机种子，使得采样样本点与之前的图能最大限度对齐
#         sample_indices = np.random.choice(len(X_valid), max_plot_points, replace=False)
#         X_plot = X_valid[sample_indices]
#         cards_plot = Y_valid_cards[sample_indices]
#         labels_true_plot = Y_valid_true[sample_indices]
#     else:
#         X_plot = X_valid
#         cards_plot = Y_valid_cards
#         labels_true_plot = Y_valid_true

#     print("Running t-SNE projection on Bitmaps (This may take a few seconds)...")
#     tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
#     X_tsne = tsne.fit_transform(X_plot)

#     # ---------- Phase 4: 绘图层 ----------
#     print("\n--- Phase 4: Plotting and Saving Figures ---")
    
#     # 确保输出目录一定存在，防止 plt.savefig 报错
#     for path in [out_heatmap, out_discrete]:
#         parent_dir = os.path.dirname(path)
#         if parent_dir:
#             os.makedirs(parent_dir, exist_ok=True)

#     # 图1：连续真实基数对数热力图
#     plt.figure(figsize=(10, 8), dpi=150)
#     color_values = np.log1p(cards_plot)
#     sc = plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
#                      c=color_values, cmap='viridis',
#                      alpha=0.6, edgecolors='none', s=25)
#     cb = plt.colorbar(sc)
#     cb.set_label('Log(True Cardinality + 1)', fontsize=11, fontweight='bold', labelpad=10)
#     cb.ax.tick_params(labelsize=10)

#     plt.title('t-SNE Visualization of Bitmap Space (Continuous True Cardinality)',
#               fontsize=12, fontweight='bold', pad=15)
#     plt.xlabel('t-SNE Dimension 1', fontsize=11)
#     plt.ylabel('t-SNE Dimension 2', fontsize=11)
#     plt.grid(True, linestyle='--', alpha=0.3)
#     plt.savefig(out_heatmap, bbox_inches='tight')
#     print(f"📊 Plot 1 (Heatmap) successfully saved to: {out_heatmap}")
#     plt.close()

#     # 图2：离散真实频率三分类图
#     plt.figure(figsize=(10, 8), dpi=150)
#     discrete_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
#     class_names = ['Low Frequency (1 <= Card <= 10)',
#                    'Medium Frequency (50 <= Card <= 100)',
#                    'High Frequency (Card >= 300)']

#     for class_id in [0, 1, 2]:
#         mask_id = (labels_true_plot == class_id)
#         plt.scatter(X_tsne[mask_id, 0], X_tsne[mask_id, 1],
#                     c=discrete_colors[class_id],
#                     label=class_names[class_id],
#                     alpha=0.6, edgecolors='none', s=25)

#     plt.title('t-SNE Visualization of Bitmap Space (Ground Truth Frequency Bounds)',
#               fontsize=12, fontweight='bold', pad=15)
#     plt.xlabel('t-SNE Dimension 1', fontsize=11)
#     plt.ylabel('t-SNE Dimension 2', fontsize=11)
#     plt.legend(loc='best', frameon=True, shadow=True, fontsize=10)
#     plt.grid(True, linestyle='--', alpha=0.3)
#     plt.savefig(out_discrete, bbox_inches='tight')
#     print(f"📊 Plot 2 (Discrete Classes) successfully saved to: {out_discrete}")
#     plt.close()

#     print("\n============= Bitmap Analysis Complete =============")


# def main():
#     parser = argparse.ArgumentParser(description="Bitmap t-SNE visualization")
#     parser.add_argument("--bitmap_path", type=str, default=BITMAP_PATH,
#                         help="Path to .bitmaps file")
#     parser.add_argument("--test_csv_path", type=str, default=TEST_CSV_PATH,
#                         help="Path to CSV file with true cardinalities")
#     parser.add_argument("--num_samples", type=int, default=NUM_MATERIALIZED_SAMPLES,
#                         help="Number of materialized samples")
#     parser.add_argument("--out_heatmap", type=str, default=PLOT_HEATMAP_OUT,
#                         help="Output path for heatmap png")
#     parser.add_argument("--out_discrete", type=str, default=PLOT_DISCRETE_OUT,
#                         help="Output path for 3-classes discrete png")
#     args = parser.parse_args()

#     run_bitmap_analysis(
#         bitmap_path=args.bitmap_path,
#         csv_path=args.test_csv_path,
#         num_samples=args.num_samples,
#         out_heatmap=args.out_heatmap,
#         out_discrete=args.out_discrete
#     )

# if __name__ == "__main__":
#     main()


import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# =============== ⚙️ 请在此处配置文件路径 ===============
BITMAP_PATH = "/data2/xuyining/learnedcardinalities/data/zipf/test-qa.bitmaps"
TEST_CSV_PATH = "/data2/xuyining/learnedcardinalities/data/zipf/test.csv"
NUM_MATERIALIZED_SAMPLES = 500
N_CLUSTERS = 3  # 想聚成多少个簇（建议 3 到 5 个）

OUTPUT_DIR = "/data2/xuyining/learnedcardinalities/data/zipf/plots/"
# ===================================================

def load_bitmaps(filepath, num_materialized_samples):
    num_bytes_per_bitmap = (num_materialized_samples + 7) // 8
    bitmap_dim = num_bytes_per_bitmap * 8
    bitmaps_all = []
    with open(filepath, 'rb') as f:
        while True:
            four_bytes = f.read(4)
            if not four_bytes:
                break
            num_bitmaps_curr = int.from_bytes(four_bytes, byteorder='little')
            bitmaps = np.empty((num_bitmaps_curr, bitmap_dim), dtype=np.uint8)
            for j in range(num_bitmaps_curr):
                bitmap_bytes = f.read(num_bytes_per_bitmap)
                bitmaps[j] = np.unpackbits(np.frombuffer(bitmap_bytes, dtype=np.uint8))
            bitmaps_all.append(bitmaps[0])
    return np.array(bitmaps_all, dtype=np.float32)

def load_labels_from_csv(filepath):
    labels = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            card_str = line.split('#')[-1].strip()
            labels.append(float(card_str))
    return np.array(labels, dtype=np.float64)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 加载数据
    X_bitmaps = load_bitmaps(BITMAP_PATH, NUM_MATERIALIZED_SAMPLES)
    Y_cards = load_labels_from_csv(TEST_CSV_PATH)
    
    print(f"Loaded {len(X_bitmaps)} queries.")

    # 2. 对 Bitmap 运行 K-Means 聚类
    print(f"\n🚀 Running K-Means on Bitmaps (K={N_CLUSTERS})...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_transform(X_bitmaps).argmin(axis=1) # 拿到每个样本的 Cluster ID

    # 3. 核心画图：绘制基数分布小提琴图（Violin Plot）
    print("📊 Plotting Cardinality Distribution per Cluster...")
    plt.figure(figsize=(10, 6), dpi=150)
    
    # 使用 Log(True Card + 1) 让分布更易读
    log_cards = np.log1p(Y_cards)
    
    # 绘制小提琴图
    sns.violinplot(x=cluster_labels, y=log_cards, hue=cluster_labels, palette="Set2", legend=False)
    # 叠加一层箱线图，看清中位数
    sns.boxplot(x=cluster_labels, y=log_cards, width=0.1, color="white", fliersize=0)
    
    plt.title(f'True Cardinality Distribution within each Bitmap Cluster (K={N_CLUSTERS})', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Bitmap K-Means Cluster ID', fontsize=11, fontweight='bold')
    plt.ylabel('Log(True Cardinality + 1)', fontsize=11, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    violin_out = os.path.join(OUTPUT_DIR, "bitmap_cluster_card_violin.png")
    plt.savefig(violin_out, bbox_inches='tight')
    print(f"🎯 Success! Violin plot saved to: {violin_out}")
    plt.close()

    # 4. 辅助画图：降维看看这些簇在空间上的分布
    print("\n🌀 Running t-SNE for Cluster Spatial Visualization...")
    # 限制点数防止 t-SNE 太慢
    max_pts = 2000
    if len(X_bitmaps) > max_pts:
        np.random.seed(42)
        idx = np.random.choice(len(X_bitmaps), max_pts, replace=False)
        X_tsne_in = X_bitmaps[idx]
        cluster_plot = cluster_labels[idx]
    else:
        X_tsne_in = X_bitmaps
        cluster_plot = cluster_labels

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_tsne = tsne.fit_transform(X_tsne_in)

    plt.figure(figsize=(10, 8), dpi=150)
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_plot, cmap='tab10', alpha=0.6, s=20)
    plt.legend(*scatter.legend_elements(), title="Cluster ID", loc="best", frameon=True, shadow=True)
    plt.title('t-SNE Space Partitioned by Bitmap K-Means Clusters', fontsize=12, fontweight='bold', pad=15)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    scatter_out = os.path.join(OUTPUT_DIR, "bitmap_cluster_tsne_space.png")
    plt.savefig(scatter_out, bbox_inches='tight')
    print(f"🎯 Success! Cluster Scatter plot saved to: {scatter_out}")
    plt.close()

if __name__ == "__main__":
    main()