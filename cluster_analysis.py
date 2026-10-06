import os
import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from mscn.data import load_and_encode_all_data
from mscn.model import SetConv

# 全局容器，用于暂存钩子拦截到的特征向量
extracted_features = []

def embedding_hook_fn(module, input_tensor, output_tensor):
    """Forward hook to capture embeddings from single_emb_mlp1 layer."""
    activated = F.leaky_relu(output_tensor, negative_slope=0.01)
    pooled = torch.mean(activated, dim=1)  # Average pooling over tables
    extracted_features.append(pooled.cpu().numpy())

def run_kmeans_analysis(eval_config_path):
    global extracted_features
    extracted_features = []  # Reset container

    with open(eval_config_path, 'r') as f:
        eval_cfg = json.load(f)

    # Base model configuration
    base_cfg_path = eval_cfg["base_model"]["train_config_path"]
    base_model_path = eval_cfg["base_model"]["model_path"]
    base_epoch = eval_cfg["base_model"]["epoch"]
    
    print("\n--- Phase 1: Loading Configuration & Test Dataset ---")
    with open(base_cfg_path, 'r') as f:
        train_config = json.load(f)
        
    cuda = train_config.get("cuda", True)
    hid_units = train_config.get("hid", 256)
    num_buckets = train_config.get("num_buckets", 16)
    use_join_embedding = train_config.get("use_join_embedding", 0)
    use_single_embedding = train_config.get("use_single_embedding", 0)

    if use_single_embedding != 1:
        raise ValueError("Analysis aborted: Model does not use single embedding.")

    # Load encoded data (Using TEST set)
    dicts, _, _, _, _, _, test_data, _, test_label_raw = load_and_encode_all_data(train_config)
    
    table2vec, column2vec, op2vec, join2vec = dicts
    join_sample_feats = test_data[0][3].shape[0] if use_join_embedding == 1 else 0
    table_vec_size = len(table2vec)
    total_sample_feats = test_data[0][0].shape[1]
    sample_vec_size = total_sample_feats - table_vec_size 
    predicate_feats = len(column2vec) + len(op2vec) + 1 + num_buckets
    join_feats = len(join2vec)

    print("\n--- Phase 2: Initializing Model & Registering Hook ---")
    model = SetConv(table_vec_size, sample_vec_size, predicate_feats, join_feats, join_sample_feats, hid_units, use_single_embedding)
    all_epochs_dict = torch.load(base_model_path, map_location="cpu")
    model.load_state_dict(all_epochs_dict[base_epoch])
    
    if cuda:
        model.cuda()
    model.eval()

    hook_handle = model.single_emb_mlp1.register_forward_hook(embedding_hook_fn)

    print("\n--- Phase 3: Extracting Embeddings from Test Set ---")
    test_data_loader = DataLoader(test_data, batch_size=1024, shuffle=False)
    
    with torch.no_grad():
        for data_batch in test_data_loader:
            samples, predicates, joins, join_samples, _, s_mask, p_mask, j_mask = data_batch
            if cuda:
                samples, predicates, joins, join_samples = samples.cuda(), predicates.cuda(), joins.cuda(), join_samples.cuda()
                s_mask, p_mask, j_mask = s_mask.cuda(), p_mask.cuda(), j_mask.cuda()
            _ = model(samples, predicates, joins, join_samples, s_mask, p_mask, j_mask)

    hook_handle.remove()

    X_embeddings = np.concatenate(extracted_features, axis=0)
    Y_true_cards = np.array(test_label_raw).astype(np.float64).flatten()
    print(f"Extracted embedding matrix shape: {X_embeddings.shape}")

    print("\n--- Phase 4: Filtering Target Density Regions ---")
    # Define ground truth density labels for verification
    true_labels = np.full_like(Y_true_cards, -1, dtype=int)
    mask_low = (Y_true_cards >= 1) & (Y_true_cards <= 10)
    mask_med = (Y_true_cards >= 50) & (Y_true_cards <= 100)
    mask_high = (Y_true_cards >= 300)
    
    true_labels[mask_low] = 0   # Ground Truth 0: Low Density
    true_labels[mask_med] = 1   # Ground Truth 1: Medium Density
    true_labels[mask_high] = 2  # Ground Truth 2: High Density

    valid_mask = true_labels != -1
    X_valid = X_embeddings[valid_mask]
    Y_valid_true = true_labels[valid_mask]
    Y_valid_cards = Y_true_cards[valid_mask]
    
    counts = np.bincount(Y_valid_true)
    print(f"Target samples filtered -> Total: {len(Y_valid_true)} (Low: {counts[0]}, Med: {counts[1]}, High: {counts[2]})")

    print("\n--- Phase 5: Running Unsupervised KMeans Clustering (K=3) ---")
    # KMeans is completely blind to Y_valid_true
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    kmeans_labels = kmeans.fit_predict(X_valid)
    print("KMeans clustering finished successfully.")

    print("\n--- Phase 6: Evaluating KMeans Clustering Metrics ---")
    # 1. Standard Clustering Quality (Using KMeans prediction)
    sil = silhouette_score(X_valid, kmeans_labels)
    db_idx = davies_bouldin_score(X_valid, kmeans_labels)
    ch_idx = calinski_harabasz_score(X_valid, kmeans_labels)

    # 2. Alignment with Ground Truth (How well does KMeans match reality?)
    ari = adjusted_rand_score(Y_valid_true, kmeans_labels)
    nmi = normalized_mutual_info_score(Y_valid_true, kmeans_labels)

    print(f" Internal Clustering Quality (Based on KMeans Clusters):")
    print(f"  -> Silhouette Score     : {sil:.4f}  (Higher means better cluster separation)")
    print(f"  -> Davies-Bouldin Index  : {db_idx:.4f} (Lower means less cluster overlap)")
    print(f"  -> Calinski-Harabasz Max : {ch_idx:.2f} (Higher means tighter clusters)")
    print(f" External Alignment Metrics (KMeans vs Ground Truth Density):")
    print(f"  -> Adjusted Rand Index (ARI)       : {ari:.4f} (Closer to 1.0 = Perfect match with density regions)")
    print(f"  -> Normalized Mutual Info (NMI)    : {nmi:.4f} (Closer to 1.0 = High structural correlation)")

    # print("\n--- Phase 7: Generating 2D t-SNE Plot of KMeans Clusters ---")
    # max_plot_points = 3000
    # if len(kmeans_labels) > max_plot_points:
    #     print(f"Downsampling visualization data to {max_plot_points} points...")
    #     sample_indices = np.random.choice(len(kmeans_labels), max_plot_points, replace=False)
    #     X_plot = X_valid[sample_indices]
    #     labels_plot = kmeans_labels[sample_indices]
    # else:
    #     X_plot = X_valid
    #     labels_plot = kmeans_labels

    # print("Running t-SNE projection...")
    # tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    # X_tsne = tsne.fit_transform(X_plot)

    # plt.figure(figsize=(10, 8), dpi=150)
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # for cluster_id in [0, 1, 2]:
    #     mask_id = labels_plot == cluster_id
    #     plt.scatter(X_tsne[mask_id, 0], X_tsne[mask_id, 1], 
    #                 c=colors[cluster_id], label=f'KMeans Cluster {cluster_id}', 
    #                 alpha=0.6, edgecolors='none', s=25)

    # plt.title('t-SNE Visualization of KMeans Clusters in MLP Embedding Space', fontsize=12, fontweight='bold', pad=15)
    # plt.xlabel('t-SNE Dimension 1', fontsize=11)
    # plt.ylabel('t-SNE Dimension 2', fontsize=11)
    # plt.legend(loc='best', frameon=True, shadow=True, fontsize=10)
    # plt.grid(True, linestyle='--', alpha=0.5)
    
    # plot_out = "/data2/xuyining/learnedcardinalities/data/zipf/results/kmeans_embedding_clustering.png"
    # plt.savefig(plot_out, bbox_inches='tight')
    # print(f"Plot saved successfully as: {plot_out}")
    # print("\n============= Analysis Complete =============")

    print("\n--- Phase 7: Generating 2D t-SNE Plot with True-Card Heatmap ---")
    max_plot_points = 3000
    if len(X_valid) > max_plot_points:
        print(f"Downsampling visualization data to {max_plot_points} points...")
        sample_indices = np.random.choice(len(X_valid), max_plot_points, replace=False)
        X_plot = X_valid[sample_indices]
        cards_plot = Y_valid_cards[sample_indices]
        labels_true_plot = Y_valid_true[sample_indices]
    else:
        X_plot = X_valid
        cards_plot = Y_valid_cards
        labels_true_plot = Y_valid_true

    print("Running t-SNE projection...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_tsne = tsne.fit_transform(X_plot)

    # ------------------ 图 1：连续真实基数对数热力图 ------------------
    plt.figure(figsize=(10, 8), dpi=150)
    color_values = np.log1p(cards_plot)
    sc = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                     c=color_values, cmap='viridis', 
                     alpha=0.7, edgecolors='none', s=25)
    cb = plt.colorbar(sc)
    cb.set_label('Log(True Cardinality + 1)', fontsize=11, fontweight='bold', labelpad=10)
    cb.ax.tick_params(labelsize=10)

    plt.title('t-SNE Visualization of Embedding Space (Continuous True Cardinality)', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('t-SNE Dimension 1', fontsize=11)
    plt.ylabel('t-SNE Dimension 2', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plot_out_heatmap = "/data2/xuyining/learnedcardinalities/data/zipf/results/embedding_true_card_heatmap.png"
    plt.savefig(plot_out_heatmap, bbox_inches='tight')
    print(f"Plot 1 (Heatmap) saved as: {plot_out_heatmap}")
    plt.close()

    # ------------------ 图 2：离散真实频率三分类图------------------
    plt.figure(figsize=(10, 8), dpi=150)
    
    discrete_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    class_names = ['Low Frequency (1 <= Card <= 10)', 
                   'Medium Frequency (50 <= Card <= 100)', 
                   'High Frequency (Card >= 300)']
    
    for class_id in [0, 1, 2]:
        mask_id = (labels_true_plot == class_id)
        plt.scatter(X_tsne[mask_id, 0], X_tsne[mask_id, 1], 
                    c=discrete_colors[class_id], 
                    label=class_names[class_id], 
                    alpha=0.7, edgecolors='none', s=25)

    plt.title('t-SNE Visualization of Embedding Space (Ground Truth Frequency Bounds)', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('t-SNE Dimension 1', fontsize=11)
    plt.ylabel('t-SNE Dimension 2', fontsize=11)
    plt.legend(loc='best', frameon=True, shadow=True, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plot_out_discrete = "/data2/xuyining/learnedcardinalities/data/zipf/results/embedding_ground_truth_3classes.png"
    plt.savefig(plot_out_discrete, bbox_inches='tight')
    print(f"Plot 2 (Discrete Classes) saved as: {plot_out_discrete}")
    plt.close()

    print("\n============= Analysis Complete =============")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_config", help="Path to evaluation config JSON file", required=True)
    args = parser.parse_args()
    run_kmeans_analysis(args.eval_config)

if __name__ == "__main__":
    main()

