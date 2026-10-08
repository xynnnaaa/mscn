# import os
# import torch
# import numpy as np

# # ==================== 用户配置区域 ====================
# # 基础预测文件（必填），每行格式：pred,true（无表头）
# base_csv_path = "/data2/xuyining/learnedcardinalities/data/imdb/results/bitmap-rs.csv"

# # 输出合并后的 CSV 文件路径
# output_csv_path = "/data2/xuyining/learnedcardinalities/data/imdb/results/bitmap-qa-merge.csv"

# # 可选版本列表：每个元素为 (预测CSV路径, 命中状态文件路径)
# # 命中状态文件可以是 .pt（torch tensor）或 .npy（numpy array），
# # 内容应为 0/1 的向量，1 表示该查询使用该版本的预测值替换当前值。
# versions = [
#     ("/data2/xuyining/learnedcardinalities/data/imdb/results/bitmap-qa.csv",
#      "/data2/xuyining/Sampler/single_table/workloads/imdb/new_model/embedding/test_hit_status.pt")
#     # ("/data2/xuyining/learnedcardinalities/data/imdb/results/full.csv",
#     #  "/data2/xuyining/Sampler/join_sampling/workloads/imdb/embedding/hit_status_test.pt")
# ]
# # ====================================================


# def load_csv_predictions(csv_path):
#     """读取 CSV 文件，返回 (preds, trues) 两个 numpy 数组"""
#     preds = []
#     trues = []
#     with open(csv_path, 'r') as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             parts = line.split(',')
#             if len(parts) != 2:
#                 raise ValueError(f"Invalid CSV line (expected 2 values): {line}")
#             pred, true = float(parts[0]), float(parts[1])
#             preds.append(pred)
#             trues.append(true)
#     return np.array(preds), np.array(trues)


# def load_hit_status(status_path):
#     """加载命中状态文件，支持 .pt 和 .npy 格式"""
#     if status_path.endswith('.pt'):
#         data = torch.load(status_path, map_location='cpu')
#         if isinstance(data, torch.Tensor):
#             data = data.numpy()
#         return data.flatten()
#     elif status_path.endswith('.npy'):
#         return np.load(status_path).flatten()
#     else:
#         raise ValueError("Hit status file must have extension .pt or .npy")


# def print_qerror_stats(qerror):
#     """打印 Q‑error 分位数统计"""
#     print("Median: {:.4f}".format(np.median(qerror)))
#     print("80th percentile: {:.4f}".format(np.percentile(qerror, 80)))
#     print("90th percentile: {:.4f}".format(np.percentile(qerror, 90)))
#     print("95th percentile: {:.4f}".format(np.percentile(qerror, 95)))
#     print("99th percentile: {:.4f}".format(np.percentile(qerror, 99)))
#     print("Max: {:.4f}".format(np.max(qerror)))
#     print("Mean: {:.4f}".format(np.mean(qerror)))


# def main():
#     # 1. 加载基础预测
#     print(f"Loading base predictions from: {base_csv_path}")
#     base_preds, base_trues = load_csv_predictions(base_csv_path)
#     n_queries = len(base_preds)
#     print(f"Base: {n_queries} queries loaded.")

#     # 2. 初始化最终预测为 base 的副本
#     final_preds = base_preds.copy()

#     # 3. 按顺序应用每个版本
#     all_prev_hit = np.ones(n_queries, dtype=bool)
#     for idx, (csv_path, status_path) in enumerate(versions):
#         print(f"\nApplying version {idx+1}:")
#         print(f"  Prediction CSV: {csv_path}")
#         print(f"  Hit status: {status_path}")

#         # 检查文件是否存在
#         if not os.path.isfile(csv_path):
#             print(f"  Warning: CSV file not found, skipping.")
#             continue
#         if not os.path.isfile(status_path):
#             print(f"  Warning: Hit status file not found, skipping.")
#             continue

#         # 加载版本预测
#         preds_ver, trues_ver = load_csv_predictions(csv_path)
#         if len(preds_ver) != n_queries:
#             raise ValueError(f"Version {idx+1} predictions length ({len(preds_ver)}) != base length ({n_queries})")
#         # 检查真实值是否一致（可选项，这里仅警告）
#         if not np.allclose(trues_ver, base_trues, rtol=1e-6, atol=1e-6):
#             print(f"  Warning: True values in version CSV differ from base (ignoring true values).")

#         # 加载命中状态
#         hit = load_hit_status(status_path)
#         if len(hit) != n_queries:
#             raise ValueError(f"Version {idx+1} hit status length ({len(hit)}) != base length ({n_queries})")

#         hit_mask = np.asarray(hit).flatten() == 1

#         # 只有当前版本命中，且所有前面版本都命中时才替换当前版本
#         replace_mask = hit_mask & all_prev_hit
#         all_prev_hit = all_prev_hit & hit_mask

#         replaced_count = int(np.count_nonzero(replace_mask))
#         if replaced_count > 0:
#             final_preds[replace_mask] = preds_ver[replace_mask]
#             print(f"  Replaced {replaced_count} queries.")
#         else:
#             print("  No replacements.")

#     # 4. 保存最终合并结果
#     print(f"\nSaving merged predictions to: {output_csv_path}")
#     os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
#     with open(output_csv_path, 'w') as f:
#         for p, t in zip(final_preds, base_trues):
#             f.write(f"{p:.6f},{t:.6f}\n")   # 保留足够精度，也可用 str

#     # 5. 计算并输出 Q‑error 统计
#     print("\n=== Final Q-error Statistics ===")
#     pred = final_preds.astype(np.float64)
#     true = np.array(base_trues).astype(np.float64)
#     # 避免除零，加小 epsilon
#     qerror = np.maximum(pred / (true + 1e-8), true / (pred + 1e-8))
#     print_qerror_stats(qerror)

#     print("\nDone.")


# if __name__ == "__main__":
#     main()




import os
import torch
import numpy as np

# ==================== 用户配置区域 ====================
base_csv_path = "/data2/xuyining/learnedcardinalities/data/tpch-skew/results/no-join.csv"
output_csv_path = "/data2/xuyining/learnedcardinalities/data/tpch-skew/results/full-merge.csv"

versions = [
    ("/data2/xuyining/learnedcardinalities/data/tpch-skew/results/full.csv",
     "/data2/xuyining/Sampler/join_sampling/new_model_results/tpch/embedding/pca_hit_status_test.pt")
]
# ====================================================


def load_csv_predictions(csv_path):
    preds = []
    trues = []
    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) != 2:
                raise ValueError(f"Invalid CSV line: {line}")
            pred, true = int(float(parts[0])), int(float(parts[1]))
            preds.append(pred)
            trues.append(true)
    return np.array(preds), np.array(trues)


def load_hit_status(status_path):
    if status_path.endswith('.pt'):
        data = torch.load(status_path, map_location='cpu')
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        return data.flatten()
    elif status_path.endswith('.npy'):
        return np.load(status_path).flatten()
    else:
        raise ValueError("Hit status file must have extension .pt or .npy")


def compute_qerror(pred, true):
    pred = float(pred)
    true = float(true)
    if true == 0 and pred == 0:
        return 1.0
    if true == 0:
        return pred / 1e-8
    if pred == 0:
        return true / 1e-8
    return max(pred / true, true / pred)


def print_qerror_stats(qerror):
    print("Median: {:.4f}".format(np.median(qerror)))
    print("80th percentile: {:.4f}".format(np.percentile(qerror, 80)))
    print("90th percentile: {:.4f}".format(np.percentile(qerror, 90)))
    print("95th percentile: {:.4f}".format(np.percentile(qerror, 95)))
    print("99th percentile: {:.4f}".format(np.percentile(qerror, 99)))
    print("Max: {:.4f}".format(np.max(qerror)))
    print("Mean: {:.4f}".format(np.mean(qerror)))


def print_queries_at_percentiles(qerror, base_preds, ver_preds, final_preds, trues, percentiles=[50, 80, 90, 95, 99, 100]):
    n = len(qerror)
    sorted_indices = np.argsort(qerror)
    for p in percentiles:
        q_val = np.percentile(qerror, p)
        idx_in_sorted = np.searchsorted(qerror[sorted_indices], q_val)
        if idx_in_sorted >= n:
            idx_in_sorted = n - 1
        original_idx = sorted_indices[idx_in_sorted]
        base_p = base_preds[original_idx]
        ver_p = ver_preds[original_idx] if ver_preds is not None else None
        final_p = final_preds[original_idx]
        true_v = trues[original_idx]
        q = qerror[original_idx]
        print(f"\n--- {p}th percentile (index {original_idx+1}, Q-error = {q:.4f}) ---")
        print(f"  Base prediction:  {base_p}")
        if ver_preds is not None:
            print(f"  Version prediction: {ver_p}")
        print(f"  Final prediction:  {final_p}")
        print(f"  True cardinality:  {true_v}")


def print_queries_worse_than_version(qerror_merge, qerror_ver, base_preds, ver_preds, final_preds, trues):
    """
    输出所有 merge 后 Q-error 大于 version（第一个版本）的查询
    """
    worse_mask = qerror_merge > qerror_ver
    worse_indices = np.where(worse_mask)[0]
    count = len(worse_indices)
    print(f"\n=== Queries where merge Q-error > version Q-error (total {count}) ===")
    if count == 0:
        print("  (none)")
        return
    # 按 merge 与 ver 的误差比值降序排列（显示最差的在前面）
    ratio = qerror_merge[worse_indices] / qerror_ver[worse_indices]
    sorted_order = np.argsort(ratio)[::-1]  # 降序
    for idx_in_worse in sorted_order:
        idx = worse_indices[idx_in_worse]
        print(f"\nQuery index {idx+1}:")
        print(f"  Base pred:    {base_preds[idx]}")
        print(f"  Version pred: {ver_preds[idx]}")
        print(f"  Merge pred:   {final_preds[idx]}")
        print(f"  True value:   {trues[idx]}")
        print(f"  Q-error (base):    {compute_qerror(base_preds[idx], trues[idx]):.4f}")
        print(f"  Q-error (version): {qerror_ver[idx]:.4f}")
        print(f"  Q-error (merge):   {qerror_merge[idx]:.4f}")
        print(f"  Merge/Version ratio: {ratio[idx_in_worse]:.4f}")


def main():
    print(f"Loading base predictions from: {base_csv_path}")
    base_preds, base_trues = load_csv_predictions(base_csv_path)
    n_queries = len(base_preds)
    print(f"Base: {n_queries} queries loaded.")

    final_preds = base_preds.copy()
    ver_preds = None  # 存储第一个版本的预测

    all_prev_hit = np.ones(n_queries, dtype=bool)
    for idx, (csv_path, status_path) in enumerate(versions):
        print(f"\nApplying version {idx+1}:")
        print(f"  Prediction CSV: {csv_path}")
        print(f"  Hit status: {status_path}")

        if not os.path.isfile(csv_path):
            print(f"  Warning: CSV file not found, skipping.")
            continue
        if not os.path.isfile(status_path):
            print(f"  Warning: Hit status file not found, skipping.")
            continue

        preds_ver, trues_ver = load_csv_predictions(csv_path)
        if len(preds_ver) != n_queries:
            raise ValueError(f"Version {idx+1} length mismatch")
        if not np.allclose(trues_ver, base_trues, rtol=1e-6, atol=1e-6):
            print(f"  Warning: True values differ from base (ignoring).")

        hit = load_hit_status(status_path)
        if len(hit) != n_queries:
            raise ValueError(f"Hit status length mismatch")
        hit_mask = np.asarray(hit).flatten() == 1

        replace_mask = hit_mask & all_prev_hit
        all_prev_hit = all_prev_hit & hit_mask

        replaced_count = int(np.count_nonzero(replace_mask))
        if replaced_count > 0:
            final_preds[replace_mask] = preds_ver[replace_mask]
            if ver_preds is None:
                ver_preds = preds_ver.copy()
            print(f"  Replaced {replaced_count} queries.")
        else:
            print("  No replacements.")

    if ver_preds is None:
        ver_preds = base_preds.copy()  # fallback

    # 保存合并结果（整数）
    print(f"\nSaving merged predictions to: {output_csv_path}")
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
    with open(output_csv_path, 'w') as f:
        for p, t in zip(final_preds, base_trues):
            f.write(f"{int(p)},{int(t)}\n")

    # 计算 Q‑error
    qerror_merge = np.array([compute_qerror(p, t) for p, t in zip(final_preds, base_trues)])
    qerror_ver = np.array([compute_qerror(p, t) for p, t in zip(ver_preds, base_trues)])
    qerror_base = np.array([compute_qerror(p, t) for p, t in zip(base_preds, base_trues)])

    print("\n=== Final Merge Q-error Statistics ===")
    print_qerror_stats(qerror_merge)

    # print("\n=== Version (bitmap-qa) Q-error Statistics ===")
    # print_qerror_stats(qerror_ver)

    # # 分位数详情（基于合并）
    # print("\n=== Queries at Percentiles (based on merge Q-error) ===")
    # print_queries_at_percentiles(qerror_merge, base_preds, ver_preds, final_preds, base_trues)

    # # 输出所有 merge 比 version 差的查询
    # print_queries_worse_than_version(qerror_merge, qerror_ver, base_preds, ver_preds, final_preds, base_trues)

    print("\nDone.")


if __name__ == "__main__":
    main()