# import os
# import argparse
# import json
# import torch
# import numpy as np
# from torch.autograd import Variable
# from torch.utils.data import DataLoader

# # 根据你的目录结构导入
# from mscn.data import load_and_encode_all_data
# from mscn.model import SetConv

# def unnormalize_torch(vals, min_val, max_val):
#     vals = (vals * (max_val - min_val)) + min_val
#     return torch.exp(vals)

# def get_predictions(model, data_loader, cuda, min_val, max_val):
#     """辅助函数：执行推理，仅返回反归一化后的预测基数数组"""
#     model.eval()
#     all_preds = []
    
#     with torch.no_grad():
#         for data_batch in data_loader:
#             samples, predicates, joins, join_samples, targets, s_mask, p_mask, j_mask = data_batch
#             if cuda:
#                 samples, predicates, joins, join_samples = samples.cuda(), predicates.cuda(), joins.cuda(), join_samples.cuda()
#                 s_mask, p_mask, j_mask = s_mask.cuda(), p_mask.cuda(), j_mask.cuda()
            
#             outputs = model(samples, predicates, joins, join_samples, s_mask, p_mask, j_mask)
#             all_preds.append(outputs)
    
#     preds_norm = torch.cat(all_preds, dim=0).cpu().numpy().flatten()
    
#     # 反归一化
#     preds_un = np.exp(preds_norm * (max_val - min_val) + min_val)
#     preds_int = np.array(np.round(preds_un), dtype=np.int64)
    
#     return preds_int

# def print_qerror_from_array(qerror):
#     print("Median: {:.4f}".format(np.median(qerror)))
#     print("80th percentile: {:.4f}".format(np.percentile(qerror, 80)))
#     print("90th percentile: {:.4f}".format(np.percentile(qerror, 90)))
#     print("95th percentile: {:.4f}".format(np.percentile(qerror, 95)))
#     print("99th percentile: {:.4f}".format(np.percentile(qerror, 99)))
#     print("Max: {:.4f}".format(np.max(qerror)))
#     print("Mean: {:.4f}".format(np.mean(qerror)))

# def load_model_and_infer(train_config, model_path, target_epoch):
#     """根据单个训练配置加载数据和模型，并返回该模型的全量预测结果"""
#     cuda = train_config.get("cuda", True)
#     batch_size = train_config.get("batch", 1024)
#     hid_units = train_config.get("hid", 256)
#     num_buckets = train_config.get("num_buckets", 16)
#     use_join_embedding = train_config.get("use_join_embedding", 0)
#     use_single_embedding = train_config.get("use_single_embedding", 0)

#     # 此处加载数据是为了获取维度以及测试集。由于两个模型训练时的配置可能不同，必须分别解析
#     dicts, column_min_max_vals, min_val, max_val, train_data, val_data, test_data, max_lens, test_label_raw = load_and_encode_all_data(train_config)
    
#     table2vec, column2vec, op2vec, join2vec = dicts
#     join_sample_feats = train_data[0][3].shape[0] if use_join_embedding == 1 else 0

#     table_vec_size = len(table2vec)
#     first_sample_tensor = train_data[0][0]
#     total_sample_feats = first_sample_tensor.shape[1]
#     sample_vec_size = total_sample_feats - table_vec_size 

#     predicate_feats = len(column2vec) + len(op2vec) + 1 + num_buckets
#     join_feats = len(join2vec)

#     print(f"Initializing Model from {model_path}: Table One-Hot={table_vec_size}, Sample={sample_vec_size}, Predicate={predicate_feats}, Join={join_feats}")
    
#     model = SetConv(table_vec_size, sample_vec_size, predicate_feats, join_feats, join_sample_feats, hid_units, use_single_embedding)
    
#     all_epochs_dict = torch.load(model_path, map_location="cpu")
#     if target_epoch not in all_epochs_dict:
#         raise ValueError(f"Epoch {target_epoch} not found! Available epochs: {list(all_epochs_dict.keys())}")
        
#     model.load_state_dict(all_epochs_dict[target_epoch])
    
#     if cuda:
#         model.cuda()

#     # 显式设置 shuffle=False，保证预测顺序与 SQL 文件严格对应
#     test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
#     preds_int = get_predictions(model, test_data_loader, cuda, min_val, max_val)
    
#     return preds_int, test_label_raw

# def evaluate_hybrid(eval_config_path):
#     with open(eval_config_path, 'r') as f:
#         eval_cfg = json.load(f)

#     # 1. 必选：加载并运行 Base 模型 (无 Join Embedding 的模型)
#     base_cfg_path = eval_cfg["base_model"]["train_config_path"]
#     base_model_path = eval_cfg["base_model"]["model_path"]
#     base_epoch = eval_cfg["base_model"]["epoch"]
    
#     print("\n--- Phase 1: Running Base Model (Fallback) ---")
#     with open(base_cfg_path, 'r') as f:
#         train_config_base = json.load(f)
#     preds_base, labels_raw = load_model_and_infer(train_config_base, base_model_path, base_epoch)
    
#     # 最终结果暂时全部初始化为 Base 模型的预测
#     final_preds = preds_base.copy()


#     # ================= 2. 可选：运行 Single 模型 (回退第一层) =================
#     if "single_model" in eval_cfg and "single_hit_status_path" in eval_cfg:
#         single_cfg_path = eval_cfg["single_model"]["train_config_path"]
#         single_model_path = eval_cfg["single_model"]["model_path"]
#         single_epoch = eval_cfg["single_model"]["epoch"]
#         single_hit_status_path = eval_cfg["single_hit_status_path"]
        
#         print("\n--- Phase 2: Running Single Model & Merging ---")
#         with open(single_cfg_path, 'r') as f:
#             train_config_single = json.load(f)
#         preds_single, _ = load_model_and_infer(train_config_single, single_model_path, single_epoch)
        
#         single_hit_status = torch.load(single_hit_status_path).cpu().numpy().flatten()
#         if len(single_hit_status) != len(final_preds):
#             raise ValueError(f"Single hit status length ({len(single_hit_status)}) does not match test set length ({len(final_preds)})!")
        
#         # 覆盖：单表命中的查询使用 single_model 结果
#         single_hit_count = 0
#         for i in range(len(final_preds)):
#             if single_hit_status[i] == 1:
#                 final_preds[i] = preds_single[i]
#                 single_hit_count += 1
#         print(f"Applied Single Model for {single_hit_count}/{len(final_preds)} queries ({(single_hit_count/len(final_preds)):.2%}).")

#     # ================= 3. 可选：运行 Join 模型 (回退最顶层) =================
#     if "join_model" in eval_cfg and "join_hit_status_path" in eval_cfg:
#         join_cfg_path = eval_cfg["join_model"]["train_config_path"]
#         join_model_path = eval_cfg["join_model"]["model_path"]
#         join_epoch = eval_cfg["join_model"]["epoch"]
#         join_hit_status_path = eval_cfg["join_hit_status_path"]
        
#         print("\n--- Phase 3: Running Join Model & Merging ---")
#         with open(join_cfg_path, 'r') as f:
#             train_config_join = json.load(f)
#         preds_join, _ = load_model_and_infer(train_config_join, join_model_path, join_epoch)
        
#         join_hit_status = torch.load(join_hit_status_path).cpu().numpy().flatten()
#         if len(join_hit_status) != len(final_preds):
#             raise ValueError(f"Join hit status length ({len(join_hit_status)}) does not match test set length ({len(final_preds)})!")
        
#         # 再次覆盖：Join 命中的查询使用 join_model 结果
#         join_hit_count = 0
#         for i in range(len(final_preds)):
#             if join_hit_status[i] == 1 and single_hit_status[i] == 1:  # 只有当单表也命中时才覆盖为 Join 模型结果
#                 final_preds[i] = preds_join[i]
#                 join_hit_count += 1
#         print(f"Applied Join Model for {join_hit_count}/{len(final_preds)} queries ({(join_hit_count/len(final_preds)):.2%}).")
#     else:
#         print("\n--- No Join Model / Hit Status configured. Evaluating Base Model only. ---")

#     # 3. 计算最终的 Q-error
#     print("\n=== Final Evaluation Results ===")
#     p_f = final_preds.astype(np.float64)
#     l_f = np.array(labels_raw).astype(np.float64)
    
#     qerror = np.maximum(p_f / (l_f + 1e-5), l_f / (p_f + 1e-5))
#     print_qerror_from_array(qerror)

#     # 4. 写出 CSV
#     output_csv = eval_cfg.get("output_csv", "results.csv")
#     os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
#     print(f"\nWriting test predictions to: {output_csv}")
#     with open(output_csv, "w") as f:
#         for i in range(len(final_preds)):
#             f.write(str(final_preds[i]) + "," + str(labels_raw[i]) + "\n")
#     print("Done!")

# def main():
#     parser = argparse.ArgumentParser()
#     # 现在只需要传入一个综合的 eval_config.json 路径
#     parser.add_argument("--eval_config", help="Path to evaluation config JSON file", required=True)
#     args = parser.parse_args()

#     evaluate_hybrid(args.eval_config)

# if __name__ == "__main__":
#     main()



import os
import argparse
import json
import torch
import numpy as np
import time  # 新增：用于高精度计时
from torch.autograd import Variable
from torch.utils.data import DataLoader

# 根据你的目录结构导入
from mscn.data import load_and_encode_all_data
from mscn.model import SetConv
from mscn.mixture import mixture_options, join_mixture_options

def unnormalize_torch(vals, min_val, max_val):
    vals = (vals * (max_val - min_val)) + min_val
    return torch.exp(vals)

# def get_predictions(model, data_loader, cuda, min_val, max_val):
#     """辅助函数：执行推理，返回预测结果、总推理时间和总查询数"""
#     model.eval()
#     all_preds = []
    
#     total_infer_time = 0.0  # 初始化总推理时间
#     total_queries = 0       # 初始化查询总数
    
#     with torch.no_grad():

#         # ================== 新增：GPU Warm-up (预热) ==================
#         # 随便取第一个 batch 让模型跑 5 次，不计入时间
#         warmup_batch = next(iter(data_loader))
#         w_samples, w_predicates, w_joins, w_join_samples, w_targets, w_s_mask, w_p_mask, w_j_mask = warmup_batch
#         if cuda:
#             w_samples, w_predicates, w_joins, w_join_samples = w_samples.cuda(), w_predicates.cuda(), w_joins.cuda(), w_join_samples.cuda()
#             w_s_mask, w_p_mask, w_j_mask = w_s_mask.cuda(), w_p_mask.cuda(), w_j_mask.cuda()
#         for _ in range(5):
#             _ = model(w_samples, w_predicates, w_joins, w_join_samples, w_s_mask, w_p_mask, w_j_mask)
#         if cuda:
#             torch.cuda.synchronize()
#         # ==============================================================

        
#         for data_batch in data_loader:
#             samples, predicates, joins, join_samples, targets, s_mask, p_mask, j_mask = data_batch
            
#             # 记录当前 batch 的大小
#             batch_size_current = samples.size(0)
#             total_queries += batch_size_current
            
#             if cuda:
#                 samples, predicates, joins, join_samples = samples.cuda(), predicates.cuda(), joins.cuda(), join_samples.cuda()
#                 s_mask, p_mask, j_mask = s_mask.cuda(), p_mask.cuda(), j_mask.cuda()
            
#             # --- 开始计时 ---
#             if cuda:
#                 torch.cuda.synchronize()  # 确保GPU之前的任务都已完成
#             start_time = time.perf_counter()
            
#             outputs = model(samples, predicates, joins, join_samples, s_mask, p_mask, j_mask)
            
#             # --- 结束计时 ---
#             if cuda:
#                 torch.cuda.synchronize()  # 强制CPU等待GPU计算完成
#             end_time = time.perf_counter()
            
#             total_infer_time += (end_time - start_time)
            
#             all_preds.append(outputs)
    
#     preds_norm = torch.cat(all_preds, dim=0).cpu().numpy().flatten()
    
#     # 反归一化
#     preds_un = np.exp(preds_norm * (max_val - min_val) + min_val)
#     preds_int = np.array(np.round(preds_un), dtype=np.int64)
    
#     return preds_int, total_infer_time, total_queries

def get_predictions(model, data_loader, cuda, min_val, max_val):
    """辅助函数：执行推理，返回预测结果、总推理时间和总查询数"""
    model.eval()
    all_preds = []
    
    total_infer_time = 0.0  # 初始化总推理时间
    total_queries = 0       # 初始化查询总数
    
    with torch.no_grad():

        # ================== GPU Warm-up (预热) ==================
        # 随便取第一个 batch 让模型跑 5 次，不计入时间。
        # 这一步对于消除 CUDA 初始化耗时极其重要
        data_iter = iter(data_loader)
        w_batch = next(data_iter)
        w_samples, w_predicates, w_joins, w_join_samples, w_targets, w_s_mask, w_p_mask, w_j_mask = w_batch
        if cuda:
            w_samples, w_predicates, w_joins, w_join_samples = w_samples.cuda(), w_predicates.cuda(), w_joins.cuda(), w_join_samples.cuda()
            w_s_mask, w_p_mask, w_j_mask = w_s_mask.cuda(), w_p_mask.cuda(), w_j_mask.cuda()
        for _ in range(5):
            _ = model(w_samples, w_predicates, w_joins, w_join_samples, w_s_mask, w_p_mask, w_j_mask)
        if cuda:
            torch.cuda.synchronize()
        # ==============================================================

        # 重新初始化迭代器，开始真实的端到端计时
        data_iter = iter(data_loader)
        
        for _ in range(len(data_loader)):
            # --- 1. 同步并开始计时 ---
            # 计时器启动于“优化器刚把 SQL 甩给模型”的瞬间
            if cuda:
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            # --- 2. 数据预处理时间 (Data Preprocessing) ---
            # next() 会触发 dataset.__getitem__，包含生成 Tensor、Mask 填充等 CPU 逻辑
            data_batch = next(data_iter)
            samples, predicates, joins, join_samples, targets, s_mask, p_mask, j_mask = data_batch
            
            # --- 3. CPU 到 GPU 传输时间 (Memory Transfer) ---
            if cuda:
                samples = samples.cuda(non_blocking=True)
                predicates = predicates.cuda(non_blocking=True)
                joins = joins.cuda(non_blocking=True)
                join_samples = join_samples.cuda(non_blocking=True)
                s_mask = s_mask.cuda(non_blocking=True)
                p_mask = p_mask.cuda(non_blocking=True)
                j_mask = j_mask.cuda(non_blocking=True)
            
            # --- 4. 模型前向传播时间 (Model Forward) ---
            outputs = model(samples, predicates, joins, join_samples, s_mask, p_mask, j_mask)
            
            # --- 5. 同步并结束计时 ---
            if cuda:
                torch.cuda.synchronize()  # 必须等 GPU 算完
            end_time = time.perf_counter()
            
            # 累加端到端总时间
            total_infer_time += (end_time - start_time)
            
            # 记录查询数和预测结果
            total_queries += samples.size(0) # 因为 batch_size=1，这里其实就是 +1
            all_preds.append(outputs)
    
    preds_norm = torch.cat(all_preds, dim=0).cpu().numpy().flatten()
    
    # 反归一化
    preds_un = np.exp(preds_norm * (max_val - min_val) + min_val)
    preds_int = np.array(np.round(preds_un), dtype=np.int64)
    
    return preds_int, total_infer_time, total_queries

def print_qerror_from_array(qerror):
    print("Median: {:.4f}".format(np.median(qerror)))
    print("80th percentile: {:.4f}".format(np.percentile(qerror, 80)))
    print("90th percentile: {:.4f}".format(np.percentile(qerror, 90)))
    print("95th percentile: {:.4f}".format(np.percentile(qerror, 95)))
    print("99th percentile: {:.4f}".format(np.percentile(qerror, 99)))
    print("Max: {:.4f}".format(np.max(qerror)))
    print("Mean: {:.4f}".format(np.mean(qerror)))

def load_model_and_infer(train_config, model_path):
    """根据单个训练配置加载数据和模型，并返回预测结果和推理耗时信息"""
    cuda = train_config.get("cuda", True)
    # batch_size = train_config.get("batch", 1024)
    batch_size = 1  # 强制 batch=1 以精确测量单条查询的线上延迟
    hid_units = train_config.get("hid", 256)
    num_buckets = train_config.get("num_buckets", 16)
    
    # 获取各种特征控制开关
    use_join_embedding = train_config.get("use_join_embedding", 0)
    use_single_embedding = train_config.get("use_single_embedding", 0)
    has_unmatched_embedding = train_config.get("has_unmatched_embedding", 0) # 💡 新增：读取不命中特征开关

    dicts, column_min_max_vals, min_val, max_val, train_data, val_data, test_data, max_lens, test_label_raw = load_and_encode_all_data(train_config)
    
    table2vec, column2vec, op2vec, join2vec = dicts
    
    # Match the training loader: bitmap, bitmap+embedding+PCA, or bitmap+embedding.
    if use_join_embedding in [1, 2, 3]:
        join_sample_feats = train_data[0][3].shape[0]
    else:
        join_sample_feats = 0

    table_vec_size = len(table2vec)
    first_sample_tensor = train_data[0][0]
    total_sample_feats = first_sample_tensor.shape[1]
    sample_vec_size = total_sample_feats - table_vec_size 

    predicate_feats = len(column2vec) + len(op2vec) + 1 + num_buckets
    join_feats = len(join2vec)

    print(f"Initializing Model from {model_path}: Table One-Hot={table_vec_size}, Sample={sample_vec_size}, Predicate={predicate_feats}, Join={join_feats}")
    print(f"Feature Flags -> Single_Emb: {use_single_embedding} | Join_Emb: {use_join_embedding} | Unmatched: {has_unmatched_embedding == 1}")
    
    # 💡 修改：补齐实例化参数，传入 has_unmatched_embedding
    model = SetConv(table_vec_size, sample_vec_size, predicate_feats, join_feats, join_sample_feats, hid_units, use_single_embedding, has_unmatched_embedding=has_unmatched_embedding,
                    single_mixture_options=mixture_options(train_config),
                    join_mixture_options=join_mixture_options(train_config))
    
    model_state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(model_state)
    
    if cuda:
        model.cuda()

    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    # 接收新增的返回参数
    preds_int, total_infer_time, num_queries = get_predictions(model, test_data_loader, cuda, min_val, max_val)
    
    return preds_int, test_label_raw, total_infer_time, num_queries

def evaluate_hybrid(eval_config_path):
    with open(eval_config_path, 'r') as f:
        eval_cfg = json.load(f)

    # 变量用于记录最高级模型的推理时间
    highest_model_name = ""
    highest_model_total_time = 0.0
    total_queries = 0

    # 1. Base 模型 (Fallback)
    base_cfg_path = eval_cfg["base_model"]["train_config_path"]
    base_model_path = eval_cfg["base_model"]["model_path"]
    
    print("\n--- Phase 1: Running Base Model (Fallback) ---")
    with open(base_cfg_path, 'r') as f:
        train_config_base = json.load(f)
    preds_base, labels_raw, base_time, q_count = load_model_and_infer(train_config_base, base_model_path)
    
    final_preds = preds_base.copy()
    highest_model_name = "Base Model"
    highest_model_total_time = base_time
    total_queries = q_count

    # 2. Single 模型
    if "single_model" in eval_cfg and "single_hit_status_path" in eval_cfg:
        single_cfg_path = eval_cfg["single_model"]["train_config_path"]
        single_model_path = eval_cfg["single_model"]["model_path"]
        single_hit_status_path = eval_cfg["single_hit_status_path"]
        
        print("\n--- Phase 2: Running Single Model & Merging ---")
        with open(single_cfg_path, 'r') as f:
            train_config_single = json.load(f)
        preds_single, _, single_time, _ = load_model_and_infer(train_config_single, single_model_path)
        
        highest_model_name = "Single Model"
        highest_model_total_time = single_time

        single_hit_status = torch.load(single_hit_status_path).cpu().numpy().flatten()
        if len(single_hit_status) != len(final_preds):
            raise ValueError(f"Single hit status length ({len(single_hit_status)}) does not match test set length ({len(final_preds)})!")
        
        single_hit_count = 0
        for i in range(len(final_preds)):
            if single_hit_status[i] == 1:
                final_preds[i] = preds_single[i]
                single_hit_count += 1
        print(f"Applied Single Model for {single_hit_count}/{len(final_preds)} queries.")

    # 3. Join 模型 (通常是最高级配置)
    if "join_model" in eval_cfg and "join_hit_status_path" in eval_cfg:
        join_cfg_path = eval_cfg["join_model"]["train_config_path"]
        join_model_path = eval_cfg["join_model"]["model_path"]
        join_hit_status_path = eval_cfg["join_hit_status_path"]
        
        print("\n--- Phase 3: Running Join Model & Merging ---")
        with open(join_cfg_path, 'r') as f:
            train_config_join = json.load(f)
        preds_join, _, join_time, _ = load_model_and_infer(train_config_join, join_model_path)
        
        highest_model_name = "Join Model"
        highest_model_total_time = join_time

        join_hit_status = torch.load(join_hit_status_path).cpu().numpy().flatten()
        if len(join_hit_status) != len(final_preds):
            raise ValueError(f"Join hit status length ({len(join_hit_status)}) does not match test set length ({len(final_preds)})!")
        
        join_hit_count = 0
        for i in range(len(final_preds)):
            if join_hit_status[i] == 1 and single_hit_status[i] == 1: 
                final_preds[i] = preds_join[i]
                join_hit_count += 1
        print(f"Applied Join Model for {join_hit_count}/{len(final_preds)} queries.")
    else:
        print("\n--- No Join Model / Hit Status configured. ---")

    # --- 输出最高级配置的推理时间 ---
    print(f"\n=== Inference Time Statistics (Highest Configured: {highest_model_name}) ===")
    print(f"Total Inference Time for {total_queries} queries: {highest_model_total_time:.4f} seconds")
    avg_time_ms = (highest_model_total_time / total_queries) * 1000
    print(f"Average Inference Time per query: {avg_time_ms:.4f} ms")

    # 4. 计算最终的 Q-error
    print("\n=== Final Evaluation Results ===")
    p_f = final_preds.astype(np.float64)
    l_f = np.array(labels_raw).astype(np.float64)
    
    qerror = np.maximum(p_f / (l_f + 1e-5), l_f / (p_f + 1e-5))
    print_qerror_from_array(qerror)

    # 5. 写出 CSV
    output_csv = eval_cfg.get("output_csv", "results.csv")
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
    print(f"\nWriting test predictions to: {output_csv}")
    with open(output_csv, "w") as f:
        for i in range(len(final_preds)):
            f.write(str(final_preds[i]) + "," + str(labels_raw[i]) + "\n")
    print("Done!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_config", help="Path to evaluation config JSON file", required=True)
    args = parser.parse_args()

    evaluate_hybrid(args.eval_config)

if __name__ == "__main__":
    main()
