import os
import argparse
import json
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader

# 根据你的目录结构导入
from mscn.data import load_and_encode_all_data
from mscn.model import SetConv

def unnormalize_torch(vals, min_val, max_val):
    vals = (vals * (max_val - min_val)) + min_val
    return torch.exp(vals)

def get_metrics(model, data_loader, cuda, min_val, max_val):
    """辅助函数：计算 Q-error，同时返回反归一化的预测值用于写出csv"""
    model.eval()
    all_preds = []
    all_labels =[]
    with torch.no_grad():
        for data_batch in data_loader:
            samples, predicates, joins, join_samples, targets, s_mask, p_mask, j_mask = data_batch
            if cuda:
                samples, predicates, joins, join_samples, targets = samples.cuda(), predicates.cuda(), joins.cuda(), join_samples.cuda(), targets.cuda()
                s_mask, p_mask, j_mask = s_mask.cuda(), p_mask.cuda(), j_mask.cuda()
            
            outputs = model(samples, predicates, joins, join_samples, s_mask, p_mask, j_mask)
            all_preds.append(outputs)
            all_labels.append(targets)
    
    preds_norm = torch.cat(all_preds, dim=0).cpu().numpy().flatten()
    labels_norm = torch.cat(all_labels, dim=0).cpu().numpy().flatten()
    
    # 反归一化并计算 Q-error
    preds_un = np.exp(preds_norm * (max_val - min_val) + min_val)
    preds_int = np.array(np.round(preds_un), dtype=np.int64)
    
    labels_un = np.exp(labels_norm * (max_val - min_val) + min_val)
    labels_int = np.array(np.round(labels_un), dtype=np.int64)
    
    # 强转为 float64 以进行除法计算（防止因为两个整数相除直接截断）
    p_f = preds_int.astype(np.float64)
    l_f = labels_int.astype(np.float64)
    
    # 计算 Q-error
    qerror = np.maximum(p_f / (l_f + 1e-5), l_f / (p_f + 1e-5))    
    # 修改：把反归一化后的预测基数也一起返回
    return qerror, preds_int

def print_qerror_from_array(qerror):
    print("Median: {:.4f}".format(np.median(qerror)))
    print("90th percentile: {:.4f}".format(np.percentile(qerror, 90)))
    print("95th percentile: {:.4f}".format(np.percentile(qerror, 95)))
    print("99th percentile: {:.4f}".format(np.percentile(qerror, 99)))
    print("Max: {:.4f}".format(np.max(qerror)))
    print("Mean: {:.4f}".format(np.mean(qerror)))

def evaluate_specific_epoch(config, model_path, target_epoch, output_csv):
    cuda = config.get("cuda", True)
    batch_size = config.get("batch", 1024)
    hid_units = config.get("hid", 256)
    num_buckets = config.get("num_buckets", 16)
    use_join_embedding = config.get("use_join_embedding", 0)
    use_single_embedding = config.get("use_single_embedding", 0) # <--- 新增：读取单表 embedding 开关

    print("Loading datasets to initialize model structure...")
    dicts, column_min_max_vals, min_val, max_val, train_data, val_data, test_data, max_lens, test_label_raw = load_and_encode_all_data(config)
    
    # 获取维度用于初始化模型
    table2vec, column2vec, op2vec, join2vec = dicts
    if use_join_embedding == 1:
        join_sample_feats = train_data[0][3].shape[0]
    else:
        join_sample_feats = 0

    # ---------------- 核心修改逻辑 ----------------
    table_vec_size = len(table2vec)
    first_sample_tensor = train_data[0][0]
    total_sample_feats = first_sample_tensor.shape[1]
    sample_vec_size = total_sample_feats - table_vec_size # 计算得到 Bitmap 或 Embedding 的真实维度

    predicate_feats = len(column2vec) + len(op2vec) + 1 + num_buckets
    join_feats = len(join2vec)

    print(f"Initializing Model: Table One-Hot={table_vec_size}, Sample(Emb/Bit)={sample_vec_size}, Predicate={predicate_feats}, Join={join_feats}")
    
    # 初始化模型时传入切分后的维度和开关
    model = SetConv(table_vec_size, sample_vec_size, predicate_feats, join_feats, join_sample_feats, hid_units, use_single_embedding)
    # ----------------------------------------------
    
    print(f"\nLoading saved state dictionary from: {model_path}")
    all_epochs_dict = torch.load(model_path, map_location="cpu")
    
    if target_epoch not in all_epochs_dict:
        raise ValueError(f"Epoch {target_epoch} not found in the saved file! Available epochs: {list(all_epochs_dict.keys())}")
        
    # 提取对应 epoch 的参数并加载
    model.load_state_dict(all_epochs_dict[target_epoch])
    
    if cuda:
        model.cuda()
    model.eval()

    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    print(f"\n=== Evaluation Results on Test Set for Epoch {target_epoch} ===")
    test_qerrors, preds_un = get_metrics(model, test_data_loader, cuda, min_val, max_val)
    print_qerror_from_array(test_qerrors)
    
    # ---------------- 输出到指定的 CSV 文件 ----------------
    if output_csv:
        # 如果父文件夹不存在，自动创建
        os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
        print(f"\nWriting test predictions to: {output_csv}")
        with open(output_csv, "w") as f:
            # preds_un 对应预测基数, test_label_raw 对应真实基数
            for i in range(len(preds_un)):
                f.write(str(preds_un[i]) + "," + str(test_label_raw[i]) + "\n")
        print("Done!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="JSON config file path", required=True)
    parser.add_argument("--model_path", help="Path to the .pt file containing all epochs", required=True)
    parser.add_argument("--epoch", help="The specific epoch number to load and evaluate", type=int, required=True)
    # 新增参数，用户自定义输出文件
    parser.add_argument("--output_csv", help="Path to save the predictions to a CSV file", required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    evaluate_specific_epoch(config, args.model_path, args.epoch, args.output_csv)

if __name__ == "__main__":
    main()