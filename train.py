import argparse
import time
import os
import json
import torch
import copy
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
import datetime

from mscn.util import *
from mscn.data import get_train_datasets, load_data, make_dataset, load_and_encode_all_data
from mscn.model import SetConv


def unnormalize_torch(vals, min_val, max_val):
    vals = (vals * (max_val - min_val)) + min_val
    return torch.exp(vals)


def qerror_loss(preds, targets, min_val, max_val):
    qerror = []
    preds = preds.view(-1)
    targets = targets.view(-1)
    preds = unnormalize_torch(preds, min_val, max_val)
    targets = unnormalize_torch(targets, min_val, max_val)

    return torch.mean((preds > targets) * (preds / (targets + 1e-5)) + (targets >= preds) * (targets / (preds + 1e-5)))


def predict(model, data_loader, cuda):
    preds = []
    labels = []
    t_total = 0.

    model.eval()
    for batch_idx, data_batch in enumerate(data_loader):

        samples, predicates, joins, join_samples, targets, sample_masks, predicate_masks, join_masks = data_batch

        if cuda:
            samples, predicates, joins, join_samples, targets = samples.cuda(), predicates.cuda(), joins.cuda(), join_samples.cuda(), targets.cuda()
            sample_masks, predicate_masks, join_masks = sample_masks.cuda(), predicate_masks.cuda(), join_masks.cuda()
        samples, predicates, joins, targets = Variable(samples), Variable(predicates), Variable(joins), Variable(
            targets)
        sample_masks, predicate_masks, join_masks = Variable(sample_masks), Variable(predicate_masks), Variable(
            join_masks)

        t = time.time()
        with torch.no_grad():
            outputs = model(samples, predicates, joins, join_samples, sample_masks, predicate_masks, join_masks)
        t_total += time.time() - t

        # for i in range(outputs.data.shape[0]):
        #     preds.append(outputs.data[i])
        preds.append(outputs)
        labels.append(targets)
    
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
        
    return preds, t_total, labels


def print_qerror(preds_unnorm, labels_unnorm):
    # qerror = []
    # for i in range(len(preds_unnorm)):
    #     if float(preds_unnorm[i]) > float(labels_unnorm[i]):
    #         qerror.append(float(preds_unnorm[i]) / float(labels_unnorm[i]))
    #     else:
    #         qerror.append(float(labels_unnorm[i]) / float(preds_unnorm[i]))

    preds = np.array(preds_unnorm).flatten()
    labels = np.array(labels_unnorm, dtype=np.float64).flatten()
    
    # 向量化计算 Q-error
    qerror = np.maximum(preds / (labels + 1e-5), labels / (preds + 1e-5))

    print("Median: {}".format(np.median(qerror)))
    print("90th percentile: {}".format(np.percentile(qerror, 90)))
    print("95th percentile: {}".format(np.percentile(qerror, 95)))
    print("99th percentile: {}".format(np.percentile(qerror, 99)))
    print("Max: {}".format(np.max(qerror)))
    print("Mean: {}".format(np.mean(qerror)))

def print_qerror_from_array(qerror):
    print("Median: {:.4f}".format(np.median(qerror)))
    print("90th percentile: {:.4f}".format(np.percentile(qerror, 90)))
    print("95th percentile: {:.4f}".format(np.percentile(qerror, 95)))
    print("99th percentile: {:.4f}".format(np.percentile(qerror, 99)))
    print("Max: {:.4f}".format(np.max(qerror)))
    print("Mean: {:.4f}".format(np.mean(qerror)))


def get_metrics(model, data_loader, cuda, min_val, max_val):
    """辅助函数：快速计算当前 dataloader 的所有 Q-error (严格对齐 unnormalize_labels 的舍入逻辑)"""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0  # <--- 新增：用于累加 Loss

    with torch.no_grad():
        for data_batch in data_loader:
            samples, predicates, joins, join_samples, targets, s_mask, p_mask, j_mask = data_batch
            if cuda:
                samples, predicates, joins, join_samples, targets = samples.cuda(), predicates.cuda(), joins.cuda(), join_samples.cuda(), targets.cuda()
                s_mask, p_mask, j_mask = s_mask.cuda(), p_mask.cuda(), j_mask.cuda()
            
            outputs = model(samples, predicates, joins, join_samples, s_mask, p_mask, j_mask)

            loss = qerror_loss(outputs, targets.float(), min_val, max_val)
            total_loss += loss.item()

            all_preds.append(outputs)
            all_labels.append(targets)

    avg_loss = total_loss / len(data_loader)
    
    preds_norm = torch.cat(all_preds, dim=0).cpu().numpy().flatten()
    labels_norm = torch.cat(all_labels, dim=0).cpu().numpy().flatten()
    
    preds_un = np.exp(preds_norm * (max_val - min_val) + min_val)
    preds_int = np.array(np.round(preds_un), dtype=np.int64)

    labels_un = np.exp(labels_norm * (max_val - min_val) + min_val)
    labels_int = np.array(np.round(labels_un), dtype=np.int64)

    p_f = preds_int.astype(np.float64)
    l_f = labels_int.astype(np.float64)

    qerror = np.maximum(p_f / (l_f + 1e-5), l_f / (p_f + 1e-5))

    return qerror, avg_loss


def train_and_predict(config):
    # 读取参数
    testset = config["testset"]
    trainset = config["trainset"]
    num_epochs = config["epochs"]
    batch_size = config["batch"]
    hid_units = config["hid"]
    cuda = config["cuda"]
    lr = config.get("lr", 0.001) # 新增：学习率
    # --- 新增：权重衰减参数，默认给 1e-4，防止 Embedding 过拟合 ---
    weight_decay = config.get("weight_decay", 1e-4) 
    num_materialized_samples = config.get("num_samples", 1000) # 样本大小
    num_buckets = config.get("num_buckets", 16)

    use_single_embedding = config.get("use_single_embedding", 0)
    test_emb_path = config.get("test_embedding_file", "")

    results_dir = config.get("results_dir", "results") # 保存目录
    workloads_dir = config.get("workloads_dir", "workloads") # 数据集目录
    workload_name = config.get("workload_name", "imdb") # 数据集名称

    use_join_embedding = config.get("use_join_embedding", 0)

    model_id = f"{workload_name}_{trainset}_{lr}_{num_epochs}_{batch_size}"
    checkpoint_dir = "/data2/xuyining/learnedcardinalities/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    # 添加时间戳
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")

    # best_model_path = os.path.join(checkpoint_dir, f"exp_{timestamp}_best.pt")
    # print(f"Training model will be saved to: {best_model_path}")

    models_save_path = os.path.join(checkpoint_dir, f"exp_{timestamp}_all_epochs.pt")
    print(f"All epochs of the model will be saved to: {models_save_path}")

    print(f"learning rate: {lr}")
    print(f"batch size: {batch_size}")
    print(f"hidden units: {hid_units}")
    print(f"epochs: {num_epochs}")

    print("Loading and encoding all datasets (global scan)...")
    time_start = time.time()
    dicts, column_min_max_vals, min_val, max_val, train_data, val_data, test_data, max_lens, test_label_raw = load_and_encode_all_data(config)
    time_end = time.time()
    print(f"Data loading and encoding time: {time_end - time_start:.2f} seconds")

    # Load training and validation data
    # dicts, column_min_max_vals, min_val, max_val, labels_train, labels_val, max_num_joins, max_num_predicates, train_data, val_data = get_train_datasets(
    #     config)

    table2vec, column2vec, op2vec, join2vec = dicts
    max_num_joins, max_num_predicates = max_lens

    # 动态确定维度
    # train_dataset[0][3] 是 join_samples
    if use_join_embedding == 1:
        join_sample_feats = train_data[0][3].shape[0]
    else:
        join_sample_feats = 0 # 传入 0，模型会自动回退到 3 个头的结构

    # Train model
    # sample_feats = len(table2vec) + num_materialized_samples
    first_sample_tensor = train_data[0][0]
    sample_feats = first_sample_tensor.shape[1]
    predicate_feats = len(column2vec) + len(op2vec) + 1 + num_buckets
    join_feats = len(join2vec)

    print(f"Sample features: {sample_feats}, Predicate features: {predicate_feats}")

    model = SetConv(sample_feats, predicate_feats, join_feats, join_sample_feats, hid_units)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # --- 新增：学习率调度器 ---
    # mode='min'：监控指标越小越好（针对 Loss）
    # factor=0.5：触发时学习率减半
    # patience=15：如果连续 15 个 epoch 验证集 Loss 都不降，就触发衰减
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

    if cuda:
        model.cuda()

    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    all_epoch_states = {}
    best_tracker = {
        "val_loss_val": float('inf'), "val_loss_ep": -1, # <--- 新增：记录最佳验证集 Loss
        "val_mean_val": float('inf'), "val_mean_ep": -1,
        "val_median_val": float('inf'), "val_median_ep": -1,
        "test_loss_val": float('inf'), "test_loss_ep": -1, # <--- 新增：记录最佳测试集 Loss
        "test_mean_val": float('inf'), "test_mean_ep": -1,
        "test_median_val": float('inf'), "test_median_ep": -1,
    }

    for epoch in range(num_epochs):
        loss_total = 0.

        # 训练阶段
        model.train()
        for batch_idx, data_batch in enumerate(train_data_loader):
            samples, predicates, joins, join_samples, targets, sample_masks, predicate_masks, join_masks = data_batch

            if cuda:
                samples, predicates, joins, join_samples, targets = samples.cuda(), predicates.cuda(), joins.cuda(), join_samples.cuda(), targets.cuda()
                sample_masks, predicate_masks, join_masks = sample_masks.cuda(), predicate_masks.cuda(), join_masks.cuda()
            samples, predicates, joins, join_samples, targets = Variable(samples), Variable(predicates), Variable(joins), Variable(join_samples), Variable(targets)
            sample_masks, predicate_masks, join_masks = Variable(sample_masks), Variable(predicate_masks), Variable(join_masks)

            optimizer.zero_grad()
            outputs = model(samples, predicates, joins, join_samples, sample_masks, predicate_masks, join_masks)
            loss = qerror_loss(outputs, targets.float(), min_val, max_val)
            loss_total += loss.item()
            loss.backward()

            # --- 新增：梯度裁剪，防止梯度爆炸 ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            optimizer.step()

        print("\nEpoch {}, Train Loss: {:.4f}".format(epoch, loss_total / len(train_data_loader)))

        # # 验证阶段
        # model.eval()
        # val_loss_total = 0.
        # with torch.no_grad():
        #     for batch_idx, data_batch in enumerate(val_data_loader):
        #         samples, predicates, joins, join_samples, targets, sample_masks, predicate_masks, join_masks = data_batch

        #         if cuda:
        #             samples, predicates, joins, join_samples, targets = samples.cuda(), predicates.cuda(), joins.cuda(), join_samples.cuda(), targets.cuda()
        #             sample_masks, predicate_masks, join_masks = sample_masks.cuda(), predicate_masks.cuda(), join_masks.cuda()
        #         samples, predicates, joins, join_samples, targets = Variable(samples), Variable(predicates), Variable(joins), Variable(join_samples), Variable(targets)
        #         sample_masks, predicate_masks, join_masks = Variable(sample_masks), Variable(predicate_masks), Variable(join_masks)

        #         outputs = model(samples, predicates, joins, join_samples, sample_masks, predicate_masks, join_masks)
        #         val_loss = qerror_loss(outputs, targets.float(), min_val, max_val)
        #         val_loss_total += val_loss.item()

        # val_loss_avg = val_loss_total / len(val_data_loader)
        # print("Epoch {}, Validation Loss: {}".format(epoch, val_loss_avg))

        # 验证集评估
        val_qerrors, val_loss_avg = get_metrics(model, val_data_loader, cuda, min_val, max_val)
        print("Epoch {}, Validation Loss: {:.4f}".format(epoch, val_loss_avg))
        # print(f"--- Epoch {epoch} Validation Set qerror ---")
        # print_qerror_from_array(val_qerrors)

        # --- 新增：让 Scheduler 根据当前的 验证集 Loss 决定是否要减小学习率 ---
        scheduler.step(val_loss_avg)
         # 当前学习率打印 (可选，方便你在日志里看它什么时候降了)
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != lr:
            print(f" [!] Learning Rate adjusted to: {current_lr}")
            lr = current_lr # 仅仅是为了防止重复打印

        # 同时每个epoch也在测试集上评估性能
        test_qerrors, test_loss_avg = get_metrics(model, test_data_loader, cuda, min_val, max_val)
        print("Epoch {}, Test Loss: {:.4f}".format(epoch, test_loss_avg))
        # print(f"--- Epoch {epoch} Test Set qerror ---")
        # print_qerror_from_array(test_qerrors)

        update_cur_epoch = False

        # 记录验证集 Loss 最小的 Epoch
        if val_loss_avg < best_tracker["val_loss_val"]:
            best_tracker["val_loss_val"] = val_loss_avg
            best_tracker["val_loss_ep"] = epoch
            print(f" --> New best model found at epoch {epoch} with validation loss: {val_loss_avg:.4f}")
            update_cur_epoch = True
        
        # 记录测试集 Loss 最小的 Epoch
        if test_loss_avg < best_tracker["test_loss_val"]:
            best_tracker["test_loss_val"] = test_loss_avg
            best_tracker["test_loss_ep"] = epoch
            print(f" --> New best model found at epoch {epoch} with test loss: {test_loss_avg:.4f}")
            update_cur_epoch = True

        v_mean, v_median = np.mean(val_qerrors), np.median(val_qerrors)
        t_mean, t_median = np.mean(test_qerrors), np.median(test_qerrors)

        if v_mean < best_tracker["val_mean_val"]:
            best_tracker["val_mean_val"] = v_mean
            best_tracker["val_mean_ep"] = epoch
            update_cur_epoch = True
        if v_median < best_tracker["val_median_val"]:
            best_tracker["val_median_val"] = v_median
            best_tracker["val_median_ep"] = epoch
            update_cur_epoch = True

        if t_mean < best_tracker["test_mean_val"]:
            best_tracker["test_mean_val"] = t_mean
            best_tracker["test_mean_ep"] = epoch
            update_cur_epoch = True
        if t_median < best_tracker["test_median_val"]:
            best_tracker["test_median_val"] = t_median
            best_tracker["test_median_ep"] = epoch
            update_cur_epoch = True

        if update_cur_epoch:
            # 只有在有更新时保存当前 epoch 的模型状态
            epoch_state = copy.deepcopy(model.state_dict())
            for k, v in epoch_state.items():
                epoch_state[k] = v.cpu()
            all_epoch_states[epoch] = epoch_state

    torch.save(all_epoch_states, models_save_path)
    print(f"\nAll epochs saved successfully to {models_save_path}")

    print("\n" + "="*60)
    print("FINAL SUMMARY & EVALUATIONS ON TEST SET")
    print("="*60)

    evaluation_targets = [
        ("Validation Loss", best_tracker["val_loss_ep"]),
        ("Validation Mean", best_tracker["val_mean_ep"]),
        ("Validation Median", best_tracker["val_median_ep"]),
        ("Test Loss", best_tracker["test_loss_ep"]),
        ("Test Mean", best_tracker["test_mean_ep"]),
        ("Test Median", best_tracker["test_median_ep"])
    ]

    for model_desc, ep in evaluation_targets:
        if ep == -1:
            print(f"\n>>> Skipping {model_desc}: No improvement recorded.")
            continue
        print(f"\n>>> Evaluating Model with Best {model_desc} (Epoch: {ep}) on TEST SET")
        # 从字典中加载对应的 epoch 状态
        model.load_state_dict(all_epoch_states[ep])
        if cuda: 
            model.cuda()
        
        t_qerrors, _ = get_metrics(model, test_data_loader, cuda, min_val, max_val)
        print_qerror_from_array(t_qerrors)
    print("="*60 + "\n")


    # # 加载最佳模型
    # print("Loading best model with validation loss: {:.4f}".format(best_val_loss))
    # # print("Loading best model with test median qerror: {:.4f}".format(best_test_median))
    # # print("Loading best model with validation median qerror: {:.4f}".format(best_val_median))
    # model.load_state_dict(torch.load(best_model_path))
    # model.eval()

    # ordered_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    # ordered_val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)    

    # # Get final training and validation set predictions
    # preds_train, t_total, labels_train = predict(model, ordered_train_loader, cuda)
    # print("Prediction time per training sample: {}".format(t_total / len(labels_train) * 1000))

    # preds_val, t_total, labels_val = predict(model, ordered_val_loader, cuda)
    # print("Prediction time per validation sample: {}".format(t_total / len(labels_val) * 1000))

    # # Unnormalize
    # preds_train_unnorm = unnormalize_labels(preds_train, min_val, max_val)
    # labels_train_unnorm = unnormalize_labels(labels_train, min_val, max_val)

    # preds_val_unnorm = unnormalize_labels(preds_val, min_val, max_val)
    # labels_val_unnorm = unnormalize_labels(labels_val, min_val, max_val)

    # # Print metrics
    # print("\nQ-Error training set:")
    # print_qerror(preds_train_unnorm, labels_train_unnorm)

    # print("\nQ-Error validation set:")
    # print_qerror(preds_val_unnorm, labels_val_unnorm)
    # print("")

    # test_data_loader = DataLoader(test_data, batch_size=batch_size)

    # preds_test, t_total, _ = predict(model, test_data_loader, cuda)
    # print("Prediction time per test sample: {}".format(t_total / len(test_label_raw) * 1000))

    # # Unnormalize
    # preds_test_unnorm = unnormalize_labels(preds_test, min_val, max_val)

    # # Print metrics
    # print("\nQ-Error test set:")
    # print_qerror(preds_test_unnorm, test_label_raw)

    # # Write predictions
    # results_file = os.path.join(results_dir, f"exp_{timestamp}.csv")
    # print(f"Writing test predictions to: {results_file}")
    # os.makedirs(os.path.dirname(results_file), exist_ok=True)
    # with open(results_file, "w") as f:
    #     for i in range(len(preds_test_unnorm)):
    #         f.write(str(preds_test_unnorm[i]) + "," + test_label_raw[i] + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="JSON config file path", required=True)
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (overrides config if provided)")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    if args.lr is not None:
        config["lr"] = args.lr

    train_and_predict(config)


if __name__ == "__main__":
    main()
