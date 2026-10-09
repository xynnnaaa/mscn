import argparse
import time
import os
import json
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
import datetime

from mscn.util import *
from mscn.data import get_train_datasets, load_data, make_dataset, load_and_encode_all_data
from mscn.model import SetConv
from mscn.mixture import mixture_options

import random
def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    print("80th percentile: {}".format(np.percentile(qerror, 80)))
    print("90th percentile: {}".format(np.percentile(qerror, 90)))
    print("95th percentile: {}".format(np.percentile(qerror, 95)))
    print("99th percentile: {}".format(np.percentile(qerror, 99)))
    print("Max: {}".format(np.max(qerror)))
    print("Mean: {}".format(np.mean(qerror)))

def print_qerror_from_array(qerror):
    print("Median: {:.4f}".format(np.median(qerror)))
    print("80th percentile: {:.4f}".format(np.percentile(qerror, 80)))
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

    mixture_stats = None
    with torch.no_grad():
        for data_batch in data_loader:
            samples, predicates, joins, join_samples, targets, s_mask, p_mask, j_mask = data_batch
            if cuda:
                samples, predicates, joins, join_samples, targets = samples.cuda(), predicates.cuda(), joins.cuda(), join_samples.cuda(), targets.cuda()
                s_mask, p_mask, j_mask = s_mask.cuda(), p_mask.cuda(), j_mask.cuda()
            
            outputs = model(samples, predicates, joins, join_samples, s_mask, p_mask, j_mask)
            if getattr(model, "use_adaptive_single_mixture", False):
                stats = model.single_mixture.last_stats
                mixture_stats = stats.clone() if mixture_stats is None else mixture_stats + stats


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

    if mixture_stats is not None:
        total, both, alpha_sum, only_q, only_r, neither = mixture_stats.cpu().tolist()
        mean_alpha = f"{alpha_sum / both:.4f}" if both else "n/a"
        print(f"Single mixture: tables={int(total)}, both={int(both)}, "
              f"mean alpha (both)={mean_alpha}, only QA={int(only_q)}, "
              f"only random={int(only_r)}, neither={int(neither)}")
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

    # --- 新增：读取是否包含未命中特征的开关 ---
    has_unmatched_embedding = config.get("has_unmatched_embedding", 0)

    results_dir = config.get("results_dir", "results") # 保存目录
    workloads_dir = config.get("workloads_dir", "workloads") # 数据集目录
    workload_name = config.get("workload_name", "imdb") # 数据集名称

    use_join_embedding = config.get("use_join_embedding", 0)
    seed = config.get("seed")

    model_id = f"{workload_name}_{trainset}_{lr}_{num_epochs}_{batch_size}"
    best_model_path = config.get("model_output_path")
    if best_model_path:
        model_output_dir = os.path.dirname(best_model_path)
        if model_output_dir:
            os.makedirs(model_output_dir, exist_ok=True)
    else:
        checkpoint_dir = "/data2/xuyining/learnedcardinalities/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
        best_model_path = os.path.join(checkpoint_dir, f"exp_{timestamp}_best.pt")
    print(f"Best model will be saved to: {best_model_path}")

    print(f"learning rate: {lr}")
    print(f"batch size: {batch_size}")
    print(f"hidden units: {hid_units}")
    print(f"epochs: {num_epochs}")
    if seed is not None:
        print(f"random seed: {seed}")

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
    if use_join_embedding in [1, 2, 3]:
        join_sample_feats = train_data[0][3].shape[0]
    else:
        join_sample_feats = 0 # 传入 0，模型会自动回退到 3 个头的结构

    # Train model
    # sample_feats = len(table2vec) + num_materialized_samples
    table_vec_size = len(table2vec)
    first_sample_tensor = train_data[0][0]
    total_sample_feats = first_sample_tensor.shape[1]
    sample_vec_size = total_sample_feats - table_vec_size

    predicate_feats = len(column2vec) + len(op2vec) + 1 + num_buckets
    join_feats = len(join2vec)

    print(f"Table One-Hot dims: {table_vec_size}, Sample (Emb/Bitmap) dims: {sample_vec_size}")
    print(f"Predicate features: {predicate_feats}")
    print(f"Has Unmatched Embedding: {has_unmatched_embedding == 1}")

    model = SetConv(table_vec_size, sample_vec_size, predicate_feats, join_feats, join_sample_feats, hid_units, use_single_embedding, has_unmatched_embedding=has_unmatched_embedding,
                    single_mixture_options=mixture_options(config))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # --- 新增：学习率调度器 ---
    # mode='min'：监控指标越小越好（针对 Loss）
    # factor=0.5：触发时学习率减半
    # patience=20：如果连续 20 个 epoch 验证集 Loss 都不降，就触发衰减
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

    if cuda:
        model.cuda()

    # g = torch.Generator()
    # g.manual_seed(config["seed"])

    # train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=g)
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    best_val_loss = float("inf")
    best_epoch = -1
    best_model_state = None

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

        print("\nEpoch {}, Train Loss: {:.4f}".format(epoch, loss_total / len(train_data_loader)))


        # 验证集评估
        _, val_loss_avg = get_metrics(model, val_data_loader, cuda, min_val, max_val)
        print("Epoch {}, Validation Loss: {:.4f}".format(epoch, val_loss_avg))
        # print(f"--- Epoch {epoch} Validation Set qerror ---")
        # print_qerror_from_array(val_qerrors)

        # # --- 新增：让 Scheduler 根据当前的 验证集 Loss 决定是否要减小学习率 ---
        # scheduler.step(val_loss_avg)
        #  # 当前学习率打印 (可选，方便你在日志里看它什么时候降了)
        # current_lr = optimizer.param_groups[0]['lr']
        # if current_lr != lr:
        #     print(f" [!] Learning Rate adjusted to: {current_lr}")
        #     lr = current_lr # 仅仅是为了防止重复打印

        # 只根据验证集 Loss 选择模型，测试集不参与 checkpoint 决策。
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_epoch = epoch
            print(f" --> New best model found at epoch {epoch} with validation loss: {val_loss_avg:.4f}")
            best_model_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_model_state is None:
        raise RuntimeError("No best model was selected. Ensure epochs is greater than zero.")

    # Optional diagnostic checkpoint, captured BEFORE restoring the best model.
    final_model_path = config.get("final_model_output_path")
    if final_model_path:
        if os.path.abspath(final_model_path) == os.path.abspath(best_model_path):
            raise ValueError("Final and best model output paths must differ")
        os.makedirs(os.path.dirname(os.path.abspath(final_model_path)), exist_ok=True)
        torch.save({name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()}, final_model_path)
        print(f"Final model saved to {final_model_path} (epoch {num_epochs - 1})")

    # checkpoint 中只保存验证集 Loss 最低模型的原始 state_dict。
    torch.save(best_model_state, best_model_path)
    if config.get("use_adaptive_single_mixture", False):
        # Preserve the state_dict checkpoint format; architecture/data settings live beside it.
        with open(best_model_path + ".config.json", "w", encoding="utf-8") as stream:
            json.dump(config, stream, indent=2, ensure_ascii=False)

    print(
        f"\nBest model saved to {best_model_path} "
        f"(epoch {best_epoch}, validation loss {best_val_loss:.4f})"
    )

    print("\n" + "="*60)
    print("FINAL SUMMARY & EVALUATIONS ON TEST SET")
    print("="*60)

    print(f"Best Validation Loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"\n>>> Evaluating best-validation model (Epoch: {best_epoch}) on TEST SET")
    model.load_state_dict(best_model_state)
    if cuda:
        model.cuda()

    t_qerrors, test_loss_avg = get_metrics(model, test_data_loader, cuda, min_val, max_val)
    print(f"Test Loss: {test_loss_avg:.4f}")
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
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides config if provided)")
    parser.add_argument("--batch", type=int, default=None, help="Batch size (overrides config if provided)")
    parser.add_argument("--model-output-path", default=None, help="Model output path (overrides config if provided)")
    seed_options = parser.add_mutually_exclusive_group()
    seed_options.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    seed_options.add_argument("--no-seed", action="store_true", help="Ignore any seed in the config; use an unseeded run")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None, help="Override config cuda setting")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch count override")
    parser.add_argument("--final-model-output-path", default=None,
                        help="Additionally save the final epoch state before restoring the best checkpoint")

    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    with open(args.config, 'r') as f:
        config = json.load(f)

    if args.lr is not None:
        config["lr"] = args.lr
    if args.batch is not None:
        config["batch"] = args.batch
    if args.model_output_path is not None:
        config["model_output_path"] = args.model_output_path

    if args.seed is not None:
        config["seed"] = args.seed
    if args.no_seed:
        config.pop("seed", None)
    if args.device is not None:
        config["cuda"] = args.device == "cuda"
    if args.epochs is not None:
        if args.epochs <= 0:
            parser.error("--epochs must be positive")
        config["epochs"] = args.epochs
    if args.final_model_output_path is not None:
        config["final_model_output_path"] = args.final_model_output_path
    if config.get("final_model_output_path") and config.get("model_output_path"):
        if os.path.abspath(config["final_model_output_path"]) == os.path.abspath(config["model_output_path"]):
            parser.error("Final and best model output paths must differ")

    train_and_predict(config)


if __name__ == "__main__":
    main()
