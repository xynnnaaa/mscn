import csv
import torch
from torch.utils.data import dataset
import numpy as np
import os

from mscn.util import *

def get_global_info(train_prefix, test_prefix, num_samples):
    """
    预扫描训练集和测试集，获取全局统一的字典和最大长度。
    """
    # 加载两个数据集的原始数据（不包含bitmap/embedding，只看结构）
    def load_meta(prefix):
        ts, js, ps = [], [], []
        with open(prefix + ".csv", 'r') as f:
            data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))
            for row in data_raw:
                ts.append(row[0].split(','))
                js.append(row[1].split(','))
                ps.append(row[2].split(','))
        ps = [list(chunks(d, 3)) for d in ps]
        return ts, js, ps

    train_tables, train_joins, train_preds = load_meta(train_prefix)
    test_tables, test_joins, test_preds = load_meta(test_prefix)

    # 汇总所有表名、列名、操作符、Join条件
    all_tables = train_tables + test_tables
    all_joins = train_joins + test_joins
    all_preds = train_preds + test_preds

    # 获取全局字典
    table2vec, _ = get_set_encoding(get_all_table_names(all_tables))
    column2vec, _ = get_set_encoding(get_all_column_names(all_preds))
    op2vec, _ = get_set_encoding(get_all_operators(all_preds))
    join2vec, _ = get_set_encoding(get_all_joins(all_joins))

    # 获取全局最大长度 (用于 Padding)
    max_num_joins = max(max([len(j) for j in train_joins]), max([len(j) for j in test_joins]))
    max_num_predicates = max(max([len(p) for p in train_preds]), max([len(p) for p in test_preds]))

    # 注意：MSCN的sample部分长度 = num_joins + 1
    max_num_samples = max_num_joins + 1

    return (table2vec, column2vec, op2vec, join2vec), (max_num_joins, max_num_predicates)


def load_data(file_prefix, num_materialized_samples, use_single_embedding=0, embedding_file=None):
    joins = []
    predicates = []
    tables = []
    samples = []
    label = []

    # Load queries
    csv_path = file_prefix + ".csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        exit(1)

    with open(csv_path, 'r') as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))
        for row in data_raw:
            tables.append(row[0].split(','))
            joins.append(row[1].split(','))
            predicates.append(row[2].split(','))
            if int(row[3]) < 1:
                print("Queries must have non-zero cardinalities")
                exit(1)
            label.append(row[3])
    print("Loaded queries")

    if use_single_embedding == 1:
        if embedding_file and os.path.exists(embedding_file):
            # 加载 .pt 文件：{ seq_id: {alias: tensor} }
            samples = torch.load(embedding_file)
            print(f"Loaded embeddings from {embedding_file}")
        else:
            print(f"Error: Embedding file {embedding_file} not found.")
            exit(1)
    else:
        # Load bitmaps
        bitmap_path = file_prefix + ".bitmaps"
        num_bytes_per_bitmap = int((num_materialized_samples + 7) >> 3)
        with open(bitmap_path, 'rb') as f:
            for i in range(len(tables)):
                four_bytes = f.read(4)
                if not four_bytes:
                    print("Error while reading 'four_bytes'")
                    exit(1)
                num_bitmaps_curr_query = int.from_bytes(four_bytes, byteorder='little')
                bitmaps = np.empty((num_bitmaps_curr_query, num_bytes_per_bitmap * 8), dtype=np.uint8)
                for j in range(num_bitmaps_curr_query):
                    # Read bitmap
                    bitmap_bytes = f.read(num_bytes_per_bitmap)
                    if not bitmap_bytes:
                        print("Error while reading 'bitmap_bytes'")
                        exit(1)
                    bitmaps[j] = np.unpackbits(np.frombuffer(bitmap_bytes, dtype=np.uint8))
                samples.append(bitmaps)
        print("Loaded bitmaps")

    # Split predicates
    predicates = [list(chunks(d, 3)) for d in predicates]

    return joins, predicates, tables, samples, label


def load_and_encode_train_data(config):
    workload_dir = config.get("workloads_dir", "workloads") # 数据集目录
    trainset = config["trainset"]
    column_min_max_path = config["column_min_max_file"]
    num_materialized_samples = config.get("num_samples", 1000)
    num_hash_buckets = config.get("num_buckets", 16)

    use_single_embedding = config.get("use_single_embedding", 0)
    train_emb_path = config.get("train_embedding_file", "")

    train_prefix = os.path.join(workload_dir, trainset)

    joins, predicates, tables, samples, label = load_data(train_prefix, num_materialized_samples, use_single_embedding, train_emb_path)

    # Get column name dict
    column_names = get_all_column_names(predicates)
    column2vec, idx2column = get_set_encoding(column_names)

    # Get table name dict
    table_names = get_all_table_names(tables)
    table2vec, idx2table = get_set_encoding(table_names)

    # Get operator name dict
    operators = get_all_operators(predicates)
    op2vec, idx2op = get_set_encoding(operators)

    # Get join name dict
    join_set = get_all_joins(joins)
    join2vec, idx2join = get_set_encoding(join_set)

    # Get min and max values for each column
    with open(column_min_max_path, 'r') as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter=','))
        column_min_max_vals = {}
        for i, row in enumerate(data_raw):
            if i == 0:
                continue
            column_min_max_vals[row[0]] = [float(row[1]), float(row[2])]

    # Get feature encoding and proper normalization
    samples_enc = encode_samples(tables, samples, table2vec, use_single_embedding)
    predicates_enc, joins_enc = encode_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec, num_hash_buckets=num_hash_buckets)
    label_norm, min_val, max_val = normalize_labels(label)

    # Split in training and validation samples
    num_queries = len(label_norm)
    num_train = int(num_queries * 0.9)
    num_val = num_queries - num_train

    samples_train = samples_enc[:num_train]
    predicates_train = predicates_enc[:num_train]
    joins_train = joins_enc[:num_train]
    labels_train = label_norm[:num_train]

    samples_val = samples_enc[num_train:num_train + num_val]
    predicates_val = predicates_enc[num_train:num_train + num_val]
    joins_val = joins_enc[num_train:num_train + num_val]
    labels_val = label_norm[num_train:num_train + num_val]

    print("Number of training samples: {}".format(len(labels_train)))
    print("Number of validation samples: {}".format(len(labels_val)))

    max_num_joins = max(max([len(j) for j in joins_train]), max([len(j) for j in joins_val]))
    max_num_predicates = max(max([len(p) for p in predicates_train]), max([len(p) for p in predicates_val]))

    dicts = [table2vec, column2vec, op2vec, join2vec]
    train_data = [samples_train, predicates_train, joins_train]
    val_data = [samples_val, predicates_val, joins_val]
    return dicts, column_min_max_vals, min_val, max_val, labels_train, labels_val, max_num_joins, max_num_predicates, train_data, val_data


def load_and_encode_all_data(config):
    """
    整合后的加载函数，确保训练/测试维度完全一致
    """
    workloads_dir = config["workloads_dir"]
    train_prefix = os.path.join(workloads_dir, config["trainset"])
    test_prefix = os.path.join(workloads_dir, config["testset"])
    num_samples = config.get("num_samples", 1000)
    use_single_embedding = config.get("use_single_embedding", 0)
    num_hash_buckets = config.get("num_buckets", 16)

    # join embedding
    use_join_embedding = config.get("use_join_embedding", 0)
    train_join_emb_path = config.get("train_join_embedding_file", "")
    test_join_emb_path = config.get("test_join_embedding_file", "")

    if use_join_embedding == 1:
        print(f"train_join_emb_path: {train_join_emb_path}")
        print(f"test_join_emb_path: {test_join_emb_path}")

    # 1. 获取全局字典和最大长度
    dicts, max_lengths = get_global_info(train_prefix, test_prefix, num_samples)
    table2vec, column2vec, op2vec, join2vec = dicts
    max_num_joins, max_num_predicates = max_lengths

    # 2. 读取 Min-Max 归一化文件
    with open(config["column_min_max_file"], 'r') as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter=','))
        column_min_max_vals = {row[0]: [float(row[1]), float(row[2])] for i, row in enumerate(data_raw) if i > 0}

    # 3. 加载并编码 训练/验证集
    t_joins, t_preds, t_tables, t_samples, t_label = load_data(
        train_prefix, num_samples, use_single_embedding, config.get("train_embedding_file")
    )
    t_samples_enc = encode_samples(t_tables, t_samples, table2vec, use_single_embedding)
    t_preds_enc, t_joins_enc = encode_data(t_preds, t_joins, column_min_max_vals, column2vec, op2vec, join2vec, num_hash_buckets)
    label_norm, min_val, max_val = normalize_labels(t_label)

    if use_join_embedding == 1:
        train_js_all = torch.load(train_join_emb_path)
        train_js_enc = np.array([train_js_all[i].cpu().numpy() for i in range(len(t_label))], dtype=np.float32)
    else:
        # 即使不用，也生成一个 dummy 占位符 [N, 1]
        train_js_enc = np.zeros((len(t_label), 1), dtype=np.float32)

    # 训练/验证切分
    num_queries = len(label_norm)
    num_train = int(num_queries * 0.9)
    train_data = [t_samples_enc[:num_train], t_preds_enc[:num_train], t_joins_enc[:num_train]]
    val_data = [t_samples_enc[num_train:], t_preds_enc[num_train:], t_joins_enc[num_train:]]
    labels_train = label_norm[:num_train]
    labels_val = label_norm[num_train:]

    # 4. 加载并编码 测试集
    test_emb_path = config.get("test_embedding_file", "")
    test_joins, test_preds, test_tables, test_samples, test_label_raw = load_data(
        test_prefix, num_samples, use_single_embedding, test_emb_path
    )
    test_samples_enc = encode_samples(test_tables, test_samples, table2vec, use_single_embedding)
    test_preds_enc, test_joins_enc = encode_data(test_preds, test_joins, column_min_max_vals, column2vec, op2vec, join2vec, num_hash_buckets)
    test_label_norm, _, _ = normalize_labels(test_label_raw, min_val, max_val)

    if use_join_embedding == 1:
        test_js_all = torch.load(test_join_emb_path)
        test_js_enc = np.array([test_js_all[i].cpu().numpy() for i in range(len(test_label_raw))], dtype=np.float32)
    else:
        test_js_enc = np.zeros((len(test_label_raw), 1), dtype=np.float32)

    # 封装为 TensorDataset
    train_dataset = make_dataset(*train_data, join_samples=train_js_enc[:num_train], labels=labels_train, max_num_joins=max_num_joins, max_num_predicates=max_num_predicates)
    val_dataset = make_dataset(*val_data, join_samples=train_js_enc[num_train:], labels=labels_val, max_num_joins=max_num_joins, max_num_predicates=max_num_predicates)
    test_dataset = make_dataset(test_samples_enc, test_preds_enc, test_joins_enc, test_js_enc, test_label_norm, max_num_joins, max_num_predicates)

    return dicts, column_min_max_vals, min_val, max_val, train_dataset, val_dataset, test_dataset, (max_num_joins, max_num_predicates), test_label_raw


def make_dataset(samples, predicates, joins, join_samples, labels, max_num_joins, max_num_predicates):
    """Add zero-padding and wrap as tensor dataset."""

    sample_masks = []
    sample_tensors = []
    for sample in samples:
        sample_tensor = np.vstack(sample)
        num_pad = max_num_joins + 1 - sample_tensor.shape[0]
        sample_mask = np.ones_like(sample_tensor).mean(1, keepdims=True)
        sample_tensor = np.pad(sample_tensor, ((0, num_pad), (0, 0)), 'constant')
        sample_mask = np.pad(sample_mask, ((0, num_pad), (0, 0)), 'constant')
        sample_tensors.append(np.expand_dims(sample_tensor, 0))
        sample_masks.append(np.expand_dims(sample_mask, 0))
    sample_tensors = np.vstack(sample_tensors)
    sample_tensors = torch.FloatTensor(sample_tensors)
    sample_masks = np.vstack(sample_masks)
    sample_masks = torch.FloatTensor(sample_masks)

    predicate_masks = []
    predicate_tensors = []
    for predicate in predicates:
        predicate_tensor = np.vstack(predicate)
        num_pad = max_num_predicates - predicate_tensor.shape[0]
        predicate_mask = np.ones_like(predicate_tensor).mean(1, keepdims=True)
        predicate_tensor = np.pad(predicate_tensor, ((0, num_pad), (0, 0)), 'constant')
        predicate_mask = np.pad(predicate_mask, ((0, num_pad), (0, 0)), 'constant')
        predicate_tensors.append(np.expand_dims(predicate_tensor, 0))
        predicate_masks.append(np.expand_dims(predicate_mask, 0))
    predicate_tensors = np.vstack(predicate_tensors)
    predicate_tensors = torch.FloatTensor(predicate_tensors)
    predicate_masks = np.vstack(predicate_masks)
    predicate_masks = torch.FloatTensor(predicate_masks)

    join_masks = []
    join_tensors = []
    for join in joins:
        join_tensor = np.vstack(join)
        num_pad = max_num_joins - join_tensor.shape[0]
        join_mask = np.ones_like(join_tensor).mean(1, keepdims=True)
        join_tensor = np.pad(join_tensor, ((0, num_pad), (0, 0)), 'constant')
        join_mask = np.pad(join_mask, ((0, num_pad), (0, 0)), 'constant')
        join_tensors.append(np.expand_dims(join_tensor, 0))
        join_masks.append(np.expand_dims(join_mask, 0))
    join_tensors = np.vstack(join_tensors)
    join_tensors = torch.FloatTensor(join_tensors)
    join_masks = np.vstack(join_masks)
    join_masks = torch.FloatTensor(join_masks)

    join_sample_tensors = torch.FloatTensor(join_samples)

    target_tensor = torch.FloatTensor(labels)

    return dataset.TensorDataset(sample_tensors, predicate_tensors, join_tensors, join_sample_tensors, target_tensor, sample_masks,
                                 predicate_masks, join_masks)


def get_train_datasets(config):
    dicts, column_min_max_vals, min_val, max_val, labels_train, labels_val, max_num_joins, max_num_predicates, train_data, val_data = load_and_encode_train_data(config)
    train_dataset = make_dataset(*train_data, labels=labels_train, max_num_joins=max_num_joins,
                                 max_num_predicates=max_num_predicates)
    print("Created TensorDataset for training data")
    val_dataset = make_dataset(*val_data, labels=labels_val, max_num_joins=max_num_joins,
                                max_num_predicates=max_num_predicates)
    print("Created TensorDataset for validation data")
    return dicts, column_min_max_vals, min_val, max_val, labels_train, labels_val, max_num_joins, max_num_predicates, train_dataset, val_dataset
