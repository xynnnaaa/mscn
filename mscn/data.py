# import csv
# import torch
# from torch.utils.data import dataset
# import numpy as np
# import os

# from mscn.util import *

# def get_global_info(train_prefix, test_prefix, num_samples):
#     """
#     预扫描训练集和测试集，获取全局统一的字典和最大长度。
#     """
#     # 加载两个数据集的原始数据（不包含bitmap/embedding，只看结构）
#     def load_meta(prefix):
#         ts, js, ps = [], [], []
#         with open(prefix + ".csv", 'r') as f:
#             data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))
#             for row in data_raw:
#                 ts.append(row[0].split(','))
#                 js.append(row[1].split(','))
#                 ps.append(('#'.join(row[2:-1])).split(','))
#         ps = [list(chunks(d, 3)) for d in ps]
#         return ts, js, ps

#     train_tables, train_joins, train_preds = load_meta(train_prefix)
#     test_tables, test_joins, test_preds = load_meta(test_prefix)

#     # 汇总所有表名、列名、操作符、Join条件
#     all_tables = train_tables + test_tables
#     all_joins = train_joins + test_joins
#     all_preds = train_preds + test_preds

#     # 获取全局字典
#     table2vec, _ = get_set_encoding(get_all_table_names(all_tables))
#     column2vec, _ = get_set_encoding(get_all_column_names(all_preds))
#     op2vec, _ = get_set_encoding(get_all_operators(all_preds))
#     join2vec, _ = get_set_encoding(get_all_joins(all_joins))

#     # 获取全局最大长度 (用于 Padding)
#     max_num_joins = max(max([len(j) for j in train_joins]), max([len(j) for j in test_joins]))
#     max_num_predicates = max(max([len(p) for p in train_preds]), max([len(p) for p in test_preds]))

#     # 注意：MSCN的sample部分长度 = num_joins + 1
#     max_num_samples = max_num_joins + 1

#     return (table2vec, column2vec, op2vec, join2vec), (max_num_joins, max_num_predicates)


# def load_data(file_prefix, num_materialized_samples, use_single_embedding=0, embedding_file=None):
#     joins = []
#     predicates = []
#     tables = []
#     samples = []
#     label = []

#     # Load queries
#     csv_path = file_prefix + ".csv"
#     if not os.path.exists(csv_path):
#         print(f"Error: {csv_path} not found.")
#         exit(1)

#     with open(csv_path, 'r') as f:
#         data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))
#         for row in data_raw:
#             tables.append(row[0].split(','))
#             joins.append(row[1].split(','))
#             predicates_str = '#'.join(row[2:-1])
#             predicates.append(predicates_str.split(','))
#             # predicates.append(row[2].split(','))
#             if int(row[-1]) < 1:
#                 print("Queries must have non-zero cardinalities")
#                 exit(1)
#             label.append(row[-1])
#     print("Loaded queries")

#     if use_single_embedding == 1:
#         if embedding_file and os.path.exists(embedding_file):
#             # 加载 .pt 文件：{ seq_id: {alias: tensor} }
#             samples = torch.load(embedding_file)
#             print(f"Loaded embeddings from {embedding_file}")
#         else:
#             print(f"Error: Embedding file {embedding_file} not found.")
#             exit(1)
#     else:
#         # Load bitmaps
#         if num_materialized_samples == 2000:
#             bitmap_path = file_prefix + "-2k.bitmaps"
#         elif num_materialized_samples == 3000:
#             bitmap_path = file_prefix + "-3k.bitmaps"
#         elif num_materialized_samples == 4000:
#             bitmap_path = file_prefix + "-4k.bitmaps"
#         elif num_materialized_samples == 5000:
#             bitmap_path = file_prefix + "-5k.bitmaps"
#         elif num_materialized_samples == 500:
#             bitmap_path = file_prefix + "-qa.bitmaps"
#         else:
#             bitmap_path = file_prefix + "-qa.bitmaps"
#         print(f"Loading bitmaps from {bitmap_path}...")
#         num_bytes_per_bitmap = int((num_materialized_samples + 7) >> 3)
#         with open(bitmap_path, 'rb') as f:
#             for i in range(len(tables)):
#                 four_bytes = f.read(4)
#                 if not four_bytes:
#                     print("Error while reading 'four_bytes'")
#                     exit(1)
#                 num_bitmaps_curr_query = int.from_bytes(four_bytes, byteorder='little')
#                 bitmaps = np.empty((num_bitmaps_curr_query, num_bytes_per_bitmap * 8), dtype=np.uint8)
#                 for j in range(num_bitmaps_curr_query):
#                     # Read bitmap
#                     bitmap_bytes = f.read(num_bytes_per_bitmap)
#                     if not bitmap_bytes:
#                         print("Error while reading 'bitmap_bytes'")
#                         exit(1)
#                     bitmaps[j] = np.unpackbits(np.frombuffer(bitmap_bytes, dtype=np.uint8))
#                 samples.append(bitmaps)
#         print("Loaded bitmaps")

#     # Split predicates
#     predicates = [list(chunks(d, 3)) for d in predicates]

#     return joins, predicates, tables, samples, label


# def load_and_encode_train_data(config):
#     workload_dir = config.get("workloads_dir", "workloads") # 数据集目录
#     trainset = config["trainset"]
#     column_min_max_path = config["column_min_max_file"]
#     num_materialized_samples = config.get("num_samples", 1000)
#     num_hash_buckets = config.get("num_buckets", 16)

#     use_single_embedding = config.get("use_single_embedding", 0)
#     train_emb_path = config.get("train_embedding_file", "")

#     train_prefix = os.path.join(workload_dir, trainset)

#     joins, predicates, tables, samples, label = load_data(train_prefix, num_materialized_samples, use_single_embedding, train_emb_path)

#     # Get column name dict
#     column_names = get_all_column_names(predicates)
#     column2vec, idx2column = get_set_encoding(column_names)

#     # Get table name dict
#     table_names = get_all_table_names(tables)
#     table2vec, idx2table = get_set_encoding(table_names)

#     # Get operator name dict
#     operators = get_all_operators(predicates)
#     op2vec, idx2op = get_set_encoding(operators)

#     # Get join name dict
#     join_set = get_all_joins(joins)
#     join2vec, idx2join = get_set_encoding(join_set)

#     # Get min and max values for each column
#     with open(column_min_max_path, 'r') as f:
#         data_raw = list(list(rec) for rec in csv.reader(f, delimiter=','))
#         column_min_max_vals = {}
#         for i, row in enumerate(data_raw):
#             if i == 0:
#                 continue
#             column_min_max_vals[row[0]] = [float(row[1]), float(row[2])]

#     # Get feature encoding and proper normalization
#     samples_enc = encode_samples(tables, samples, table2vec, use_single_embedding)
#     predicates_enc, joins_enc = encode_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec, num_hash_buckets=num_hash_buckets)
#     label_norm, min_val, max_val = normalize_labels(label)

#     # Split in training and validation samples
#     num_queries = len(label_norm)
#     num_train = int(num_queries * 0.9)
#     num_val = num_queries - num_train

#     samples_train = samples_enc[:num_train]
#     predicates_train = predicates_enc[:num_train]
#     joins_train = joins_enc[:num_train]
#     labels_train = label_norm[:num_train]

#     samples_val = samples_enc[num_train:num_train + num_val]
#     predicates_val = predicates_enc[num_train:num_train + num_val]
#     joins_val = joins_enc[num_train:num_train + num_val]
#     labels_val = label_norm[num_train:num_train + num_val]

#     print("Number of training samples: {}".format(len(labels_train)))
#     print("Number of validation samples: {}".format(len(labels_val)))

#     max_num_joins = max(max([len(j) for j in joins_train]), max([len(j) for j in joins_val]))
#     max_num_predicates = max(max([len(p) for p in predicates_train]), max([len(p) for p in predicates_val]))

#     dicts = [table2vec, column2vec, op2vec, join2vec]
#     train_data = [samples_train, predicates_train, joins_train]
#     val_data = [samples_val, predicates_val, joins_val]
#     return dicts, column_min_max_vals, min_val, max_val, labels_train, labels_val, max_num_joins, max_num_predicates, train_data, val_data


# def load_and_encode_all_data(config):
#     """
#     整合后的加载函数，确保训练/测试维度完全一致
#     """
#     workloads_dir = config["workloads_dir"]
#     train_prefix = os.path.join(workloads_dir, config["trainset"])
#     test_prefix = os.path.join(workloads_dir, config["testset"])
#     num_samples = config.get("num_samples", 1000)
#     use_single_embedding = config.get("use_single_embedding", 0)
#     num_hash_buckets = config.get("num_buckets", 16)

#     # join embedding
#     use_join_embedding = config.get("use_join_embedding", 0)
#     train_join_emb_path = config.get("train_join_embedding_file", "")
#     test_join_emb_path = config.get("test_join_embedding_file", "")

#     if use_join_embedding == 1:
#         print(f"train_join_emb_path: {train_join_emb_path}")
#         print(f"test_join_emb_path: {test_join_emb_path}")

#     # 1. 获取全局字典和最大长度
#     dicts, max_lengths = get_global_info(train_prefix, test_prefix, num_samples)
#     table2vec, column2vec, op2vec, join2vec = dicts
#     max_num_joins, max_num_predicates = max_lengths

#     # 2. 读取 Min-Max 归一化文件
#     with open(config["column_min_max_file"], 'r') as f:
#         data_raw = list(list(rec) for rec in csv.reader(f, delimiter=','))
#         column_min_max_vals = {row[0]: [float(row[1]), float(row[2])] for i, row in enumerate(data_raw) if i > 0}

#     # 3. 加载并编码 训练/验证集
#     t_joins, t_preds, t_tables, t_samples, t_label = load_data(
#         train_prefix, num_samples, use_single_embedding, config.get("train_embedding_file")
#     )
#     t_samples_enc = encode_samples(t_tables, t_samples, table2vec, use_single_embedding)
#     t_preds_enc, t_joins_enc = encode_data(t_preds, t_joins, column_min_max_vals, column2vec, op2vec, join2vec, num_hash_buckets)
#     label_norm, min_val, max_val = normalize_labels(t_label)

#     if use_join_embedding == 1:
#         train_js_all = torch.load(train_join_emb_path)
#         train_js_enc = np.array([train_js_all[i].cpu().numpy() for i in range(len(t_label))], dtype=np.float32)
#     else:
#         # 即使不用，也生成一个 dummy 占位符 [N, 1]
#         train_js_enc = np.zeros((len(t_label), 1), dtype=np.float32)

#     # 训练/验证切分
#     num_queries = len(label_norm)
#     num_train = int(num_queries * 0.9)
#     train_data = [t_samples_enc[:num_train], t_preds_enc[:num_train], t_joins_enc[:num_train]]
#     val_data = [t_samples_enc[num_train:], t_preds_enc[num_train:], t_joins_enc[num_train:]]
#     labels_train = label_norm[:num_train]
#     labels_val = label_norm[num_train:]

#     # 4. 加载并编码 测试集
#     test_emb_path = config.get("test_embedding_file", "")
#     test_joins, test_preds, test_tables, test_samples, test_label_raw = load_data(
#         test_prefix, num_samples, use_single_embedding, test_emb_path
#     )
#     test_samples_enc = encode_samples(test_tables, test_samples, table2vec, use_single_embedding)
#     test_preds_enc, test_joins_enc = encode_data(test_preds, test_joins, column_min_max_vals, column2vec, op2vec, join2vec, num_hash_buckets)
#     test_label_norm, _, _ = normalize_labels(test_label_raw, min_val, max_val)

#     if use_join_embedding == 1:
#         test_js_all = torch.load(test_join_emb_path)
#         test_js_enc = np.array([test_js_all[i].cpu().numpy() for i in range(len(test_label_raw))], dtype=np.float32)
#     else:
#         test_js_enc = np.zeros((len(test_label_raw), 1), dtype=np.float32)

#     # 封装为 TensorDataset
#     train_dataset = make_dataset(*train_data, join_samples=train_js_enc[:num_train], labels=labels_train, max_num_joins=max_num_joins, max_num_predicates=max_num_predicates)
#     val_dataset = make_dataset(*val_data, join_samples=train_js_enc[num_train:], labels=labels_val, max_num_joins=max_num_joins, max_num_predicates=max_num_predicates)
#     test_dataset = make_dataset(test_samples_enc, test_preds_enc, test_joins_enc, test_js_enc, test_label_norm, max_num_joins, max_num_predicates)

#     return dicts, column_min_max_vals, min_val, max_val, train_dataset, val_dataset, test_dataset, (max_num_joins, max_num_predicates), test_label_raw


# def make_dataset(samples, predicates, joins, join_samples, labels, max_num_joins, max_num_predicates):
#     """Add zero-padding and wrap as tensor dataset."""

#     sample_masks = []
#     sample_tensors = []
#     for sample in samples:
#         sample_tensor = np.vstack(sample)
#         num_pad = max_num_joins + 1 - sample_tensor.shape[0]
#         sample_mask = np.ones_like(sample_tensor).mean(1, keepdims=True)
#         sample_tensor = np.pad(sample_tensor, ((0, num_pad), (0, 0)), 'constant')
#         sample_mask = np.pad(sample_mask, ((0, num_pad), (0, 0)), 'constant')
#         sample_tensors.append(np.expand_dims(sample_tensor, 0))
#         sample_masks.append(np.expand_dims(sample_mask, 0))
#     sample_tensors = np.vstack(sample_tensors)
#     sample_tensors = torch.FloatTensor(sample_tensors)
#     sample_masks = np.vstack(sample_masks)
#     sample_masks = torch.FloatTensor(sample_masks)

#     predicate_masks = []
#     predicate_tensors = []
#     for predicate in predicates:
#         predicate_tensor = np.vstack(predicate)
#         num_pad = max_num_predicates - predicate_tensor.shape[0]
#         predicate_mask = np.ones_like(predicate_tensor).mean(1, keepdims=True)
#         predicate_tensor = np.pad(predicate_tensor, ((0, num_pad), (0, 0)), 'constant')
#         predicate_mask = np.pad(predicate_mask, ((0, num_pad), (0, 0)), 'constant')
#         predicate_tensors.append(np.expand_dims(predicate_tensor, 0))
#         predicate_masks.append(np.expand_dims(predicate_mask, 0))
#     predicate_tensors = np.vstack(predicate_tensors)
#     predicate_tensors = torch.FloatTensor(predicate_tensors)
#     predicate_masks = np.vstack(predicate_masks)
#     predicate_masks = torch.FloatTensor(predicate_masks)

#     join_masks = []
#     join_tensors = []
#     for join in joins:
#         join_tensor = np.vstack(join)
#         num_pad = max_num_joins - join_tensor.shape[0]
#         join_mask = np.ones_like(join_tensor).mean(1, keepdims=True)
#         join_tensor = np.pad(join_tensor, ((0, num_pad), (0, 0)), 'constant')
#         join_mask = np.pad(join_mask, ((0, num_pad), (0, 0)), 'constant')
#         join_tensors.append(np.expand_dims(join_tensor, 0))
#         join_masks.append(np.expand_dims(join_mask, 0))
#     join_tensors = np.vstack(join_tensors)
#     join_tensors = torch.FloatTensor(join_tensors)
#     join_masks = np.vstack(join_masks)
#     join_masks = torch.FloatTensor(join_masks)

#     join_sample_tensors = torch.FloatTensor(join_samples)

#     target_tensor = torch.FloatTensor(labels)

#     return dataset.TensorDataset(sample_tensors, predicate_tensors, join_tensors, join_sample_tensors, target_tensor, sample_masks,
#                                  predicate_masks, join_masks)


# def get_train_datasets(config):
#     dicts, column_min_max_vals, min_val, max_val, labels_train, labels_val, max_num_joins, max_num_predicates, train_data, val_data = load_and_encode_train_data(config)
#     train_dataset = make_dataset(*train_data, labels=labels_train, max_num_joins=max_num_joins,
#                                  max_num_predicates=max_num_predicates)
#     print("Created TensorDataset for training data")
#     val_dataset = make_dataset(*val_data, labels=labels_val, max_num_joins=max_num_joins,
#                                 max_num_predicates=max_num_predicates)
#     print("Created TensorDataset for validation data")
#     return dicts, column_min_max_vals, min_val, max_val, labels_train, labels_val, max_num_joins, max_num_predicates, train_dataset, val_dataset






import csv
import torch
from torch.utils.data import dataset
import numpy as np
import os

from mscn.util import *

def get_global_info(train_prefix, test_prefix, num_samples, encoding_test_prefix=None):
    """
    预扫描训练集和测试集，获取全局统一的字典和最大长度。
    """
    def load_meta(prefix):
        ts, js, ps = [], [], []
        with open(prefix + ".csv", 'r') as f:
            data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))
            for row in data_raw:
                ts.append(row[0].split(','))
                js.append(row[1].split(','))
                ps.append(('#'.join(row[2:-1])).split(','))
        ps = [list(chunks(d, 3)) for d in ps]
        return ts, js, ps

    train_tables, train_joins, train_preds = load_meta(train_prefix)
    test_tables, test_joins, test_preds = load_meta(test_prefix)

    # A trained checkpoint fixes the one-hot feature dimensions and ordering.
    # During drift evaluation, derive vocabularies from the same reference test
    # set used for training, while still sizing padding from the current test set.
    if encoding_test_prefix:
        vocab_test_tables, vocab_test_joins, vocab_test_preds = load_meta(encoding_test_prefix)
    else:
        vocab_test_tables, vocab_test_joins, vocab_test_preds = test_tables, test_joins, test_preds

    all_tables = train_tables + vocab_test_tables
    all_joins = train_joins + vocab_test_joins
    all_preds = train_preds + vocab_test_preds

    table2vec, _ = get_set_encoding(get_all_table_names(all_tables))
    column2vec, _ = get_set_encoding(get_all_column_names(all_preds))
    op2vec, _ = get_set_encoding(get_all_operators(all_preds))
    join2vec, _ = get_set_encoding(get_all_joins(all_joins))

    max_num_joins = max(max([len(j) for j in train_joins]), max([len(j) for j in test_joins]))
    max_num_predicates = max(max([len(p) for p in train_preds]), max([len(p) for p in test_preds]))
    max_num_samples = max_num_joins + 1

    return (table2vec, column2vec, op2vec, join2vec), (max_num_joins, max_num_predicates)


def load_data(
    file_prefix,
    num_materialized_samples,
    use_single_embedding=0,
    embedding_file=None,
    bitmap_file=None,
):
    joins = []
    predicates = []
    tables = []
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
            predicates_str = '#'.join(row[2:-1])
            predicates.append(predicates_str.split(','))
            if int(row[-1]) < 1:
                print("Queries must have non-zero cardinalities")
                exit(1)
            label.append(row[-1])
    print("Loaded queries")

    # 💡 [重大重构]：无论传统模式还是混合模式，均必须首先加载原始 Bitmaps 保底硬信号
    if num_materialized_samples == 2000:
        bitmap_path = file_prefix + "-2k.bitmaps"
    elif num_materialized_samples == 3000:
        bitmap_path = file_prefix + "-3k.bitmaps"
    elif num_materialized_samples == 4000:
        bitmap_path = file_prefix + "-4k.bitmaps"
    elif num_materialized_samples == 500:
        bitmap_path = file_prefix + "-500.bitmaps"
    else:
        bitmap_path = file_prefix + ".bitmaps"

    if bitmap_file:
        bitmap_path = bitmap_file

    if use_single_embedding == -1:
        print("Mode -1: Skipping Bitmap and Embedding loading (No single table sample features).")
        samples = None

    else:
        print(f"Loading bitmaps from {bitmap_path}...")
        num_bytes_per_bitmap = int((num_materialized_samples + 7) >> 3)
        bitmaps_list = []
        with open(bitmap_path, 'rb') as f:
            for i in range(len(tables)):
                four_bytes = f.read(4)
                if not four_bytes:
                    print("Error while reading 'four_bytes'")
                    exit(1)
                num_bitmaps_curr_query = int.from_bytes(four_bytes, byteorder='little')
                bitmaps = np.empty((num_bitmaps_curr_query, num_bytes_per_bitmap * 8), dtype=np.uint8)
                for j in range(num_bitmaps_curr_query):
                    bitmap_bytes = f.read(num_bytes_per_bitmap)
                    if not bitmap_bytes:
                        print("Error while reading 'bitmap_bytes'")
                        exit(1)
                    bitmaps[j] = np.unpackbits(np.frombuffer(bitmap_bytes, dtype=np.uint8))
                bitmaps_list.append(bitmaps)
        print("Loaded bitmaps")

        if use_single_embedding in [1, 2]:
            if embedding_file and os.path.exists(embedding_file):
                emb_dict = torch.load(embedding_file, map_location="cpu")
                print(f"Loaded embeddings from {embedding_file}")
                # 混合打包返回：将 Bitmaps 数组和高维 Emb 字典打包到一起
                samples = (bitmaps_list, emb_dict)
            else:
                print(f"Error: Embedding file {embedding_file} not found.")
                exit(1)
        else:
            samples = bitmaps_list

    predicates = [list(chunks(d, 3)) for d in predicates]
    return joins, predicates, tables, samples, label


def hybrid_encode_samples(tables, samples, table2vec, use_single_embedding):
    """
    混合特征编码：
    mode 1: [Table One-Hot] + [Bitmap] + [剥离 log_cnt 的 Embedding]
    mode 2: [Table One-Hot] + [包含 log_cnt 的完整 Embedding]
    """

    # 新增：Mode -1 (只使用表名 One-Hot)
    if use_single_embedding == -1:
        samples_enc = []
        for i, query in enumerate(tables):
            query_features = []
            for table in query:
                # 只填入 Table One-Hot
                sample_vec = np.array(table2vec[table], dtype=np.float32)
                query_features.append(sample_vec)
            samples_enc.append(query_features)
        return samples_enc

    if use_single_embedding in [1, 2]:
        bitmaps_list, emb_dict = samples
        samples_enc = []
        
        for i, query in enumerate(tables):
            query_bitmaps = bitmaps_list[i]
            # 兼容 int 和 str 类型的 seq_id 键
            alias_dict = emb_dict.get(i) or emb_dict.get(str(i))
            if alias_dict is None:
                raise KeyError(f"Error: seq_id {i} not found in embedding file.")
                
            query_features = []
            for j, table in enumerate(query):
                sample_vec = []
                
                # 1. 注入 Table One-Hot 向量（严格对齐原版，使用完整的 table 字符串）
                sample_vec.append(table2vec[table])
                
                if use_single_embedding == 1:
                    # 2. 注入 Bitmap 保底硬特征
                    sample_vec.append(query_bitmaps[j])
                
                # 3. 严格对齐原版：提取真正的别名 (Alias)
                parts = table.strip().split(' ')
                alias = parts[1] if len(parts) > 1 else parts[0]
                
                # 4. 获取高维 Embedding 向量并转为 numpy
                embedding_vec = alias_dict[alias]
                if hasattr(embedding_vec, 'cpu'):
                    embedding_vec = embedding_vec.cpu().numpy()
                elif isinstance(embedding_vec, torch.Tensor):
                    embedding_vec = embedding_vec.detach().cpu().numpy()
                
                # 💡 修改点：mode 1 保持原来的剥离逻辑，mode 2 原封不动保留
                if use_single_embedding == 1:
                    # 5. 剥离并裁剪掉第 768 维的 log_cnt
                    if embedding_vec.shape[0] == 769:
                        e_vec = embedding_vec[:768]    # 只有 Base 向量，去掉最后一维 count
                    elif embedding_vec.shape[0] == 2305 or embedding_vec.shape[0] == (2305 - 768) or embedding_vec.shape[0] == (2305 + 768):
                        base_emb = embedding_vec[:768]
                        pca_emb = embedding_vec[769:]  # 越过第 768 维的 count，截取后 1536 维 PCA
                        e_vec = np.concatenate([base_emb, pca_emb], axis=0)
                    else:
                        e_vec = embedding_vec
                else:
                    e_vec = embedding_vec # 保留 log_cnt 留给网络去切分
                
                sample_vec.append(e_vec)
                
                # 6. 使用与原版完全一致的 np.hstack 进行横向打平拼接
                sample_vec = np.hstack(sample_vec)
                query_features.append(sample_vec)
                
            samples_enc.append(query_features)
        return samples_enc
    else:
        # 如果不启用 embedding，退回原版的纯传统编码
        return encode_samples(tables, samples, table2vec, use_single_embedding)


def load_and_encode_train_data(config):
    workload_dir = config.get("workloads_dir", "workloads")
    trainset = config["trainset"]
    column_min_max_path = config["column_min_max_file"]
    num_materialized_samples = config.get("num_samples", 1000)
    num_hash_buckets = config.get("num_buckets", 16)

    use_single_embedding = config.get("use_single_embedding", 0)
    train_emb_path = config.get("train_embedding_file", "")

    train_prefix = os.path.join(workload_dir, trainset)

    joins, predicates, tables, samples, label = load_data(train_prefix, num_materialized_samples, use_single_embedding, train_emb_path)

    column_names = get_all_column_names(predicates)
    column2vec, idx2column = get_set_encoding(column_names)

    table_names = get_all_table_names(tables)
    table2vec, idx2table = get_set_encoding(table_names)

    operators = get_all_operators(predicates)
    op2vec, idx2op = get_set_encoding(operators)

    join_set = get_all_joins(joins)
    join2vec, idx2join = get_set_encoding(join_set)

    with open(column_min_max_path, 'r') as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter=','))
        column_min_max_vals = {}
        for i, row in enumerate(data_raw):
            if i == 0:
                continue
            column_min_max_vals[row[0]] = [float(row[1]), float(row[2])]

    # 💡 [修改] 调用全新的混合特征编码层
    samples_enc = hybrid_encode_samples(tables, samples, table2vec, use_single_embedding)
    predicates_enc, joins_enc = encode_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec, num_hash_buckets=num_hash_buckets)
    label_norm, min_val, max_val = normalize_labels(label)

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
    workloads_dir = config["workloads_dir"]
    train_prefix = os.path.join(workloads_dir, config["trainset"])
    test_prefix = os.path.join(workloads_dir, config["testset"])
    num_samples = config.get("num_samples", 1000)
    use_single_embedding = config.get("use_single_embedding", 0)
    num_hash_buckets = config.get("num_buckets", 16)

    use_join_embedding = config.get("use_join_embedding", 0)
    train_join_emb_path = config.get("train_join_embedding_file", "")
    test_join_emb_path = config.get("test_join_embedding_file", "")

    if use_join_embedding in [1, 2, 3]:
        print(f"train_join_emb_path: {train_join_emb_path}")
        print(f"test_join_emb_path: {test_join_emb_path}")

    encoding_reference_testset = config.get("encoding_reference_testset")
    encoding_test_prefix = (
        os.path.join(workloads_dir, encoding_reference_testset)
        if encoding_reference_testset
        else None
    )
    dicts, max_lengths = get_global_info(
        train_prefix,
        test_prefix,
        num_samples,
        encoding_test_prefix=encoding_test_prefix,
    )
    table2vec, column2vec, op2vec, join2vec = dicts
    max_num_joins, max_num_predicates = max_lengths

    with open(config["column_min_max_file"], 'r') as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter=','))
        column_min_max_vals = {row[0]: [float(row[1]), float(row[2])] for i, row in enumerate(data_raw) if i > 0}

    t_joins, t_preds, t_tables, t_samples, t_label = load_data(
        train_prefix,
        num_samples,
        use_single_embedding,
        config.get("train_embedding_file"),
        config.get("train_bitmap_file"),
    )
    
    # 💡 [修改] 调用全新的混合特征编码层 (训练集)
    t_samples_enc = hybrid_encode_samples(t_tables, t_samples, table2vec, use_single_embedding)
    allow_unknown_features = config.get("allow_unknown_features", False)
    t_preds_enc, t_joins_enc = encode_data(
        t_preds,
        t_joins,
        column_min_max_vals,
        column2vec,
        op2vec,
        join2vec,
        num_hash_buckets,
        allow_unknown_features=allow_unknown_features,
    )
    label_norm, min_val, max_val = normalize_labels(t_label)


    if use_join_embedding in [1, 2, 3]:
        train_js_all = torch.load(train_join_emb_path)
        if use_join_embedding == 1:
            # 模式1：只保留前 100 维 (Bitmap)
            train_js_enc = np.array([train_js_all[i][:100].cpu().numpy() for i in range(len(t_label))], dtype=np.float32)
        elif use_join_embedding == 2:
            # 模式2：保留完整向量 (Bitmap + Embedding + PCA)
            train_js_enc = np.array([train_js_all[i].cpu().numpy() for i in range(len(t_label))], dtype=np.float32)
        elif use_join_embedding == 3:
            # 模式3：去掉PCA (Bitmap + Embedding)
            train_js_enc = np.array([train_js_all[i][:868].cpu().numpy() for i in range(len(t_label))], dtype=np.float32)
    else:
        train_js_enc = np.zeros((len(t_label), 1), dtype=np.float32)

    num_queries = len(label_norm)
    num_train = int(num_queries * 0.9)
    train_data = [t_samples_enc[:num_train], t_preds_enc[:num_train], t_joins_enc[:num_train]]
    val_data = [t_samples_enc[num_train:], t_preds_enc[num_train:], t_joins_enc[num_train:]]
    labels_train = label_norm[:num_train]
    labels_val = label_norm[num_train:]

    test_emb_path = config.get("test_embedding_file", "")
    test_joins, test_preds, test_tables, test_samples, test_label_raw = load_data(
        test_prefix,
        num_samples,
        use_single_embedding,
        test_emb_path,
        config.get("test_bitmap_file"),
    )
    
    # 💡 [修改] 调用全新的混合特征编码层 (测试集)
    test_samples_enc = hybrid_encode_samples(test_tables, test_samples, table2vec, use_single_embedding)
    test_preds_enc, test_joins_enc = encode_data(
        test_preds,
        test_joins,
        column_min_max_vals,
        column2vec,
        op2vec,
        join2vec,
        num_hash_buckets,
        allow_unknown_features=allow_unknown_features,
    )
    test_label_norm, _, _ = normalize_labels(test_label_raw, min_val, max_val)

    if use_join_embedding in [1, 2, 3]:
        test_js_all = torch.load(test_join_emb_path)
        if use_join_embedding == 1:
            test_js_enc = np.array([test_js_all[i][:100].cpu().numpy() for i in range(len(test_label_raw))], dtype=np.float32)
        elif use_join_embedding == 2:
            test_js_enc = np.array([test_js_all[i].cpu().numpy() for i in range(len(test_label_raw))], dtype=np.float32)
        elif use_join_embedding == 3:
            test_js_enc = np.array([test_js_all[i][:868].cpu().numpy() for i in range(len(test_label_raw))], dtype=np.float32)
    else:
        test_js_enc = np.zeros((len(test_label_raw), 1), dtype=np.float32)

    train_dataset = make_dataset(*train_data, join_samples=train_js_enc[:num_train], labels=labels_train, max_num_joins=max_num_joins, max_num_predicates=max_num_predicates)
    val_dataset = make_dataset(*val_data, join_samples=train_js_enc[num_train:], labels=labels_val, max_num_joins=max_num_joins, max_num_predicates=max_num_predicates)
    test_dataset = make_dataset(test_samples_enc, test_preds_enc, test_joins_enc, test_js_enc, test_label_norm, max_num_joins, max_num_predicates)

    return dicts, column_min_max_vals, min_val, max_val, train_dataset, val_dataset, test_dataset, (max_num_joins, max_num_predicates), test_label_raw


def make_dataset(samples, predicates, joins, join_samples, labels, max_num_joins, max_num_predicates):
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
