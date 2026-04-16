import torch
import torch.nn as nn
import torch.nn.functional as F


# Define model architecture

class SetConv(nn.Module):
    # 修改参数：拆分 sample_feats 为 table_vec_size 和 sample_vec_size
    def __init__(self, table_vec_size, sample_vec_size, predicate_feats, join_feats, join_sample_feats, hid_units, use_single_embedding=0):
        super(SetConv, self).__init__()

        self.table_vec_size = table_vec_size
        self.use_single_embedding = use_single_embedding

        # 单表 Embedding，单独建MLP
        if self.use_single_embedding == 1:
            self.single_emb_mlp1 = nn.Linear(sample_vec_size, hid_units)
            self.single_emb_mlp2 = nn.Linear(hid_units, hid_units)
            combined_sample_feats = table_vec_size + hid_units
        else:
            # 原始 Bitmap，不需要单独 MLP，总维度不变
            combined_sample_feats = table_vec_size + sample_vec_size

        # 拼接后的整体再过原来的 Sample 分支
        self.sample_mlp1 = nn.Linear(combined_sample_feats, hid_units)
        self.sample_mlp2 = nn.Linear(hid_units, hid_units)

        self.predicate_mlp1 = nn.Linear(predicate_feats, hid_units)
        self.predicate_mlp2 = nn.Linear(hid_units, hid_units)
        self.join_mlp1 = nn.Linear(join_feats, hid_units)
        self.join_mlp2 = nn.Linear(hid_units, hid_units)

        self.use_join_sample = join_sample_feats > 0

        # self.join_sample_dropout = nn.Dropout(p=dropout_p)

        if self.use_join_sample:
            print("Using join sample embedding model.")
            self.join_sample_mlp1 = nn.Linear(join_sample_feats, hid_units)
            self.join_sample_mlp2 = nn.Linear(hid_units, hid_units)
            # 输入维度为 4 个头的拼接
            self.out_mlp1 = nn.Linear(hid_units * 4, hid_units)
        else:
            # 输入维度回退为 3 个头的拼接
            print("Using default model without join sample embedding.")
            self.out_mlp1 = nn.Linear(hid_units * 3, hid_units)
        self.out_mlp2 = nn.Linear(hid_units, 1)

        # self.out_mlp1 = nn.Linear(hid_units * 3, hid_units)
        # self.out_mlp2 = nn.Linear(hid_units, 1)

    def forward(self, samples, predicates, joins, join_samples, sample_mask, predicate_mask, join_mask):
        # samples has shape [batch_size x num_joins+1 x sample_feats]
        # predicates has shape [batch_size x num_predicates x predicate_feats]
        # joins has shape [batch_size x num_joins x join_feats]

        t_vecs = samples[:, :, :self.table_vec_size]   # 表的 One-Hot 部分
        s_vecs = samples[:, :, self.table_vec_size:]   # Bitmap 或 Embedding 部分

        if self.use_single_embedding == 1:
            s_vecs = F.relu(self.single_emb_mlp1(s_vecs))
            s_vecs = F.relu(self.single_emb_mlp2(s_vecs))
        # 处理后的 s_vecs 与原始的 t_vecs 重新拼接
        combined_samples = torch.cat((t_vecs, s_vecs), dim=2)

        hid_sample = F.relu(self.sample_mlp1(combined_samples))
        hid_sample = F.relu(self.sample_mlp2(hid_sample))
        hid_sample = hid_sample * sample_mask  # Mask
        hid_sample = torch.sum(hid_sample, dim=1, keepdim=False)
        sample_norm = sample_mask.sum(1, keepdim=False)
        hid_sample = hid_sample / sample_norm  # Calculate average only over non-masked parts

        hid_predicate = F.relu(self.predicate_mlp1(predicates))
        hid_predicate = F.relu(self.predicate_mlp2(hid_predicate))
        hid_predicate = hid_predicate * predicate_mask
        hid_predicate = torch.sum(hid_predicate, dim=1, keepdim=False)
        predicate_norm = predicate_mask.sum(1, keepdim=False)
        hid_predicate = hid_predicate / predicate_norm

        hid_join = F.relu(self.join_mlp1(joins))
        hid_join = F.relu(self.join_mlp2(hid_join))
        hid_join = hid_join * join_mask
        hid_join = torch.sum(hid_join, dim=1, keepdim=False)
        join_norm = join_mask.sum(1, keepdim=False)
        hid_join = hid_join / join_norm

        # 动态拼接
        if self.use_join_sample:
            hid_js = F.relu(self.join_sample_mlp1(join_samples))
            hid_js = F.relu(self.join_sample_mlp2(hid_js))

            # 添加dropout
            # hid_js = self.join_sample_dropout(hid_js)

            hid = torch.cat((hid_sample, hid_predicate, hid_join, hid_js), 1)
        else:
            hid = torch.cat((hid_sample, hid_predicate, hid_join), 1)
        
        hid = F.relu(self.out_mlp1(hid))
        out = torch.sigmoid(self.out_mlp2(hid))
        return out
