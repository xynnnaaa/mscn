import torch
import torch.nn as nn
import torch.nn.functional as F


# Define model architecture

class SetConv(nn.Module):
    def __init__(self, sample_feats, predicate_feats, join_feats, join_sample_feats, hid_units, dropout_p=0.3):
        super(SetConv, self).__init__()
        self.sample_mlp1 = nn.Linear(sample_feats, hid_units)
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

        hid_sample = F.relu(self.sample_mlp1(samples))
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
