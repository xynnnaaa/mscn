# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# # Define model architecture

# class SetConv(nn.Module):
#     # 修改参数：拆分 sample_feats 为 table_vec_size 和 sample_vec_size
#     def __init__(self, table_vec_size, sample_vec_size, predicate_feats, join_feats, join_sample_feats, hid_units, use_single_embedding=0, dropout_p=0.1, has_unmatched_embedding=0):
#         super(SetConv, self).__init__()

#         self.table_vec_size = table_vec_size
#         self.use_single_embedding = use_single_embedding
#         self.has_unmatched_embedding = has_unmatched_embedding

#         # 单表 Embedding，单独建MLP
#         if self.use_single_embedding == 1:
#             # self.single_emb_mlp1 = nn.Linear(sample_vec_size, hid_units)
#             # combined_sample_feats = table_vec_size + hid_units

#             # new for tuple-level embedding
#             # self.emb_norm = nn.LayerNorm(sample_vec_size)
#             # self.single_emb_mlp1 = nn.Linear(sample_vec_size, hid_units)
#             # self.single_emb_drop = nn.Dropout(p=dropout_p)
#             # combined_sample_feats = table_vec_size + hid_units

#             self.use_pca = (sample_vec_size != 769)
#             print(f"Using PCA for single embedding: {self.use_pca}")

#             if self.has_unmatched_embedding == 1:
#                 self.half_size = sample_vec_size // 2
#                 emb_dim = self.half_size - 1
#                 # 为命中和未命中分别建立 LayerNorm
#                 self.emb_norm_matched = nn.LayerNorm(emb_dim)
#                 self.emb_norm_unmatched = nn.LayerNorm(emb_dim)
#             else:
#                 if self.use_pca:
#                     pca_dim = sample_vec_size - 769
#                     emb_dim = 768
#                     self.pca_norm = nn.LayerNorm(pca_dim)
#                     self.pca_proj = nn.Linear(pca_dim, 768)
#                     mlp1_in_dim = 768 + 768 + 1
#                 else:
#                     emb_dim = sample_vec_size - 1 
#                     mlp1_in_dim = sample_vec_size
#                 self.emb_norm = nn.LayerNorm(emb_dim)

#             self.single_emb_mlp1 = nn.Linear(mlp1_in_dim, hid_units)
#             self.single_emb_drop = nn.Dropout(p=dropout_p)
#             combined_sample_feats = table_vec_size + hid_units

#         else:
#             # 原始 Bitmap，不需要单独 MLP，总维度不变
#             combined_sample_feats = table_vec_size + sample_vec_size

#         # 拼接后的整体再过原来的 Sample 分支
#         self.sample_mlp1 = nn.Linear(combined_sample_feats, hid_units)
#         self.sample_mlp2 = nn.Linear(hid_units, hid_units)

#         self.predicate_mlp1 = nn.Linear(predicate_feats, hid_units)
#         self.predicate_mlp2 = nn.Linear(hid_units, hid_units)
#         self.join_mlp1 = nn.Linear(join_feats, hid_units)
#         self.join_mlp2 = nn.Linear(hid_units, hid_units)

#         self.use_join_sample = join_sample_feats > 0

#         self.dropout_p = dropout_p

#         if self.use_join_sample:
#             print("Using join sample embedding model.")

#             # add for sum pooling
#             # self.join_emb_norm = nn.LayerNorm(join_sample_feats)

#             self.join_sample_mlp1 = nn.Linear(join_sample_feats, hid_units)
#             # self.join_sample_mlp2 = nn.Linear(hid_units, hid_units)
#             # 输入维度为 4 个头的拼接
#             self.out_mlp1 = nn.Linear(hid_units * 4, hid_units)
#             self.join_sample_dropout = nn.Dropout(p=dropout_p)
#         else:
#             # 输入维度回退为 3 个头的拼接
#             print("Using default model without join sample embedding.")
#             self.out_mlp1 = nn.Linear(hid_units * 3, hid_units)
#         self.out_mlp2 = nn.Linear(hid_units, 1)

#         # self.out_mlp1 = nn.Linear(hid_units * 3, hid_units)
#         # self.out_mlp2 = nn.Linear(hid_units, 1)

#     def forward(self, samples, predicates, joins, join_samples, sample_mask, predicate_mask, join_mask):
#         # samples has shape [batch_size x num_joins+1 x sample_feats]
#         # predicates has shape [batch_size x num_predicates x predicate_feats]
#         # joins has shape [batch_size x num_joins x join_feats]

#         t_vecs = samples[:, :, :self.table_vec_size]   # 表的 One-Hot 部分
#         s_vecs = samples[:, :, self.table_vec_size:]   # Bitmap 或 Embedding 部分

#         if self.use_single_embedding == 1:
#             # s_vecs = self.single_emb_mlp1(s_vecs)

#             # new for tuple-level embedding
#             # s_vecs = self.emb_norm(s_vecs)
#             # s_vecs = F.leaky_relu(self.single_emb_mlp1(s_vecs), negative_slope=0.01)
#             # s_vecs = self.single_emb_drop(s_vecs)

#             if self.has_unmatched_embedding == 1:
#                 s_matched = s_vecs[:, :, :self.half_size]
#                 s_unmatched = s_vecs[:, :, self.half_size:]

#                 s_m_emb, s_m_cnt = s_matched[:, :, :-1], s_matched[:, :, -1:]
#                 s_u_emb, s_u_cnt = s_unmatched[:, :, :-1], s_unmatched[:, :, -1:]

#                 s_m_emb = self.emb_norm_matched(s_m_emb)
#                 s_u_emb = self.emb_norm_unmatched(s_u_emb)

#                 s_vecs = torch.cat([s_m_emb, s_m_cnt, s_u_emb, s_u_cnt], dim=-1)

#             else:
#                 if self.use_pca:
#                     s_emb = s_vecs[:, :, :768]
#                     s_cnt = s_vecs[:, :, 768:769]
#                     s_pca = s_vecs[:, :, 769:]
#                     s_emb = self.emb_norm(s_emb)
#                     s_pca = self.pca_norm(s_pca)

#                     s_pca_feat = self.pca_proj(s_pca)

#                     s_vecs = torch.cat([s_emb, s_pca_feat, s_cnt], dim=-1)
#                 else:
#                     s_emb = s_vecs[:, :, :-1]  # 取前 768 维 (Embedding)
#                     s_cnt = s_vecs[:, :, -1:]  # 取最后 1 维 (log_count)
#                     s_emb = self.emb_norm(s_emb)
#                     s_vecs = torch.cat([s_emb, s_cnt], dim=-1)
                
#             s_vecs = F.leaky_relu(self.single_emb_mlp1(s_vecs), negative_slope=0.01)
#             # s_vecs = self.single_emb_mlp1(s_vecs)
#             s_vecs = self.single_emb_drop(s_vecs)

            
#         # 处理后的 s_vecs 与原始的 t_vecs 重新拼接
#         combined_samples = torch.cat((t_vecs, s_vecs), dim=2)

#         hid_sample = F.relu(self.sample_mlp1(combined_samples))
#         hid_sample = F.relu(self.sample_mlp2(hid_sample))
#         hid_sample = hid_sample * sample_mask  # Mask
#         hid_sample = torch.sum(hid_sample, dim=1, keepdim=False)
#         sample_norm = sample_mask.sum(1, keepdim=False)
#         hid_sample = hid_sample / sample_norm  # Calculate average only over non-masked parts

#         hid_predicate = F.relu(self.predicate_mlp1(predicates))
#         hid_predicate = F.relu(self.predicate_mlp2(hid_predicate))
#         hid_predicate = hid_predicate * predicate_mask
#         hid_predicate = torch.sum(hid_predicate, dim=1, keepdim=False)
#         predicate_norm = predicate_mask.sum(1, keepdim=False)
#         hid_predicate = hid_predicate / predicate_norm

#         hid_join = F.relu(self.join_mlp1(joins))
#         hid_join = F.relu(self.join_mlp2(hid_join))
#         hid_join = hid_join * join_mask
#         hid_join = torch.sum(hid_join, dim=1, keepdim=False)
#         join_norm = join_mask.sum(1, keepdim=False)
#         hid_join = hid_join / join_norm

#         # 动态拼接
#         if self.use_join_sample:
#             # add layer norm for sum pooling
#             # join_samples = self.join_emb_norm(join_samples)
            
#             hid_js = F.relu(self.join_sample_mlp1(join_samples))
#             # hid_js = F.relu(self.join_sample_mlp2(hid_js))

#             # hid_js = F.leaky_relu(self.join_sample_mlp1(join_samples), negative_slope=0.01)

#             # 添加dropout
#             hid_js = self.join_sample_dropout(hid_js)

#             hid = torch.cat((hid_sample, hid_predicate, hid_join, hid_js), 1)
#         else:
#             hid = torch.cat((hid_sample, hid_predicate, hid_join), 1)
        
#         hid = F.relu(self.out_mlp1(hid))
#         out = torch.sigmoid(self.out_mlp2(hid))
#         return out







# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # Define model architecture

# class SetConv(nn.Module):
#     def __init__(self, table_vec_size, sample_vec_size, predicate_feats, join_feats, join_sample_feats, hid_units, use_single_embedding=0, dropout_p=0.1, has_unmatched_embedding=0):
#         super(SetConv, self).__init__()

#         self.table_vec_size = table_vec_size
#         self.use_single_embedding = use_single_embedding
#         self.has_unmatched_embedding = has_unmatched_embedding

#         self.pca_dim = 0

#         # 混合特征模式 (Bitmap + Embedding)
#         if self.use_single_embedding == 1:

#             if sample_vec_size > 2304 + 500:
#                 self.use_pca = True
#                 self.bitmap_dim = sample_vec_size - 2304
#                 self.pca_dim = 1536
#             elif sample_vec_size > 1536 + 500:
#                 self.use_pca = True
#                 self.bitmap_dim = sample_vec_size - 1536
#                 self.pca_dim = 768
#             else:
#                 self.use_pca = False
#                 self.bitmap_dim = sample_vec_size - 768

#             print(f"[Hybrid Config] Detected Bitmap Dim: {self.bitmap_dim} | Enable PCA Features: {self.use_pca} | PCA Dim: {self.pca_dim}")

#             if self.has_unmatched_embedding == 1:
#                 self.half_size = (sample_vec_size - self.bitmap_dim) // 2
#                 emb_dim = self.half_size
#                 self.emb_norm_matched = nn.LayerNorm(emb_dim)
#                 self.emb_norm_unmatched = nn.LayerNorm(emb_dim)
#                 mlp1_in_dim = sample_vec_size
#             else:
#                 if self.use_pca:
#                     self.emb_norm = nn.LayerNorm(768)
#                     self.pca_norm = nn.LayerNorm(self.pca_dim)

#                     # 瓶颈映射层：将 1536 维的纯 PCA 压缩到 768，达到 1:1 特征平权
#                     if self.pca_dim > 768:
#                         # 瓶颈映射层：PCA 压缩到 768
#                         self.pca_proj = nn.Linear(self.pca_dim, 768)
                    
#                     mlp1_in_dim = self.bitmap_dim + 768 + 768  # Bitmap + Base + PCA_Proj
#                 else:
#                     self.emb_norm = nn.LayerNorm(768)
#                     mlp1_in_dim = self.bitmap_dim + 768        # Bitmap + Base

#             self.single_emb_mlp1 = nn.Linear(mlp1_in_dim, hid_units)
#             self.single_emb_drop = nn.Dropout(p=dropout_p)
#             combined_sample_feats = table_vec_size + hid_units

#         else:
#             # 原始纯 Bitmap 传统模式
#             combined_sample_feats = table_vec_size + sample_vec_size

#         # 后续传统的 MSCN 分支保持原样不动
#         self.sample_mlp1 = nn.Linear(combined_sample_feats, hid_units)
#         self.sample_mlp2 = nn.Linear(hid_units, hid_units)

#         self.predicate_mlp1 = nn.Linear(predicate_feats, hid_units)
#         self.predicate_mlp2 = nn.Linear(hid_units, hid_units)
#         self.join_mlp1 = nn.Linear(join_feats, hid_units)
#         self.join_mlp2 = nn.Linear(hid_units, hid_units)

#         self.use_join_sample = join_sample_feats > 0
#         self.dropout_p = dropout_p

#         if self.use_join_sample:
#             print("Using join sample embedding model.")
#             self.join_sample_mlp1 = nn.Linear(join_sample_feats, hid_units)
#             self.out_mlp1 = nn.Linear(hid_units * 4, hid_units)
#             self.join_sample_dropout = nn.Dropout(p=dropout_p)
#         else:
#             print("Using default model without join sample embedding.")
#             self.out_mlp1 = nn.Linear(hid_units * 3, hid_units)
#         self.out_mlp2 = nn.Linear(hid_units, 1)

#     def forward(self, samples, predicates, joins, join_samples, sample_mask, predicate_mask, join_mask):
#         t_vecs = samples[:, :, :self.table_vec_size]   # 表的 One-Hot 部分
#         s_vecs = samples[:, :, self.table_vec_size:]   # 特征数据段

#         if self.use_single_embedding == 1:
#             # 1. 拆解特征：切分出原始 Bitmap 信号和高维 Embedding 信号
#             bitmap_part = s_vecs[:, :, :self.bitmap_dim]
#             sub_s_vecs = s_vecs[:, :, self.bitmap_dim:]

#             if self.has_unmatched_embedding == 1:
#                 s_matched = sub_s_vecs[:, :, :self.half_size]
#                 s_unmatched = sub_s_vecs[:, :, self.half_size:]
#                 s_matched = self.emb_norm_matched(s_matched)
#                 s_unmatched = self.emb_norm_unmatched(s_unmatched)
#                 s_vecs = torch.cat([bitmap_part, s_matched, s_unmatched], dim=-1)
#             else:
#                 if self.use_pca:
#                     # 2. 剥离并裁剪掉 log_cnt (原 index 768 被完全跳过)
#                     s_emb = sub_s_vecs[:, :, :768]
#                     s_pca = sub_s_vecs[:, :, 768:] # 不包含 log_cnt 后的 1536 维
                    
#                     s_emb = self.emb_norm(s_emb)
#                     s_pca = self.pca_norm(s_pca)

#                     if self.pca_dim > 768:
#                         s_pca_feat = F.leaky_relu(self.pca_proj(s_pca), negative_slope=0.01)
#                     else:
#                         s_pca_feat = s_pca
                    
#                     # 3. 终极融合拼接：[Bitmap] + [768维Base] + [768维PCA投影]
#                     s_vecs = torch.cat([bitmap_part, s_emb, s_pca_feat], dim=-1)
#                 else:
#                     s_emb = sub_s_vecs[:, :, :768]  # 取前 768 维 (Embedding)
#                     s_emb = self.emb_norm(s_emb)
#                     s_vecs = torch.cat([bitmap_part, s_emb], dim=-1)
                
#             s_vecs = F.leaky_relu(self.single_emb_mlp1(s_vecs), negative_slope=0.01)
#             s_vecs = self.single_emb_drop(s_vecs)

#         # 重新组合进入 MSCN 主干神经网络
#         combined_samples = torch.cat((t_vecs, s_vecs), dim=2)

#         hid_sample = F.relu(self.sample_mlp1(combined_samples))
#         hid_sample = F.relu(self.sample_mlp2(hid_sample))
#         hid_sample = hid_sample * sample_mask  
#         hid_sample = torch.sum(hid_sample, dim=1, keepdim=False)
#         sample_norm = sample_mask.sum(1, keepdim=False)
#         hid_sample = hid_sample / sample_norm  

#         hid_predicate = F.relu(self.predicate_mlp1(predicates))
#         hid_predicate = F.relu(self.predicate_mlp2(hid_predicate))
#         hid_predicate = hid_predicate * predicate_mask
#         hid_predicate = torch.sum(hid_predicate, dim=1, keepdim=False)
#         predicate_norm = predicate_mask.sum(1, keepdim=False)
#         hid_predicate = hid_predicate / predicate_norm

#         hid_join = F.relu(self.join_mlp1(joins))
#         hid_join = F.relu(self.join_mlp2(hid_join))
#         hid_join = hid_join * join_mask
#         hid_join = torch.sum(hid_join, dim=1, keepdim=False)
#         join_norm = join_mask.sum(1, keepdim=False)
#         hid_join = hid_join / join_norm

#         if self.use_join_sample:
#             hid_js = F.relu(self.join_sample_mlp1(join_samples))
#             hid_js = self.join_sample_dropout(hid_js)
#             hid = torch.cat((hid_sample, hid_predicate, hid_join, hid_js), 1)
#         else:
#             hid = torch.cat((hid_sample, hid_predicate, hid_join), 1)
        
#         hid = F.relu(self.out_mlp1(hid))
#         out = torch.sigmoid(self.out_mlp2(hid))
#         return out



import torch
import torch.nn as nn
import torch.nn.functional as F

# Define model architecture

class SetConv(nn.Module):
    def __init__(self, table_vec_size, sample_vec_size, predicate_feats, join_feats, join_sample_feats, hid_units, use_single_embedding=0, dropout_p=0.1, has_unmatched_embedding=0, single_mixture_options=None, join_mixture_options=None):
        super(SetConv, self).__init__()

        self.table_vec_size = table_vec_size
        self.use_single_embedding = use_single_embedding
        self.has_unmatched_embedding = has_unmatched_embedding

        self.pca_dim = 0
        self.use_adaptive_single_mixture = single_mixture_options is not None
        self.use_adaptive_join_mixture = join_mixture_options is not None


        if self.use_adaptive_single_mixture:
            if use_single_embedding != 1 or has_unmatched_embedding != 0:
                raise ValueError("Single mixture requires mode 1 without unmatched embeddings")
            from mscn.mixture import SingleSampleMixture
            self.single_mixture = SingleSampleMixture(table_vec_size, hid_units, **single_mixture_options)
            if sample_vec_size != self.single_mixture.input_dim:
                raise ValueError("Single mixture feature width does not match configuration")
            self.single_emb_mlp1 = nn.Linear(self.single_mixture.output_dim, hid_units)
            self.single_emb_drop = nn.Dropout(p=dropout_p)
            combined_sample_feats = table_vec_size + hid_units
        elif self.use_single_embedding == 1:

            # ==========================================
            # Mode 1: 混合模式 (Bitmap + Embedding)
            # ==========================================

            if sample_vec_size > 2304 + 500 + 768:
                self.use_pca = True
                self.bitmap_dim = sample_vec_size - 2304 - 768
                self.pca_dim = 1536 + 768
            elif sample_vec_size > 2304 + 500:
                self.use_pca = True
                self.bitmap_dim = sample_vec_size - 2304
                self.pca_dim = 1536
            elif sample_vec_size > 1536 + 500:
                self.use_pca = True
                self.bitmap_dim = sample_vec_size - 1536
                self.pca_dim = 768
            else:
                self.use_pca = False
                self.bitmap_dim = sample_vec_size - 768



            # # 💡 恢复为原本对齐 3304 维文件的标准公式 (1000 + 768 + 1536)
            # bitmap_dim_with_pca = sample_vec_size - 2304  # 768 (Base) + 1536 (PCA) = 2304
            # bitmap_dim_no_pca = sample_vec_size - 768    # 768 (Base)
            
            # if bitmap_dim_with_pca <= 0:
            #     self.use_pca = False
            #     self.bitmap_dim = bitmap_dim_no_pca
            # else:
            #     if bitmap_dim_with_pca % 100 == 0:
            #         self.use_pca = True
            #         self.bitmap_dim = bitmap_dim_with_pca
            #     elif bitmap_dim_no_pca % 100 == 0:
            #         self.use_pca = False
            #         self.bitmap_dim = bitmap_dim_no_pca
            #     else:
            #         # 终极兜底逻辑：精准匹配 3304 对应的余数边界 (3304 % 500 == 304)
            #         if sample_vec_size % 500 == 304:
            #             self.use_pca = True
            #             self.bitmap_dim = bitmap_dim_with_pca
            #         else:
            #             self.use_pca = False
            #             self.bitmap_dim = bitmap_dim_no_pca

            print(f"[Hybrid Config] Bitmap Dim: {self.bitmap_dim} | Enable PCA Features: {self.use_pca} | PCA Dim: {self.pca_dim}")

            # 🛠️ 修改点 1：Bitmap 专门的降维映射层 (压缩至 512 维)
            self.encoded_bitmap_dim = 128
            self.bitmap_proj = nn.Linear(self.bitmap_dim, self.encoded_bitmap_dim)

            if self.has_unmatched_embedding == 1:
                self.half_size = (sample_vec_size - self.bitmap_dim) // 2
                emb_dim = self.half_size
                self.emb_norm_matched = nn.LayerNorm(emb_dim)
                self.emb_norm_unmatched = nn.LayerNorm(emb_dim)
                mlp1_in_dim = self.encoded_bitmap_dim + (sample_vec_size - self.bitmap_dim)
            else:
                if self.use_pca:
                    self.emb_norm = nn.LayerNorm(768)
                    self.pca_norm = nn.LayerNorm(self.pca_dim)

                    if self.pca_dim > 768:
                        # 瓶颈映射层：PCA 压缩到 768
                        self.pca_proj = nn.Linear(self.pca_dim, 768)

                    # 🛠️ 修改点 2：最终输入主网络维度 (128 + 768 + 768 = 1664)
                    mlp1_in_dim = self.encoded_bitmap_dim + 768 + 768
                else:
                    self.emb_norm = nn.LayerNorm(768)
                    # 🛠️ 修改点 3：无 PCA 时的维度 (128 + 768 维动态对数计数 = 896)
                    mlp1_in_dim = self.encoded_bitmap_dim + 768      

            self.single_emb_mlp1 = nn.Linear(mlp1_in_dim, hid_units)
            self.single_emb_drop = nn.Dropout(p=dropout_p)
            combined_sample_feats = table_vec_size + hid_units

        elif self.use_single_embedding == 2:
            # ==========================================
            # Mode 2: 纯 Embedding 模式 (无 Bitmap)
            # ==========================================

            self.pca_dim = sample_vec_size - 768 - 1
            self.use_pca = self.pca_dim > 0
            
            self.emb_norm = nn.LayerNorm(768)
            if self.use_pca:
                self.pca_norm = nn.LayerNorm(self.pca_dim)
                if self.pca_dim > 768:
                    self.pca_proj = nn.Linear(self.pca_dim, 768)
                mlp1_in_dim = 768 + 768
            else:
                mlp1_in_dim = 768

            print(f"[Pure Emb Config] Enable PCA: {self.use_pca} | PCA Dim: {self.pca_dim}")

            self.single_emb_mlp1 = nn.Linear(mlp1_in_dim, hid_units)
            self.single_emb_drop = nn.Dropout(p=dropout_p)
            combined_sample_feats = table_vec_size + hid_units

        # 新增：Mode -1 (纯 One-Hot，无底层特征)
        elif self.use_single_embedding == -1:
            print("[No Sample Config] Using ONLY Table One-Hot features.")
            combined_sample_feats = table_vec_size

        else:
            # 原始纯 Bitmap 传统模式
            combined_sample_feats = table_vec_size + sample_vec_size

        # 后续传统的 MSCN 分支保持原样不动
        self.sample_mlp1 = nn.Linear(combined_sample_feats, hid_units)
        self.sample_mlp2 = nn.Linear(hid_units, hid_units)

        self.predicate_mlp1 = nn.Linear(predicate_feats, hid_units)
        self.predicate_mlp2 = nn.Linear(hid_units, hid_units)
        self.join_mlp1 = nn.Linear(join_feats, hid_units)
        self.join_mlp2 = nn.Linear(hid_units, hid_units)

        # self.use_join_sample = join_sample_feats > 0
        # self.dropout_p = dropout_p

        # if self.use_join_sample:
        #     print("Using join sample embedding model.")
        #     self.join_sample_mlp1 = nn.Linear(join_sample_feats, hid_units)
        #     self.out_mlp1 = nn.Linear(hid_units * 4, hid_units)
        #     self.join_sample_dropout = nn.Dropout(p=dropout_p)
        # else:
        #     print("Using default model without join sample embedding.")
        #     self.out_mlp1 = nn.Linear(hid_units * 3, hid_units)
        # self.out_mlp2 = nn.Linear(hid_units, 1)

        self.join_sample_feats = join_sample_feats
        self.use_join_sample = join_sample_feats > 0
        self.dropout_p = dropout_p

        if self.use_adaptive_join_mixture:
            from mscn.mixture import JoinSampleMixture
            self.join_mixture = JoinSampleMixture(table_vec_size, hid_units, **join_mixture_options)
            if join_sample_feats != self.join_mixture.input_dim:
                raise ValueError("Join mixture feature width does not match configuration")
            self.join_sample_mlp1 = nn.Linear(self.join_mixture.output_dim, hid_units)
            self.join_sample_dropout = nn.Dropout(p=dropout_p)
            self.out_mlp1 = nn.Linear(hid_units * 4, hid_units)
            print("Using join mixture: query-only gate, unprojected bitmap, masked invalid PCA.")
        elif self.use_join_sample:
            print(f"Using join sample embedding model. Join feats: {join_sample_feats}")
            
            # --- 新增：针对 Join 特征动态创建 LayerNorm ---
            if self.join_sample_feats > 100:
                self.join_emb_norm = nn.LayerNorm(768)
                self.join_pca_dim = self.join_sample_feats - 100 - 768
                self.use_join_pca = self.join_pca_dim > 0
                if self.use_join_pca:
                    self.join_pca_norm = nn.LayerNorm(self.join_pca_dim)
                    print(f"[Join Config] Bitmap: 100 | Emb: 768 | PCA: {self.join_pca_dim}")
                else:
                    print(f"[Join Config] Bitmap: 100 | Emb: 768 | PCA: None")
            else:
                print(f"[Join Config] Bitmap: {self.join_sample_feats} | Emb: None")

            self.join_sample_mlp1 = nn.Linear(join_sample_feats, hid_units)
            self.out_mlp1 = nn.Linear(hid_units * 4, hid_units)
            self.join_sample_dropout = nn.Dropout(p=dropout_p)
        else:
            print("Using default model without join sample embedding.")
            self.out_mlp1 = nn.Linear(hid_units * 3, hid_units)
        self.out_mlp2 = nn.Linear(hid_units, 1)

    def forward(self, samples, predicates, joins, join_samples, sample_mask, predicate_mask, join_mask):
        t_vecs = samples[:, :, :self.table_vec_size]   # 表的 One-Hot 部分
        s_vecs = samples[:, :, self.table_vec_size:]   # 特征数据段

        if self.use_adaptive_single_mixture or self.use_adaptive_join_mixture:
            # These contexts depend only on query structure, never on the sample gate.
            hid_predicate = self._query_context(predicates, predicate_mask, self.predicate_mlp1, self.predicate_mlp2)
            hid_join = self._query_context(joins, join_mask, self.join_mlp1, self.join_mlp2)
        if self.use_adaptive_single_mixture:
            s_vecs = self.single_mixture(s_vecs, t_vecs, hid_predicate, hid_join, sample_mask)
            s_vecs = self.single_emb_drop(F.leaky_relu(self.single_emb_mlp1(s_vecs), negative_slope=0.01))
        elif self.use_single_embedding == 1:
            # 1. 拆解特征：切分出 1000 维的原始 Bitmap 和 2304 维的 Embedding 段
            bitmap_part = s_vecs[:, :, :self.bitmap_dim]
            sub_s_vecs = s_vecs[:, :, self.bitmap_dim:]

            # 2. 对 Bitmap 进行前向降维与非线性噪声过滤
            bitmap_feat = F.leaky_relu(self.bitmap_proj(bitmap_part), negative_slope=0.01)

            # bitmap_feat = bitmap_part

            if self.has_unmatched_embedding == 1:
                s_matched = sub_s_vecs[:, :, :self.half_size]
                s_unmatched = sub_s_vecs[:, :, self.half_size:]
                s_matched = self.emb_norm_matched(s_matched)
                s_unmatched = self.emb_norm_unmatched(s_unmatched)
                s_vecs = torch.cat([bitmap_feat, s_matched, s_unmatched], dim=-1)
            else:
                s_emb = sub_s_vecs[:, :, :768]
                s_emb = self.emb_norm(s_emb)

                if self.use_pca:
                    s_pca = sub_s_vecs[:, :, 768:]

                    s_pca_feat = self.pca_norm(s_pca)

                    if self.pca_dim > 768:
                        s_pca_feat = F.leaky_relu(self.pca_proj(s_pca_feat), negative_slope=0.01)

                    s_vecs = torch.cat([bitmap_feat, s_emb, s_pca_feat], dim=-1)
                else:
                    s_vecs = torch.cat([bitmap_feat, s_emb], dim=-1)
                
            s_vecs = F.leaky_relu(self.single_emb_mlp1(s_vecs), negative_slope=0.01)
            s_vecs = self.single_emb_drop(s_vecs)

        elif self.use_single_embedding == 2:
            # ==========================================
            # Mode 2 前向传播 (纯 Embedding，精准提取特征和对数计数)
            # ==========================================

            base_emb = self.emb_norm(s_vecs[:, :, :768])
            
            if self.use_pca:
                pca_emb = self.pca_norm(s_vecs[:, :, 769:])
                if self.pca_dim > 768:
                    pca_emb = F.leaky_relu(self.pca_proj(pca_emb), negative_slope=0.01)
                s_vecs_input = torch.cat([base_emb, pca_emb], dim=-1)
            else:
                s_vecs_input = torch.cat([base_emb], dim=-1)

            # 过单表聚合 MLP
            s_vecs = F.leaky_relu(self.single_emb_mlp1(s_vecs_input), negative_slope=0.01)
            s_vecs = self.single_emb_drop(s_vecs)

        if self.use_single_embedding == -1:
            combined_samples = t_vecs
        else:
            # 后续重新组合进入 MSCN 主干神经网络保持原样
            combined_samples = torch.cat((t_vecs, s_vecs), dim=2)

        hid_sample = F.relu(self.sample_mlp1(combined_samples))
        hid_sample = F.relu(self.sample_mlp2(hid_sample))
        hid_sample = hid_sample * sample_mask  
        hid_sample = torch.sum(hid_sample, dim=1, keepdim=False)
        sample_norm = sample_mask.sum(1, keepdim=False)
        hid_sample = hid_sample / sample_norm  

        if not (self.use_adaptive_single_mixture or self.use_adaptive_join_mixture):
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

        if self.use_join_sample:
            # --- 新增：切分 Join 特征并分别做 LayerNorm ---
            if self.use_adaptive_join_mixture:
                join_samples_input = self.join_mixture(
                    join_samples, t_vecs, hid_predicate, hid_join, sample_mask)
            elif self.join_sample_feats > 100:
                j_b = join_samples[:, :100]        # 前 100 维是 Bitmap
                j_e = join_samples[:, 100:868]     # 中间 768 维是 Embedding
                j_e = self.join_emb_norm(j_e)      # 独立归一化
                
                if self.use_join_pca:
                    j_pca = join_samples[:, 868:]  # 剩余的是 PCA

                    pca_sum = torch.sum(torch.abs(j_pca), dim=-1, keepdim=True)
                    valid_mask = (pca_sum > 1e-6).float() # 1 表示有真实 PCA，0 表示填零

                    j_pca = self.join_pca_norm(j_pca) # 独立归一化

                    j_pca = j_pca * valid_mask

                    join_samples_input = torch.cat([j_b, j_e, j_pca], dim=-1)
                else:
                    join_samples_input = torch.cat([j_b, j_e], dim=-1)
            else:
                # 只有 100 维 Bitmap
                join_samples_input = join_samples

            hid_js = F.leaky_relu(self.join_sample_mlp1(join_samples_input), negative_slope=0.01)
            hid_js = self.join_sample_dropout(hid_js)
            hid = torch.cat((hid_sample, hid_predicate, hid_join, hid_js), 1)
        else:
            hid = torch.cat((hid_sample, hid_predicate, hid_join), 1)
        
        hid = F.relu(self.out_mlp1(hid))
        out = torch.sigmoid(self.out_mlp2(hid))
        return out

    @staticmethod
    def _query_context(values, mask, first, second):
        hidden = F.relu(second(F.relu(first(values)))) * mask
        return hidden.sum(dim=1) / mask.sum(dim=1).clamp_min(1)
