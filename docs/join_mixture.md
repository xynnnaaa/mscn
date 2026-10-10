# Join sample mixture

本实现为每条查询融合 QA/random 两个 join sample 来源。仅新增模型、数据加载和训练/评估接口，不修改 join 数据准备脚本，也不改变当前单表 mixture 的编码、gate 或 PCA 处理。

## 模型设计

每条查询产生一个标量 `alpha_join`，所有查询共享 join gate；它与单表 gate 的参数独立。gate 输入为：

```
查询表 multi-hot + predicate context + join context
    → Linear → ReLU → Linear → sigmoid → alpha_join
```

表 multi-hot 从表 one-hot 经 sample_mask 排除 padding 后求和、截到 [0,1] 得到。predicate/join context 复用 MSCN 的查询分支（含编码后的谓词值），不使用样本 embedding、bitmap、覆盖率、命中数或 PCA 有效标记。单表与 join 任一启用 mixture 时先计算查询 context，避免依赖样本融合结果。

gate 隐藏维度默认64，最后一层权重和偏置零初始化，初始 alpha=0.5。也支持 fixed 模式，alpha 表示 QA 权重。

QA/random 共享 join 来源编码器，但不与单表来源编码器共享参数：

- Bitmap：原值保留，默认100维，无投影、降维或额外激活。
- Mean：独立 LayerNorm，保留空样本 embedding。
- PCA：独立 LayerNorm，**保留原 join 分支的无效 PCA 屏蔽**。对原始 PCA 计算 `abs().sum() > 1e-6`，在 LayerNorm 后乘 mask，避免偏置让无效 PCA 变成非零向量。PCA 不再额外投影，保持原维度。

```
h_mix = alpha_join * encode(QA) + (1-alpha_join) * encode(random)
h_join_sample = dropout(LeakyReLU(join_sample_mlp1(h_mix)))
```

之后仍与 `hid_sample / hid_predicate / hid_join` 拼接，交给原输出层。不会根据来源是否命中强制 alpha=0/1。双未命中时仍使用文件中的空 join mean；不把整个来源向量清零。模型不会从缺失文件推断空样本，缺失或不合法输入会报错。

当前 join 输入是一条查询一个向量，故没有 join sample 列表的 padding mask；查询表的 sample_mask 仅用于构造表 multi-hot。不要把 join 条件的 join_mask 当作 join sample 的命中标记。

## 文件格式

每个来源每条查询保存一个一维向量：

```
[bitmap(B), mean(E), PCA(P)]
```

默认 B=100、E=768、P=768，每份向量1636维；按 `[QA, random]` 拼接后3272维，融合后仍为1636维。**没有 log_count**，不要直接使用单表 `[mean, log_count, PCA]` 文件。

`.pt` 支持以连续 query ID 为键的字典（整数或字符串键），或按顺序排列的 tensor/list/tuple。ID 必须对应 workload CSV 行号，从0开始；每个来源必须包含全部查询。加载器检查数量、ID、维度和有限值。两个文件必须使用相同 CSV 查询顺序和相同特征定义；数值检查不能验证其 SQL 语义。

不读取 mixture metadata，也不单独读取 join bitmap 文件。空样本对应的 mean 由生成端提供。加载器先加载训练文件，再沿用原90%/10%训练/验证划分。

## 配置

在正常训练配置中增加以下字段（路径为占位示例）：

```json
{
  "use_join_embedding": 2,
  "use_adaptive_join_mixture": true,
  "join_mixture_gate_mode": "learned",
  "join_mixture_gate_hidden": 64,
  "join_mixture_fixed_alpha": 0.5,
  "join_bitmap_dim": 100,
  "join_embedding_dim": 768,
  "join_pca_dim": 768,
  "train_join_query_aware_embedding_file": "/path/train.query_aware.pt",
  "train_join_random_embedding_file": "/path/train.random.pt",
  "test_join_query_aware_embedding_file": "/path/test.query_aware.pt",
  "test_join_random_embedding_file": "/path/test.random.pt"
}
```

- 必须使用 `use_join_embedding=2`，其他模式与 join mixture 同时开启会报错，避免旧切片逻辑截断两来源向量。
- 若不用 PCA，设 `join_pca_dim=0`，对应文件也必须只包含 bitmap/mean。
- 启用时不使用旧 `train_join_embedding_file/test_join_embedding_file`。
- `use_adaptive_single_mixture` 与 join 开关独立，可仅开 join、仅开单表或同时启用。单表仍按已有配置设置。
- 关闭 join mixture 后继续使用旧 join 模式1/2/3，其 bitmap/PCA 处理与 checkpoint 键不变。

## 训练、评估与日志

训练入口 `train.py` 与评估入口 `eval-merge.py` 已传递 join mixture 参数；两端均通过 `load_and_encode_all_data` 加载四份来源文件。

```bash
python train.py --config /path/config_join_mixture.json \
  --lr 0.001 --batch 128 --epochs 500 \
  --model-output-path /path/new_join_mixture_best.pt

python eval-merge.py --eval_config /path/eval_join_mixture.json
```

`eval-merge.py` 沿用原有 eval 配置结构，将对应模型条目的 `train_config_path` 指向新增 join mixture 训练配置，将模型路径指向其 checkpoint。这里的特征融合不同于 eval-merge 原有的多模型预测回退；不要额外开启回退来代替 join mixture。

加载时打印开关、维度和四个路径。训练的验证/最终测试日志增加：

```
Join mixture: queries=..., both=..., mean alpha (both)=...,
              only QA=..., only random=..., neither=...
```

命中状态由各来源 bitmap 是否有非零位计算，只用于日志。统计单位为查询，而单表 mixture 日志单位为查询内的表。mean alpha 只统计双命中查询。

新 join mixture 需要重新训练，不兼容原单来源 join checkpoint。不要覆盖已有模型。其他旧实验脚本若自行实例化 SetConv，也需要显式传入 `join_mixture_options(config)`；本次接入的训练和评估入口为上述两个。

## 验证

`tests/test_join_mixture.py` 覆盖加载与无效输入拒绝、90/10划分、bitmap 原维度保留、PCA 屏蔽、query-only gate/padding、固定融合，以及单表/join 开关组合下的前向、反向和 state_dict 重建。

```bash
PYTHONPATH=. python tests/test_join_mixture.py
```

本次未生成真实 join mixture 文件或进行完整实验训练。已有单表模块保持原实现；它当前取消无效 PCA 屏蔽的行为不因 join mixture 改变。
