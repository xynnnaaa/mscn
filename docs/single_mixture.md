# Single mixture：当前最终实现

本文以 2026-10-08 的 `mscn/mixture.py`、`mscn/data.py`、`mscn/model.py`、`train.py`
及 `observatory/observatory/properties/Column_Order_Insignificance/final_single_online_mixture.py` 为准。

当前版本在每个查询的每张表上融合 QA/random 两份样本特征。**gate 只使用查询结构，
不输入覆盖统计、不根据命中状态强制设置权重；空表 embedding 保留，PCA 根据原始向量动态屏蔽。**
不改变查询、标签或采样预算。特征生成阶段聚合 tuple embedding 并计算 PCA，模型训练/推理阶段只读取文件。

## 1. 数据流与文件分工

```text
已有 SQL + 已注册样本表 + random/QA tuple embedding 缓存
  → 查询匹配 ID → 按 QA 缓存中的 ID 划分来源
  → 两份 bitmap + 两份 [mean, log_count, PCA] + 分析用 metadata
  → load_mixture_samples：读取四份特征，丢弃 log_count
  → 分来源共享编码 → 查询结构 gate → 加权融合 → 原 MSCN 主干
```

| 文件 | 作用 |
|---|---|
| `final_single_online_mixture.py` | 一次生成两来源 bitmap、embedding 和分析用 JSON |
| `mscn/mixture.py` | 配置校验、四文件加载、共享来源编码、gate 和诊断统计 |
| `mscn/data.py::load_and_encode_all_data` | 查询编码、选择 mixture 加载路径、划分训练/验证 |
| `mscn/model.py::SetConv` | 查询上下文提取、融合模块与 MSCN 主干连接 |
| `train.py` | Q-error 训练、验证选 best、日志与 checkpoint |
| `eval-merge.py` | 已将 mixture 配置传入 state_dict 模型重建入口 |
| `configs/single_mixture.example.json` | 训练配置示例，不是特征生成器配置 |

开启 mixture 时不经过 `hybrid_encode_samples()`。调用 `load_data(..., -1, ...)` 仅跳过旧样本加载，
随后由 `load_mixture_samples()` 加载新特征；日志中的 `Mode -1: Skipping ...` 不表示整个模型未使用样本。

## 2. 特征生成：来源与样本表

生成器只支持 `sampling.strategy="all_hit"`，要求 `has_unmatched_embedding=0`。

- 表在 `candidate_samples_info` 中：读取
  `<table><default_sample_suffix><mixture_sample_suffix>`。
- 表不在该字典中：读取 `<table><default_sample_suffix>`，全部作为 random。
- `mixture_sample_suffix` 默认 `_join_drift_train_2_qa`；value drift 配置使用 `_value_drift_qa`，
  与 `_s1000` 拼接得到 `<table>_s1000_value_drift_qa`。
- candidate 列表中的 idx 不参与表名拼接，沿用原选表逻辑。
- SQL 错误或样本表不存在会报错，不自动换表，也不从原表重新采样。

`paths.static_emb_path` 为已有 random tuple embedding 缓存；`paths.static_emb_path_2` 必须只包含
实际 QA tuple embedding。生成器在合并缓存之前保存 QA ID 集合，混合表中的 ID 在此集合内即归为 QA，
其他归为 random。不要把包含随机补齐 tuple 的完整混合集合作为 QA 来源字典。
默认样本表分支始终全部归 random，即使对应 ID 恰好出现在 QA 字典里。

缓存路径建议在配置中显式填写，避免依赖脚本中按数据库名拼接的默认路径。
缓存合并时 QA tuple embedding 覆盖同 ID 的已有值；`<table>_EMPTY` 优先保留 random 缓存中的值，
没有时取 QA 缓存中的值，两来源共用同一空表 embedding。

正常模式保留缺失 tuple embedding 的在线编码：只补编码已匹配的 tuple，不增加样本。
`--no-model` 只使用缓存，缺失 tuple 会报错；空来源缺少正确的 `_EMPTY` embedding 也会报错。

### 聚合与 PCA

每个查询、表别名、来源的文件向量为：

```text
[mean embedding(E), log1p(真实匹配 tuple 数)(1), PCA(P)]
```

mean 是命中 tuple embedding 的算术平均；零匹配时使用 `_EMPTY`，log_count=0。
PCA 对匹配 embedding 中心化后执行 `torch.svd_lowrank(q=K, niter=2)`，
`K=min(pca_components, 匹配数-1, E)`；按最大绝对值元素修复每个方向的符号，再按行展平，
不足的分量补零。`P=pca_components×E`。

匹配数不足两个、PCA 分量数为零、中心化结果全零、分解失败或分解结果不合法时，PCA 全零。
低秩 SVD 有随机性；符号修复不保证不同运行的 PCA 完全一致。

### Bitmap 与顺序

两份 bitmap 直接复用同一次 SELECT 得到并完成来源划分的 ID：

- 每个来源固定 B 位，桶位置为 `(int(id)-1) % B`。
- 每个查询先写 4 字节 little-endian 表数量，再写各表的 packed bits。
- `np.packbits` 使用高位优先，最后不足一字节补零。
- 两来源可有桶位重叠；置位数不等于真实 tuple 数。

使用 `--input-csv` 或 `paths.input_csv` 按 CSV 表顺序写入，并核对 SQL/CSV 表名、别名和查询数量。
不提供 CSV 时按 SQL 解析出的表顺序写入，调用方必须保证与训练 CSV 一致。
查询按有效 SELECT 记录从 0 连续编号；读取时支持行首 `/*...*/` 注释及 `||` 后缀。
谓词处理沿用原脚本：提取 WHERE 中仅涉及一个表别名的条件；本脚本面向现有平坦 workload，
不是任意嵌套 SQL/CTE/复杂 JOIN ON 谓词的通用执行器。
SQL/CSV 的实际谓词、行序和样本语义仍需调用方保证，别名检查不能验证这些内容。

### 命令与输出

在现有 embedding 环境运行：

```bash
python /data2/xuyining/observatory/observatory/properties/Column_Order_Insignificance/final_single_online_mixture.py \
  --config embedding_config_mixture.json \
  --sql-file workloads/train.sql --input-csv workloads/train.csv \
  --output-file features/train.pt --single-bitmap-dim 1000
```

生成器配置沿用原 embedding 配置的 `db`、`paths`、`model`、`sampling`，
其中 `model.pca_components` 决定 PCA 分量数；`--single-bitmap-dim` 覆盖配置的 `single_bitmap_dim`（默认 1000）。

默认生成：

```text
train.query_aware.pt
train.random.pt
train.query_aware.bitmaps
train.random.bitmaps
train.mixture.json
```

`paths.query_aware_embedding_file`、`random_embedding_file`、`query_aware_bitmap_file`、
`random_bitmap_file`、`single_mixture_metadata_file` 可分别覆盖输出名，显式路径优先于输出 stem。
直接运行生成器会覆盖同名输出；value drift 的批量入口另有重复输出保护。

JSON 保存真实池大小、匹配数和 PCA 有效标记，**仅供离线统计，不是训练/推理依赖**。
生成器仍会输出它，模型配置不需要 `train/test_single_mixture_metadata_file`；残留这些键也不会被读取。

分片参数为 `--total_shards N --shard_id i`：保留全局查询 ID，输出名增加 `_shard_i`。
合并时 embedding/JSON 按全局 ID 合并，bitmap 按对应 ID 交错恢复顺序，不能简单拼接各分片文件。

## 3. 模型输入与维度

每个 train/test split 只需要以下四个路径（将前缀替换为 `train` 或 `test`）：

```text
<split>_query_aware_bitmap_file
<split>_random_bitmap_file
<split>_query_aware_embedding_file
<split>_random_embedding_file
```

`.pt` 结构为 `{query_id: {alias: CPU_tensor}}`，查询键支持整数或字符串，必须恰好覆盖 CSV 行号；
表键必须对应该查询的别名。加载使用 `torch.load(..., weights_only=True)`。

文件内单来源向量长度是 `E+1+P`；加载时保留 `vector[:E]` 和 `vector[E+1:]`，
剥离 log_count，保留空表 mean。每个表的最终输入为：

```text
[table one-hot(T), QA bitmap(B)/mean(E)/PCA(P), random bitmap(B)/mean(E)/PCA(P)]
```

样本特征宽度（不含 one-hot）为 `2×(B+E+P)`，没有额外的 6 维 metadata。

| B | E | P | 每来源文件向量 | 样本输入宽度 | 每来源编码后宽度 |
|---:|---:|---:|---:|---:|---:|
| 1000 | 768 | 0 | 769 | 3536 | 896 |
| 1000 | 768 | 768 | 1537 | 5072 | 1664 |
| 1000 | 768 | 1536 | 2305 | 6608 | 1664 |

模型默认 B=1000、E=768、P=1536；生成器默认一个 PCA 分量，因此 E=768 时 P=768。
**必须按实际生成配置填写 `single_pca_dim`，不能直接假设示例的 1536。**
`num_samples` 是实验的实际样本预算，与 bitmap 桶数 B 不同；当前模型不从 metadata 验证预算。

加载器检查记录数、别名、固定维度、向量有限性、bitmap 表数量、截断/额外字节及尾部补零。
不再检查真实匹配数、池大小、bitmap/count 一致性或元数据 PCA 标记。

## 4. 编码、gate 和空匹配行为

两来源共享编码参数：

1. bitmap：Linear(B→128) → LeakyReLU。
2. mean：LayerNorm(E)。
3. PCA：LayerNorm(P)；若 P≠E，再 Linear(P→E) → LeakyReLU；P=0 时没有该分支。

PCA 有效性在归一化前按 `(raw_pca != 0).any(...)` 判断，采用**精确全零判定**。
归一化/投影后乘此 mask，避免 LayerNorm/Linear 的可学习偏置将无效输入变为非零。
该 mask 只控制 PCA 表示，不输入 gate、不控制 alpha。

所有查询、所有表共用一个 gate 网络，每个查询的每张表独立产生一个标量 alpha。
输入只有：表 one-hot(T)、全查询谓词池化表示(H)、全查询连接条件池化表示(H)，共 `T+2H` 维。
谓词/连接上下文通过现有两层 ReLU MLP 和 masked mean pooling 得到，随后也复用于最终预测。
gate 不读取 bitmap、mean/PCA 向量、真实计数、匹配率、命中标记或连接样本 embedding。

```text
alpha = Sigmoid(Linear(ReLU(Linear([table, predicate_context, join_context]))))
h = alpha × h_QA + (1-alpha) × h_random
```

gate 隐藏维度默认 64，最后一层权重和偏置零初始化，初始 alpha=0.5。
`single_mixture_gate_mode="fixed"` 时使用配置常量 `single_mixture_fixed_alpha`，默认 0.5。
bitmap、mean、PCA 共用同一个 alpha。

| 情况 | 当前处理 |
|---|---|
| 双命中 | 使用预测 alpha（fixed 模式用常量） |
| 仅一个来源命中 | 仍用同一规则；未命中分支的空表 embedding 参与融合 |
| 双未命中 | 不强制 alpha=0/1；两来源空表表示相同时融合结果与 alpha 无关 |
| padding | 编码后和融合后屏蔽，记录的 alpha 为零 |

融合后经 `single_emb_mlp1` → LeakyReLU → Dropout，与表 one-hot 拼接，进入原 MSCN 样本分支。
可继续使用已有连接样本分支；它不参与 gate 上下文。固定 0.5 融合也不等价于原合并样本的 mean/PCA 表示。

## 5. 训练、日志与推理

启用条件：`use_adaptive_single_mixture=true`、`use_single_embedding=1`、`has_unmatched_embedding=0`。
开关缺失或 false 时使用原路径。训练文件前 90% 为训练集、后 10% 为验证集，四份特征同步切分。

```bash
python train.py --config configs/single_mixture.example.json --seed 42
# 使用现有无固定 seed 流程时改为 --no-seed
```

损失仍为反归一化基数上的 Q-error，无门控辅助损失。best 按验证 loss 选择，测试集不参与选择。
当前 epoch Loss 是 batch 均值的等权平均；最终 Mean Q-error 按查询平均且预测取整，两者可能略有差异。

数据加载打印 mixture 启用状态、gate 模式、B/E/P、train/test 四个路径、查询数和表实例数。
`train.py::get_metrics()` 在验证和末尾测试打印：

```text
Single mixture: tables=2000, both=1426, mean alpha (both)=0.55,
only QA=74, only random=500, neither=0
```

这里都是有效查询表实例数量（不含 padding），四类之和等于 tables。命中只由 bitmap 是否非空推导，
仅用于诊断，不参与预测。mean alpha 只统计双命中实例，不能代表单来源实例的平均权重。
`model.single_mixture.last_alpha` 是最近批次的逐表权重，`last_stats` 是最近批次统计，均已 detach、不会存入 checkpoint。
自定义 eval 入口不会自动汇总这些统计，需显式读取；value drift 的独立评估器主要输出 Q-error。

best 保存原始 state_dict，并另写 `<model_output_path>.config.json`；可选 final checkpoint 也是 state_dict，
当前不会为 final 额外写配置文件。推理必须保留实际维度、gate 模式和训练时编码字典。
更换 test workload 时沿用原 `encoding_reference_testset`，同时更换全部四个 test 特征路径。

```python
from mscn.mixture import mixture_options
model = SetConv(..., single_mixture_options=mixture_options(config))
model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
```

`train.py` 末尾测试、`eval-merge.py` 的模型加载、value drift 的 `evaluate_single.py` 均已传入 mixture 配置。
旧 `eval.py` / `eval-new-model.py` 按历史 epoch 字典加载，不是本版 checkpoint 入口。
当前 query-only gate 比旧 coverage gate 少 10 个输入，样本输入也少 6 个元数据值，
**旧 learned-mixture checkpoint 不能严格加载，需重新训练**。
fixed 模式没有 gate 参数，部分历史 state_dict 可能形状兼容，但行为已变化，不能据此认为实验版本等价。
非 mixture 模型不因本次改动增加参数。

## 6. 核对与验证范围

本次核对确认生成器四份特征的格式、顺序和维度与当前加载器/模型一致；
修复了 `eval-merge.py` 连接特征模式 3 未与训练入口保持一致的问题。

可复现的仓库测试：

```bash
python -m unittest discover -s tests -v
# 使用相应环境，在 value drift 目录执行：
python test_mixture_pipeline.py
```

前者覆盖无 metadata 加载、log_count 剥离、gate 不依赖样本内容、动态 PCA/偏置/padding 屏蔽及梯度；
后者覆盖两库配置、生成调度/续跑保护、无 seed 训练入口及八组 eval 的模型重建与四路径切换。
另用临时模拟数据库测试检查了当前生成器的来源划分、空表处理、bitmap 与原脚本一致性、
CSV 表顺序、分片、PCA 和四文件加载对接。

本次共通过 4 项模型/加载测试、5 项 value drift 流程测试和 8 项生成器模拟测试。
另对无连接样本、100 维 bitmap、868 维 bitmap+embedding、1636 维 bitmap+embedding+PCA
四种连接输入组合完成前向、反向有限性及 checkpoint 保存/严格加载后的输出一致性检查。

这些检查不包含真实数据库重新生成特征或完整训练，不构成准确率提升的证明。
