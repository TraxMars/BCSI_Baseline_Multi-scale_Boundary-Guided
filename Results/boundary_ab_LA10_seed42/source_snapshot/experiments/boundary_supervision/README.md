# 五尺度边界监督 A/B

本阶段只增加 Student 的五个独立 `Conv3d(C, 1, 1)` 边界头及其监督。
不包含 BGS、通道筛选、特征交换、原型或有/无标签交互。
`model/vnet.py` 完全不改；Teacher 仍是原始 VNet，最终验证、选优和推理均使用 Teacher。

## 实现与梯度

`utils/boundary_supervision.py` 提供独立边界头、标签、损失、阶段划分和局部梯度函数。
Trainer 在需要边界监督时显式调用原来的 `model.encoder()` 和 `model.decoder()`，
一次前向得到分割与五层特征，没有重复编码或特征 detach。

| 模式 | 边界损失的反向传播 |
| --- | --- |
| `none`（默认） | 原始 MT 路径；不创建边界头，不计算边界标签或额外指标，不增加结果文件 |
| `deep`（A） | `(L_MT + 0.1 * mean(L_i)).backward()`；`L_i` 更新 E1…Ei 与 Head_i |
| `local`（B） | 先对每个 `0.1/5 * L_i` 用 `autograd.grad` 仅取 Ei 与 Head_i 的梯度，保留图；再 `L_MT.backward()`，把暂存梯度加到 `.grad` |

两组每个 iteration 都只有一次 SGD step、一次原始 EMA 更新和一次调度器 step。
`allow_unused=False` 用于立即发现阶段映射错误；局部梯度计算使用
`retain_graph=True, create_graph=False`，不把边界梯度写入未指定参数。
两组使用同一初始化、标签、损失、五尺度平均权重和边界权重，主要区别只有梯度作用范围。
边界头在 Teacher deepcopy 之后独立创建，并用 CPU `fork_rng` 保护原始随机流，
不会因额外初始化改变 VNet、采样或 dropout 的随机状态。

| 阶段 | 模块 | 通道 |
| --- | --- | ---: |
| E1 | block_one | 16 |
| E2 | block_one_dw + block_two | 32 |
| E3 | block_two_dw + block_three | 64 |
| E4 | block_three_dw + block_four | 128 |
| E5 | block_four_dw + block_five | 256 |

分组通过参数 identity 检查编码器参数无重复、无遗漏，不包含 Decoder。
MT 的 CE+Dice 和无标签概率 MSE 仍正常更新整个 Student。
只有有标签样本进入边界头的直接监督；VNet 原有 BatchNorm 仍使用原来的混合 batch。

## 真实边界标签与损失

先将 `label > 0` 的前景掩码 nearest 缩放到每层尺寸，再做 `3×3×3`
形态学梯度 `dilation - erosion`。使用 replicate padding，避免把裁剪体积的边框
人为当成边界；全空/全前景标签的边界均为空。这项统一约定对 A/B 完全一致。

每层损失为 `0.5 * BCEWithLogits + 0.5 * SoftDice`。BCE 直接接 logits；
只有 Dice 分支用 sigmoid，按每个样本的三个空间轴求和后平均，epsilon=`1e-5`。
五层损失取算术平均，初始总权重 `0.1`，尚未验证为最优超参数。

`audit_targets.py` 用原始 LA 前 8 个标注病例、每例两个固定 seed 的原始增强/裁剪，
检查实际标签。输出见 `target_audit/audit.json`、`boundary_ratios.csv` 和两张切片图。
以下是 16 个裁剪的平均正样本比例；输出 batch 维是 labeled batch size=2，通道均为 1。

| 层 | LA 输出 `[N,1,D,H,W]` 空间尺寸 | LA 边界占比 | 96³ 空间尺寸 | 96³ 边界占比 |
| --- | --- | ---: | --- | ---: |
| 1 | 112×112×80 | 5.122% | 96×96×96 | 4.609% |
| 2 | 56×56×40 | 10.113% | 48×48×48 | 9.093% |
| 3 | 28×28×20 | 19.328% | 24×24×24 | 17.442% |
| 4 | 14×14×10 | 35.054% | 12×12×12 | 32.288% |
| 5 | 7×7×5 | 56.173% | 6×6×6 | 54.543% |

两种尺寸各 80 个尺度目标中，未见下采样后前景消失、边界为空或边界占比 ≥80%。
LA 最深层范围为 37.959%–77.551%，6³ 范围为 27.315%–70.370%。
深层边界相对较厚是固定三体素核的尺度效应，应结合训练全程日志继续观察；
本次未改变目标构造。合成 32³ 立方体测试中的 4³/2³ 目标可达到 100%，
测试会记录告警，不能把这种小尺寸测试误当作正式数据目标质量。
运行时每层检查目标为有限二值数，记录每个样本的占比、空边界、前景消失和高覆盖计数；
首次出现每类异常会告警，后续保留在 CSV 中。

## 对齐已完成的 Baseline

首轮对应 `Results/seed_42/result_LA_10l/fold_0/` 的已完成原始 MT，
以其 `log.txt` 第一行参数及 `code/trainer.py`、`code/vnet.py` 为依据。
修改前 Trainer 与该快照 SHA-256 一致，VNet 文件也一致。
原始 log 确认已完成 30000 次迭代；无需重新训练 Baseline。

- LA，10%（train.list 前 8/80 病例），seed=42，fold=0。
- patch=112×112×80，batch=4，labeled batch=2，30000 iterations。
- SGD lr=0.01、momentum=0.9、weight decay=1e-5。
- 原始 PolyWarmRestartScheduler，power=0.9，8000 次周期；不使用已完成的另一个固定 LR 实验。
- EMA decay=0.9，沿用 `min(1 - 1/(iteration+1), decay)`；不改变 BN buffer 行为。
- consistency=0.1，rampup=200，仍以 `iteration // 150` 调度。
- 每 200 次按原滑窗、最大连通域、Teacher 验证。LA/Pancreas 仍用原 test.list；
  BraTS 使用原 val.list，最终测试用原 test.list。
- 本机数据根目录从旧命令的 `/home/duanxuelong/Dataset/LA` 映射到
  `/aita_data/home/aita_duanxuelong/Dataset/LA`。列表与归档逐字节核对，病例顺序不变。

LA 训练期验证沿用 Baseline 的 test.list，因此不是独立验证集，解读最终对照时应保留这一限制。
原始历史日志没有记录完整包版本，不能据此保证与历史运行跨环境逐位一致；A/B 使用同一当前环境。

## 测试与运行

本机可用环境为 `/aita_data/home/aita_duanxuelong/.conda/envs/SSL29/bin/python`，
PyTorch `2.9.1+cu126`。默认系统 Python 没有 torch，不应直接用它启动。
GPU 工具与训练需在能够访问 CUDA 的宿主环境运行；沙箱中的 CUDA 查询可能返回不可用。

```bash
cd /aita_data/home/aita_duanxuelong/BCSI_Baseline_Multi-scale_Boundary-Guided
BOUNDARY_PYTHON=/aita_data/home/aita_duanxuelong/.conda/envs/SSL29/bin/python
"$BOUNDARY_PYTHON" experiments/boundary_supervision/test_boundary_supervision.py \
  --report experiments/boundary_supervision/verification.json
```

强制测试逐个隔离五层损失，检查完整的深监督下三角矩阵和局部监督对角矩阵，
同时覆盖独立形态学参照、nearest/极端输入、3D BCE+Dice、有限梯度、
同初始化/同前向、未标注特征在直接头损失中的零梯度、局部梯度与 MT 梯度相加、
一次 Student/Teacher 更新、最佳 Teacher 和边界状态保存、实际尺寸、
原指标返回兼容和已完成 Baseline checkpoint 严格加载。
`none` 与 `reference/trainer_before.py` 连续三步的模型参数/BN buffer、Teacher、
SGD momentum、scheduler、eval 输出逐张量相等。`reference/` 保留本次编辑前的真实源码。
测试中的 checkpoint 持久化单测使用人为指标，只验证保存路径，不能视为模型效果。
实际测试结果以 `verification.json` 和 `validation_report.json` 为准。

真实标签审计可复现命令（目标目录必须不存在）：

```bash
"$BOUNDARY_PYTHON" experiments/boundary_supervision/audit_targets.py \
  --data_path /aita_data/home/aita_duanxuelong/Dataset/LA \
  --output /tmp/boundary_target_audit_new
```

正式 A/B 推荐入口会从已完成 Baseline 日志恢复参数、核对源码和列表，
先重新运行强制测试，再从相同源码快照独立启动两组。默认不加 `--execute` 只打印命令；
运行会拒绝覆盖已存在的结果目录。GPU 编号需结合实际空闲资源选择。

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$BOUNDARY_PYTHON" \
  experiments/boundary_supervision/run_ab.py \
  --dataset LA --labeled_num 10 \
  --data_path /aita_data/home/aita_duanxuelong/Dataset/LA \
  --gpus 2 3 --execute
```

同等完整的直接启动命令如下；必须先通过上述测试。两组从随机初始化开始，互不加载权重。

```bash
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$BOUNDARY_PYTHON" -u train.py \
  --dataset LA --data_path /aita_data/home/aita_duanxuelong/Dataset/LA \
  --labeled_num 10 --seed 42 --n_fold 1 --patch_size 112 112 80 \
  --batch_size 4 --labeled_bs 2 --max_iterations 30000 --base_lr 0.01 \
  --test_interval 200 --consistency 0.1 --consistency_rampup 200 --ema_decay 0.9 \
  --device cuda:0 --guidance_mode none --boundary_mode deep --boundary_weight 0.1 \
  --output_dir ./Results

CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$BOUNDARY_PYTHON" -u train.py \
  --dataset LA --data_path /aita_data/home/aita_duanxuelong/Dataset/LA \
  --labeled_num 10 --seed 42 --n_fold 1 --patch_size 112 112 80 \
  --batch_size 4 --labeled_bs 2 --max_iterations 30000 --base_lr 0.01 \
  --test_interval 200 --consistency 0.1 --consistency_rampup 200 --ema_decay 0.9 \
  --device cuda:0 --guidance_mode none --boundary_mode local --boundary_weight 0.1 \
  --output_dir ./Results
```

启动器在 `Results/boundary_ab_LA10_seed42/` 保留共同源码快照、源码哈希、测试报告、
两组 console 日志和进程状态，训练完成后自动测试各自最佳 Teacher。
不要在相同目录重复执行；如需新 seed 或重复实验，使用新的 output_dir。
直接 `train.py` 的辅助模型实验同样会拒绝覆盖旧 fold。
新增 `--guidance_mode` 只接受 `none`，不恢复旧 BGS；原有参数保留。

## 结果、指标与 checkpoint

正式结果：

```text
Results/Boundary_A/seed_42/result_LA_10l/fold_0/
Results/Boundary_B/seed_42/result_LA_10l/fold_0/
```

真实 GPU Smoke Test 使用相同 LA patch/batch/seed，只有 `max_iterations=2`、
`test_interval=2` 与正式训练不同，结果位于 `Results/boundary_supervision_smoke/Boundary_A|Boundary_B/...`。
Smoke Test 只检查数值与流程，不是有效性实验。当前执行状态见本目录 `validation_report.json`，
正式训练状态见结果中的 `run_status.json`；不得将启动或短程完成表述为正式训练完成。

每组保留：

- `training_metrics.csv`：iteration、supervised/consistency loss、consistency weight、
  五层 boundary loss、平均 boundary loss、total loss、当前步实际使用 LR、下一步 LR，
  以及每层输出尺寸、平均/每样本边界比例、空边界/消失/高覆盖计数。
- `validation_metrics.csv`：同样病例和后处理下的 Teacher Dice、HD95、ASD、IoU、Boundary F1。
  ASD 复用原 `calculate_metric_percase` 已有计算；原 Dice/HD95 计算不变。
  Boundary F1 用 6 邻域表面和 Euclidean 距离，容差为 1 voxel；H5 未提供 spacing，非毫米单位。
  这是表面 precision/recall 的调和均值，不冒充 Surface Dice。
- `Model_iter_N.pth`：原有选优条件下的最佳 Teacher，纯 VNet state_dict，
  不含边界头，仍可由原 `prediction.py --model_path` 严格加载。
- `BoundaryHeads_iter_N.pth`：对应最佳 Teacher 时刻的 Student 辅助头与配置。
- `boundary_training_state_latest.pth`：最近验证/结束时的 Student、Teacher、边界头、
  optimizer、scheduler、iteration、最佳 Dice、配置、Torch CPU RNG。
- `experiment_config.json`、`data_split_manifest.json`：完整参数、命令、源码哈希、
  标签构造约定、列表哈希、病例和标注顺序；`code/` 保留实际源码。
- `best_metrics.json`、`run_status.json`、`log.txt`。

原来的 `prediction.test_calculate_metric()` 默认仍返回 `(dice, hd95)`。
边界实验显式请求 `return_details=True` 才得到扩展指标。CLI 可选
`--metrics_json <新文件>` 输出扩展测试指标，默认调用不改变。

保存的训练状态支持分析/手工恢复，但本阶段未实现 DataLoader worker 和 sampler
中途逐位精确恢复。为避免误把模型权重恢复称为实验恢复，边界模式拒绝
`--resume_model_path/--start_iteration`；`none` 仍支持原来的模型权重恢复。
不要用 A 的 checkpoint 初始化 B。

## 下一阶段接口与修改清单

```python
features = trainer.model.encoder(images)
segmentation_logits = trainer.model.decoder(features)
result = trainer.boundary(features, labeled_labels)
boundary_losses = result.losses
gradient_i, = torch.autograd.grad(boundary_losses[i], result.features[i], retain_graph=True)
```

应在训练 backward 释放图之前调用；本阶段不持久缓存带图的特征以避免内存泄漏。
此接口不需要修改 VNet，不包含任何通道筛选实现。

修改的现有文件为 `trainer.py`、`train.py`、`prediction.py`、`utils/utils.py`。
新增 `utils/boundary_supervision.py` 及本目录说明、测试、标签审计、启动器、
编辑前参照、验证报告和标签可视化。`model/vnet.py`、`utils/losses.py`、
数据加载/增强/采样代码均保持不变。仓库原有未提交改动和旧结果保持原状。

尚未得出 A/B 性能优劣结论；必须等正式训练与相同评估完成后才能比较。
