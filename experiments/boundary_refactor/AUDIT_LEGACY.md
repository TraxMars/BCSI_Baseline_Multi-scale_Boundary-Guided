# 原始 MT 与旧五尺度边界 A/B 的独立代码审计

审计日期：2026-10-20。此报告在新实验实现前形成，读取原始
`BCSI_Baseline`、旧多尺度仓库真实源码、旧 A/B 冻结源码、配置和完整训练记录。
审计过程没有修改或重新训练旧实验。机器可复核数据及 18 个关键产物的 SHA256
保存在 [legacy_audit.json](legacy_audit.json)。外部 BFA/DuCiSC/CHAP 的逐文件参考审计
见 [BFA 源码报告](REFERENCES_BFA.md) 和 [DuCiSC/CHAP 源码报告](REFERENCES_DUCISC_CHAP.md)，
包含固定 commit、具体文件行、真实训练路径和本实验适配差异；本报告不以论文摘要推断原作者实现。

## 结论与分类

**没有发现旧 A/B 的边界梯度路由代码错误，也不能用 Dice 下降证明实现错误。**
重新执行旧自动化套件的 10 项测试全部通过，包括五个损失分别隔离的梯度矩阵、
局部梯度数值核对、三步 `none` 模式逐张量回归和 Student/Teacher 更新。
确认需要在新实验中改变的是监督目标与预测头的设计，而不是把旧局部梯度算法
标记为错误后替换。

| 类别 | 已确认事实 | 处理原则 |
| --- | --- | --- |
| 已确认的边界实现代码错误 | 当前审计与旧套件复跑没有检出 | 保留旧 `deep`/`local` 语义；不以结果反推 bug |
| 已确认的方法设计差异 | 五个线性 `1×1×1` 头；先缩小标签再逐层形态学边界；包含 E5 | 新增独立 A2/B2，不改旧实验定义 |
| 已确认的目标膨胀 | E5 平均正样本率 55.8672%，最高单样本 87.3469% | 用全分辨率共享 Laplacian 目标的四尺度方案做控制实验 |
| 已确认的评估协议限制 | LA 的选优验证和最终测试都读取同一 20 例 `test.list` | 明确披露，按任务要求保持原协议；不称独立测试泛化 |
| 已确认的其他基线差异 | `fixed_lr_0p01_seed42` 是恒定 LR，且有 model-only 恢复历史 | 不能与本次 warm-restart A/B 混为同一个训练条件 |
| 尚待验证的假设 | 粗尺度边界损害定位、主辅梯度冲突、新目标改善性能、local 跨 seed 更优 | 新诊断与正式对照才可检验；本次审计不下性能因果结论 |

## 1. 证据来源与冻结源码行号

为避免新分支修改造成行号漂移，下列“旧 trainer”“旧边界模块”等定位均指
`Results/boundary_ab_LA10_seed42/source_snapshot/` 中的冻结源码：

- [旧 trainer](../../Results/boundary_ab_LA10_seed42/source_snapshot/trainer.py#L21)
- [旧边界模块](../../Results/boundary_ab_LA10_seed42/source_snapshot/utils/boundary_supervision.py#L13)
- [旧 train](../../Results/boundary_ab_LA10_seed42/source_snapshot/train.py#L108)
- [旧 VNet](../../Results/boundary_ab_LA10_seed42/source_snapshot/model/vnet.py#L124)
- [旧 prediction](../../Results/boundary_ab_LA10_seed42/source_snapshot/prediction.py#L98)
- [旧测试](../../Results/boundary_ab_LA10_seed42/source_snapshot/experiments/boundary_supervision/test_boundary_supervision.py#L105)

逐一读取并核对了 `trainer.py`、`train.py`、`model/vnet.py`、
`utils/boundary_supervision.py`、`prediction.py`、`utils/losses.py`、
`utils/utils.py`、数据加载器、两流 sampler、transform、旧实验说明、审计、测试和配置。
原始 `BCSI_Baseline/trainer.py` 与旧 `reference/trainer_before.py` 字节相同；
原始和旧多尺度仓库的 VNet、Dice、transform、dataset、sampler 字节相同。
本审计时 A/B 两份实际训练快照的 10 个源码哈希全部匹配配置中的记录，且 A/B 相同。

## 2. 原始 MT 主路径、EMA 与推理

旧 trainer 第 97–116 行仍为：有标签最终分割
`0.5*(CE + 原 DiceLoss)`，无标签 Student/Teacher softmax 概率 MSE，
一致性权重 `consistency * sigmoid_rampup(iteration//150, 200)`。
只有前 `labeled_bs=2` 个样本直接参与 GT 损失，后两个参与一致性；
sampler 第 42–50 行始终把 labeled indices 放在 batch 前部。

旧 trainer 第 78–82 行保持 SGD `lr=0.01, momentum=0.9, weight_decay=1e-5`，
`PolyWarmRestartScheduler(power=0.9, warm_restart_iters=8000)`；第 140–143 行
每次只调用一次 optimizer、一次 EMA、一次 scheduler。Teacher 在边界头创建前
deepcopy，优化器中只有 Student 和独立边界头，没有 Teacher 参数。

旧 trainer 第 42–47 行的 EMA 与原始实现相同：
`alpha=min(1-1/(global_step+1), ema_decay)`，只遍历模型参数做 EMA。
**BN running buffers 不由此公式更新**；原始 Teacher 在训练态进行无梯度前向，
因此其 BN buffers 通过自己的前向更新，dropout 也保持激活。
验证结束第 232–233 行恢复两个 VNet 的训练态。该行为来自原始 MT，
不属于新加边界模块的错误，不能在 A2/B2/C 中偷偷改成 eval Teacher 或 buffer EMA。
这里的“一次 EMA”准确含义是一次上述参数 EMA 更新，不表示 BN buffers 静止。

旧 trainer 第 183–199、221–228 行把最佳 Teacher 保存为原始 VNet 的纯
`state_dict`，边界头及训练状态另存；原始 `VNet.forward` 不依赖边界头。
本次复跑测试的严格 checkpoint 加载以及 `none` 三步 Student、Teacher、
SGD momentum、scheduler、eval 输出逐张量比较通过。没有额外增强或旧 BGS 路径。
VNet 编解码拓扑没有被旧边界实现修改。

## 3. A/B 梯度是否符合定义

阶段分组来自旧边界模块第 13–32 行，包含所有编码器参数各一次：

| 阶段 | 模块 |
| --- | --- |
| E1 | block_one |
| E2 | block_one_dw + block_two |
| E3 | block_two_dw + block_three |
| E4 | block_three_dw + block_four |
| E5 | block_four_dw + block_five |

普通深度模式由旧 trainer 第 126–127 行一次总损失 backward 实现。
局部模式由旧边界模块第 113–138 行，按每个 `0.1/5 * L_i` 只请求
`Ei.parameters + Head_i.parameters` 的 `autograd.grad`；保留共享图，
`allow_unused=False`，之后主 MT backward，再累加指定参数梯度。
`autograd.grad` 不填充未请求参数的 `.grad`，没有 detach 输入或重复 encoder 前向。

本次在原仓库重新执行 10 项旧测试，**10/10 通过，8.378 秒**。命令：

```bash
/aita_data/home/aita_duanxuelong/.conda/envs/SSL29/bin/python -B \
  experiments/boundary_supervision/test_boundary_supervision.py --threads 4
```

每一行只激活该行的一个边界损失；列为 E1…E5，1 表示该损失贡献梯度：

```text
deep                   local
1 0 0 0 0              1 0 0 0 0
1 1 0 0 0              0 1 0 0 0
1 1 1 0 0              0 0 1 0 0
1 1 1 1 0              0 0 0 1 0
1 1 1 1 1              0 0 0 0 1
```

每次隔离检查还确认只有对应 Head_i 有梯度，Decoder 无边界梯度，
局部路由数值等于独立 autograd 导数，特征梯度接口可用，损失和梯度有限。
这是梯度贡献范围的证据，不是仅凭总梯度非零判断。
局部模式不会阻断 MT 主损失更新整个 Student。
VNet 原有混合 batch BN 会使 labeled 特征对混合 batch 的统计产生依赖；
这不意味着对未标注 GT 做监督，也不改变局部参数路由定义。

## 4. 真实目标构造与完整训练统计

旧边界模块第 35–51 行先把 `(label>0)` nearest 缩到当前特征分辨率，
再用 replicate padding 的 `3×3×3` dilation 减 erosion。
得到内外两侧的形态学边带，各尺度的一个体素代表不同原图空间范围。
因此深层的相对边带越来越厚。nearest 没有制造无效类别，但不保证几何细节保持。
replicate padding 避免在全前景裁剪外沿人为加边界。

第 79–110 行的五个独立头均为 `Conv3d(C,1,1)`，直接输入 labeled features。
第 54–63 行的 BCE 直接接收 logits，Dice 单独 sigmoid，空间维度为 2、3、4；
第 74–76 行五尺度等权平均。没有 double sigmoid 或明显的 3D 维度错误。
这些正是旧版设计，不能称为已经实现 BFA 原版头或共享原图 Laplacian 目标。

以下从每组 **30000 行** `training_metrics.csv` 重新统计，迭代编号 0–29999，
每组共 60000 个 labeled patch 观测。A/B 的全部形状、前景、边界比例、
逐样本比例、消失/空/高覆盖字段在 30000 行上逐一完全相同。

| 层 | 输出形状 | 平均边界比例 | 平均前景比例 | 单样本边界最小–最大 | ≥80% 样本数 |
| --- | --- | ---: | ---: | ---: | ---: |
| E1 | 2×1×112×112×80 | 5.0116% | 15.1269% | 2.4060%–6.8793% | 0 |
| E2 | 2×1×56×56×40 | 9.9077% | 15.1222% | 4.7999%–13.6695% | 0 |
| E3 | 2×1×28×28×20 | 19.0467% | 15.1090% | 9.0179%–26.5434% | 0 |
| E4 | 2×1×14×14×10 | 34.9200% | 15.0688% | 15.4592%–50.4592% | 0 |
| E5 | 2×1×7×7×5 | 55.8672% | 14.7063% | 19.5918%–87.3469% | 1355 |

所有尺度空边界样本数和前景消失样本数均为 0。
E5 的 1355 个高覆盖样本占 60000 个观测的 **2.2583%**，分布在 1355 个 iteration。
E5 边界覆盖高是确认的几何现象；它是否导致性能损害尚不能由覆盖率单独证明。
原来的 16 个固定裁剪审计并未完整呈现正式随机裁剪的尾部风险；本报告使用全量记录。

新 A2/B2 拟使用四层 nonlinear head、一次原图 Laplacian GT、预测上采样，
因此与旧版同时改变头、目标和层数。新 A2 对旧 A 的变化不是单一因素消融；
只有新 A2 与新 B2 之间应只改变梯度范围。不能把新旧差异全解释为某一个修改。

## 5. 公平性与原始 Baseline 的具体参照

旧 A/B `experiment_config.json` 的 args 差异只有 `boundary_mode`；源码哈希相同。
种子 42、同一 8 labeled/72 unlabeled 病例、batch 4/2、30000 步、LA patch
112×112×80、EMA 0.9、初始 LR 0.01、8000 步重启和每 200 步验证一致。
初始化/随机流保护由 `fork_rng(devices=[])` 及本次同初始化 forward 测试确认。
A/B 使用不同物理 GPU，目标字段相同并不等同跨设备所有浮点训练轨迹逐位一致。

明确匹配的原始参考是
`Results/seed_42/result_LA_10l/fold_0/`：冻结 trainer 使用同一 warm restart；
150 次验证中日志最高 Dice **0.873561**（27000 步，日志精度 6 位），
对应日志 HD95 **8.888493**。其 LA 列表哈希与 A/B manifest 一致。
不要将日志舍入值当作原始精度测试 JSON。

另一个已经完成的 `BCSI_Baseline/Results/fixed_lr_0p01_seed42/` 参考不是相同条件：
其 `source_snapshot/trainer.py` 第 55–58 行使用恒定 0.01 的 LambdaLR，
`metadata.json` 明确记录从 PolyWarmRestartScheduler 改为恒定 LR。
其 LA10 Dice **0.8778317908**，`run_status.json` 记录在 1400 步 checkpoint
进行过 model-only 恢复，没有恢复 optimizer momentum 和原 Teacher 状态。
这些历史结果全部保留；不能将其与 A/B 的差异完全归因于边界监督。

## 6. 验证、最终测试及指标解释

旧 prediction 第 98–121 行：LA 无论 `val=True` 或 `False` 都读
`/aita_data/home/aita_duanxuelong/Dataset/LA/test.list`，20 例，SHA256
`e55b18069e27d6eab3dc21b8d1dc05a99ef83110fedc789f81c461d7bb152199`。
训练列表 80 例，SHA256
`e88c3f0bae70edcca4bc16eca890673eee2bf3f7b74418249cb01b8cde2f924e`，
训练和测试 IDs 交集为空，LA 没有 `val.list`。
**训练集与测试集分开；模型选优验证与最终测试不独立。**
BraTS 的 val/test 路径则在第 108–116 行明确分开，不能把 LA 结论泛化到全部数据集。

旧 prediction 第 135–136、156–164 行使用原 sliding-window stride 18/4、
前景概率 >0.5 和最大连通域，Teacher Dice 选优没有改动。
`utils/utils.py` 第 10–24 行的 ASD 是 `medpy.metric.binary.asd(pred,truth)`，
不是对称 ASSD；距离没有传 voxel spacing，数值单位为 voxel。
Boundary F1 使用 6-connected erosion、1 voxel Euclidean tolerance。
这些定义在 A/B 完全相同，不能将 ASD 报作毫米或自动等同 Surface Dice。

| 旧实验 | 最佳步数 | Dice | HD95 | ASD | Boundary F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| deep A | 27600 | 0.8551337861 | 10.8649733027 | 2.8739108779 | 0.5030841559 |
| local B | 27600 | 0.8736102770 | 9.7653948833 | 2.5092545868 | 0.5209752902 |

两组最佳与最终测试 JSON 一致，这是同一模型、同一列表评估的预期结果，
不是额外独立测试证据。原始 checkpoint、日志、CSV、配置、指标已建立保护哈希。

## 7. 梯度冲突与未来接口

旧代码保持 `features` 和每层 loss 的图，可计算 `dL_i/dF_i`，未实现通道选择。
主分割/一致性和边界目标在同一编码器参数上相加，存在梯度方向不一致的可能。
旧日志没有 per-stage 梯度余弦和范数证据，因此目前只能写“可能冲突”，
不能报告已发现冲突，更不能将 A 的性能归因于冲突。
新诊断需比较实际路由后的加权辅助梯度与 MT 主梯度；
local 模式不能误用未路由总边界梯度来代表实际优化方向。
诊断不能改 BN 状态、随机流、`.grad`、optimizer 次数或 loss 权重。

旧训练 state 没有完整保存所有 CUDA/NumPy/Python RNG、sampler 及 DataLoader 状态，
旧 CLI 也显式禁止边界实验的 model-only resume。这是复现/恢复能力限制，
不是本次已完成连续 A/B 的梯度证据缺陷。正式 A2/B2/C 仍需另外运行；
本审计和合成测试没有验证新方法性能。
