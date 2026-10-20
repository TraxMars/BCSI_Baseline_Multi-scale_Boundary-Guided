# BFA 编码器边界监督 A2/B2 与解码器分割深度监督 C

本目录实现独立的三组对照，保留原 MT、旧 `deep/local` 及已完成结果。开发分支为 `experiment/bfa-a2-b2-decoder-ds`，开发工作区为 `/aita_data/home/aita_duanxuelong/BCSI_Baseline_Multi-scale_Boundary-Guided_BFA_Refactor`；没有直接更新 `main`。

本轮工作完成实现、审计、自动化验证及有限真实数据 Smoke Test。**没有启动 A2/B2/C 的完整 30000 iteration 训练**；完整实验应在本轮检查后单独开展。Smoke Test 的初始化期指标不能用于判断方法优劣。

[修改文件与兼容性说明](CHANGES.md)；[三组完整正式命令（仅已预览）](formal_commands.txt)。

## 审计与方法依据

首先阅读 [实现前审计报告](AUDIT.md)、[旧实现详细审计](AUDIT_LEGACY.md) 及 [旧实验机器证据](legacy_audit.json)。结论按证据分类：

- **已确认代码错误**：没有发现旧 A/B 的规定梯度传播或 MT 主损失实现错误；旧测试重新运行通过。旧性能下降不是梯度实现失败的证据。
- **已确认设计差异**：旧五层 1³ 头使用各尺度 nearest GT 后的形态学梯度，原图等效边界厚度随层数增长。新 A2/B2 改为四个 BFA 头及一张原图分辨率 Laplacian 内侧边界目标；这是新实验设计，旧模式定义保持原样。
- **已确认评估限制**：LA 训练选优和最终测试均读取同一 `test.list`，没有独立验证/测试集。本任务保留原划分和选优策略并明确披露。
- **待验证假设**：粗尺度膨胀、E5 监督、头表达能力、梯度冲突是否解释性能变化，以及显式边界学习是否优于普通深度分割监督，需要正式实验；本轮不调整权重、划分或模型选优来追求更高 Dice。

源码参考均已实际读取并固定 commit、文件哈希及具体行号：

| 上游 | 参考报告 | 固定 commit |
|---|---|---|
| BFA | [源码位置与适配差异](REFERENCES_BFA.md)、[来源 JSON](references_bfa.json) | `74c2f2da2c2a399148dfcbb4750ea016b1941cbe` |
| DuCiSC | [DuCiSC/CHAP 源码说明](REFERENCES_DUCISC_CHAP.md)、[来源 JSON](references_ducisc_chap.json) | `cc77a8308d0aa1c59a7c29556478bfd33d804a1a` |
| CHAP | [DuCiSC/CHAP 源码说明](REFERENCES_DUCISC_CHAP.md) | `8754bfa5e5d269719b0ca45345c8a28742b13c0d` |

这些是针对当前 MT 的受控适配。没有迁移 BFA 的预训练、强扰动或无标签 LoG，没有迁移 DuCiSC 的原型、混合和多尺度伪标签，也没有实现 CHAP 通道筛选、特征交换或双向交互。

## 三组训练目标

| 实验 | 模式 | 辅助输出与监督 | 辅助梯度范围 |
|---|---|---|---|
| BFA_Deep_A2 | `boundary_mode=bfa_deep`，`auxiliary_mode=none` | x1–x4 四层显式边界 | Li 更新 E1…Ei 及 Head_i |
| BFA_Local_B2 | `boundary_mode=bfa_local`，`auxiliary_mode=none` | 与 A2 完全相同 | Li 仅贡献 Ei 及 Head_i 参数梯度 |
| Decoder_DS_C | `boundary_mode=none`，`auxiliary_mode=decoder_deep_seg` | 一条原始解码路径的 /8、/4、/2 侧输出及最终分割输出 | 普通解码器深度分割监督 |

编码阶段依实际 VNet 划分：E1=`block_one`；E2=`block_one_dw + block_two`；E3=`block_two_dw + block_three`；E4=`block_three_dw + block_four`；E5=`block_four_dw + block_five`。分组没有重复或遗漏。A2/B2 没有第五边界头，E5 继续通过原 MT 目标正常训练。

原主目标始终为 `L_MT = 0.5*(CE_final + Dice_final) + consistency_weight*MSE_unlabeled`。Student 与 Teacher 的原始分割输出、数据增强、采样、EMA、优化器与调度器保持原机制。

默认 `boundary_mode=none` 且 `auxiliary_mode=none` 时不构造辅助模块，继续原始 MT 前向和训练路径；连续更新及随机数状态的一致性由回归测试核对。

### A2/B2：四头、一个真实目标

每个独立头为 `Conv3d(C,16,3,padding=1,bias=False) → BatchNorm3d(16) → ReLU → Conv3d(16,1,1,bias=False)`，两层卷积采用 BFA 的 Kaiming normal 初始化、`a=1`。仅有标签特征子批进入头，防止新头 BN 引入额外的有/无标签混合依赖；原编码器 BN 的整批行为照常保留。上游 BFA 自训练是整批先过头再切分，此处是有意适配。

在原始 GT 的 `label>0` 前景上只提取一次边界：3³ 核中心 26，其余 −1，卷积后 clamp 到非负，再以 `>0.1` 二值化。kernel 为设备无关 buffer。默认 `boundary_padding_mode=zero` 保留上游参考；`replicate` 是显式可选的边缘诊断模式，不能单独为某一组静默切换。

四层 logits 按全部三个空间维检查，以 trilinear、`align_corners=True` 上采样至同一 GT 尺寸；`result.targets` 的四个元素引用同一个目标对象。

```text
Li = 0.5*BCEWithLogits(upsampled_logits_i, B_gt)
   + 0.5*SoftDice(sigmoid(upsampled_logits_i), B_gt)
L_boundary = mean(L1, L2, L3, L4)
L_total = L_MT + 0.1*L_boundary
```

Dice 按样本求空间和，使用 BFA 的 `smooth=1`；BCE 直接接收 logits，不重复 sigmoid。0.1 是初始实验超参数，未证明最优。BFA 原版是四层 BCE/Dice 直接求和且还加无标签 LoG；本方案的损失尺度与完整 BFA 不同。旧 `deep/local` 的五层形态学目标与 `smooth=1e-5` 均未更改。

A2 对总损失普通 backward。B2 使用 `autograd.grad((0.1/4)*Li, Ei参数 + Head_i参数)` 暂存局部梯度，保留共享计算图；随后原 `L_MT.backward()`，再向对应 `.grad` 累加。每步统一一次 SGD、一次原始 EMA、一次调度更新，不 detach 编码特征，不为各层分别 optimizer.step。

两个新边界实验共享模型、辅助头、初始化策略、目标、损失、权重、采样和评估；主要受控差异为边界参数梯度的作用范围。自动化检查隔离各 Li，得到 A2 的下三角范围和 B2 的对角范围，并验证相同初始化下前向一致、局部梯度与独立自动微分数值一致。

以 labeled batch=2、LA patch=112×112×80 为例：

| 层 | 头输入 | native logits | 上采样 logits / 共用目标 |
|---|---|---|---|
| x1 | `[2,16,112,112,80]` | `[2,1,112,112,80]` | `[2,1,112,112,80]` |
| x2 | `[2,32,56,56,40]` | `[2,1,56,56,40]` | 同上 |
| x3 | `[2,64,28,28,20]` | `[2,1,28,28,20]` | 同上 |
| x4 | `[2,128,14,14,10]` | `[2,1,14,14,10]` | 同上 |

### C：三侧头加原最终输出，最终损失只计一次

侧头读取 decoder skip-add 之后、`block_six/seven/eight` 之前的特征，符合所参考的 DuCiSC 输出位置；没有替换主解码器。三个 1³ 侧头分别输出 `/8、/4、/2` 的类别 logits，原最终输出提供第四个尺度。二类 LA 的输出为 `[2,2,14,14,10]`、`[2,2,28,28,20]`、`[2,2,56,56,40]`、`[2,2,112,112,80]`。

有标签 GT 用 nearest 缩放为各侧头的整数类别目标，每层使用原 MT 的 `0.5*(CE+Dice)`。最终输出损失已经在 MT 主损失中，辅助模块不重复计算：

```text
L_total_C = final_supervision_weight*L_supervised_final
          + consistency_weight*L_consistency
          + decoder_aux_weight*mean(L_side_1, L_side_2, L_side_3)
默认：final_supervision_weight=1.0，decoder_aux_weight=0.1
```

这一权重形式不同于上游四输出直接平均，目的是保留当前 MT 最终监督强度并明确记录新增侧监督。C 与编码器边界模式互斥；命令行和 Trainer 均拒绝默认叠加。Teacher 保持纯 VNet，不复制侧头。

## 真实 LA 标签可视化与统计

审计只读取原 `train.list` 的前 8 个有标签病例，每例 2 个固定原始增强裁剪，共 16 个 patch；不是验证/测试指标。固定示例为首例 `06SR5RBREL16DQ6M8LWS`、seed=42、112×112×80，旋转 k=2、flip axis=1，裁剪起点 `(14,10,7)`。全部种子、变换、标签哈希和列表哈希见 [标签 JSON](artifacts/label_audit_seed42/label_audit.json)，逐 patch 数值见 [CSV](artifacts/label_audit_seed42/label_audit_ratios.csv)，解释见 [标签报告](artifacts/label_audit_seed42/label_audit_report.md)。

| 目标 | 尺寸 | 16 patch 平均正样本比 |
|---|---|---:|
| 旧 E1 | 112×112×80 | 5.12196% |
| 旧 E2 | 56×56×40 | 10.11260% |
| 旧 E3 | 28×28×20 | 19.32836% |
| 旧 E4 | 14×14×10 | 35.05421% |
| 旧 E5 | 7×7×5 | 56.17347% |
| 新四层共享，zero | 112×112×80 | 2.59672% |
| 新四层共享，replicate | 112×112×80 | 2.46211% |

新边界是前景内侧单层目标，旧形态学梯度是双侧目标；比例差异同时反映边界定义与尺度构造差异，不能只归因于移除第五层。zero/replicate 在 16 个 patch 共差 21612 个体素，均位于裁剪最外一层。检查确认 replicate 目标为 zero 的子集、两者仅标前景；本组固定 patch 未出现前景消失、空边界或 ≥80% 覆盖。zero padding 将裁剪外视为背景，目标会包含部分裁剪接触面，应作为设计取舍披露。

可点击查看四张图：

- [固定 patch 原 GT](artifacts/label_audit_seed42/label_audit_fixed_patch_gt.png)
- [原图坐标下的新旧边界对比](artifacts/label_audit_seed42/label_audit_comparison.png)
- [旧五层 native 目标](artifacts/label_audit_seed42/label_audit_old_native.png)
- [六个裁剪面与 zero-only 黄色差异](artifacts/label_audit_seed42/label_audit_crop_faces.png)

图中旧目标的 nearest 放大只用于展示；实际新损失上采样预测，不缩小真实目标。

## 梯度诊断与后续研究接口

`utils/gradient_diagnostics.py` 仅观察梯度，不重跑前向、不消耗随机数、不写参数 `.grad`。实际优化所用 backward 和 B2 的局部梯度累加先完成，再对保留的图运行诊断，最后统一 optimizer.step。对 E1…E5 记录主梯度范数、加权边界梯度范数、比值、余弦、反向标志与状态。

A2 的 `boundary_norm/ratio/cosine` 与 `raw_boundary_norm/raw_ratio/raw_cosine` 来自对同一加权边界目标重新调用 autograd 所得的梯度，是其数学边界贡献的诊断估计。由于 CUDA backward 非确定，不能宣称该结果逐位等于实际 combined backward 中的边界分量。B2 的 applied 字段直接使用缓存并实际累加到 `.grad` 的局部梯度；其 raw 字段则另行计算完整标量 `0.1*mean(Li)` 的自然梯度。两类字段不可混淆，尤其 E1 的 raw 梯度包含高层损失，而 local applied 梯度只接自身损失。E5 没有边界贡献应明确记为 missing/零范数状态，而非虚构余弦。

当前“显著反方向”诊断阈值为 cosine≤−0.2 且范数比≥0.1，只是预先固定的记录规则，不是已验证因果标准。无梯度、零范数和 NaN/Inf 使用显式状态与 JSON null；不发出虚假有限统计值。

直接 `train.py` 默认 `gradient_diagnostic_interval=0`；正式实验 launcher 默认 200，Smoke 默认每步，C 关闭边界诊断。详细采样结果与局限见 [梯度诊断报告](GRADIENT_DIAGNOSTICS.md)。CPU 的诊断开启/关闭回归验证了完整状态逐位一致；GPU 的严格重复性存在下述未通过检查，不能把 CPU 结论扩大到 GPU。

### GPU 诊断隔离与运行间重复性的区别

真实 GPU 上另做的两步诊断启停对比超过指定的 `rtol=1e-5, atol=1e-7`，**一致性检查未通过**。失败结果原样保留于 [gpu_diagnostic_invariance.json](artifacts/gpu_diagnostic_invariance.json)，没有放宽容差后改标为通过。

必需的 CUDA trilinear 上采样 backward 在同一张图上的重复梯度计算也出现不一致；严格启用 `torch.use_deterministic_algorithms(True)` 时，该算子报未提供确定性实现。零参数更新的独立证据见 [gpu_trilinear_reproducibility.json](artifacts/gpu_trilinear_reproducibility.json)。这确认了算子层面的确定性限制，**不能证明全部整网误差都来自该算子**，也不能宣称两次 GPU 训练严格一致已通过。

为直接核对“观察动作是否修改已有状态”，另完成零参数更新的真实 GPU 隔离检查：在实际 backward 完成后，对比同一图上诊断调用前后的参数 `.grad`、模型状态和 RNG，三者均完全不变，该项检查通过，记录于 [gpu_diagnostic_observation.json](artifacts/gpu_diagnostic_observation.json)。这一检查考察一次诊断调用有无写状态；它与重新运行两条 GPU 训练轨迹能否严格重现是不同问题，不能互相替代，跨运行严格检查仍为失败。完整训练尚未进行，CUDA 数值非确定性仍作为复现限制披露。

`BFABoundaryResult.features` 保留原始五层张量，`losses` 保留四层计算图；`feature_boundary_gradient(result, i)` 返回 `dLi/dFi` 并保留图，可供后续研究读取。当前没有 Top25% 筛选或交互模块。

## 已完成验证与状态

当前三个自动化套件合计 **34 项通过**。此计数不包含上述单独的 GPU 诊断启停一致性检查，后者仍保留失败状态：

| 检查 | 结果来源 |
|---|---|
| 旧模式与 MT 回归 10 项 | [legacy_mode_regression.json](artifacts/legacy_mode_regression.json) |
| A2/B2 与 MT、标签、路由等 16 项 | [synthetic_boundary_tests.json](artifacts/synthetic_boundary_tests.json) |
| C 的尺度、目标、前后向、EMA与损失计数等 8 项 | [synthetic_decoder_tests.json](artifacts/synthetic_decoder_tests.json) |
| 旧结果与源码快照保护 | [legacy_protection_check.json](artifacts/legacy_protection_check.json) |

检查覆盖独立 Li 的下三角/对角传播、同初始化前向一致、B2 局部梯度数值一致、同一 GT 对象、BCE+Dice 极端值有限、Student 与 Teacher 原 EMA 更新、旧 checkpoint 严格加载及推理。合成测试里的 mocked accuracy 不应当作实际性能。

A2、B2、C 均已完成真实 GPU 的有限 Smoke：各 2 步训练和原列表 20 例验证，状态为 completed@2。完整证据与检查范围见 [完整验证报告](VALIDATION.md) 和 [汇总 JSON](artifacts/validation_summary.json)。最终 A2/B2 使用实际 backward 后诊断的实现，输出在 `Results/refactor_smoke_postdiag_20261020/{BFA_Deep_A2,BFA_Local_B2}/seed_42/result_LA_10l/fold_0/`；日志为 [A2 最终 Smoke](artifacts/bfa_deep_postdiag_smoke_console.log)、[B2 最终 Smoke](artifacts/bfa_local_postdiag_smoke_console.log)。两组源码和数据 manifest 相同，方法配置仅 boundary_mode 不同；运行设备分别为 `cuda:5`、`cuda:3`，均为 A100，因此不声称完整 args 只有 mode 差异。

C 的输出在 `Results/refactor_smoke_20261020/Decoder_DS_C/seed_42/result_LA_10l/fold_0/`，见 [C 日志](artifacts/decoder_smoke_console.log)。首轮 A2/B2 的结果和 [A2 首轮日志](artifacts/bfa_deep_smoke_console.log)、[B2 首轮日志](artifacts/bfa_local_smoke_console.log) 也保留在原 `Results/refactor_smoke_20261020/` 层级，不与最终 A2/B2 证据混用。有限 Smoke 通过不表示上述 GPU 运行间一致性检查通过。

完整训练尚未启动，尚无 A2/B2/C 的正式性能比较。旧已完成结果保持不变：A Dice=0.855134、HD95=10.864973、ASD=2.873911；B Dice=0.873610、HD95=9.765395、ASD=2.509255。

## 固定的正式实验设置

`run_experiment.py` 从已审计旧 A 的配置读取匹配的 LA10、seed42、warm-restart MT 设置；不会将另一个 fixed-LR 实验当成相同对照。

| 项目 | 配置 |
|---|---|
| 数据 | LA train.list=80 例，前 8 例有标签；原 test.list=20 例 |
| Patch / batch | 112×112×80；总 batch=4，labeled batch=2 |
| 初始化 / fold | seed=42；1 fold；三组各自从头训练 |
| 优化器 | SGD，LR=0.01，momentum=0.9，weight_decay=1e−5 |
| LR 调度 | 原 PolyWarmRestartScheduler，power=0.9，warm_restart_iters=8000 |
| MT | consistency=0.1，rampup=200，原 iter//150 调度；ema_decay=0.9，原 warm-start EMA 公式 |
| 长度 / 选优 | 30000 iterations；每 200 步验证；原最佳 Teacher Dice 选优 |
| A2/B2 | boundary_weight=0.1；zero padding；Dice smooth=1；诊断间隔=200 |
| C | final_supervision_weight=1；decoder_aux_weight=0.1；无边界项 |

Teacher 训练时的 train 模式、BN buffer 自身更新和参数 EMA 区别保持原实现，不在此次方法对照中修改。额外模块在保留原随机数状态的上下文中初始化，避免辅助头初始化改变 MT 主模型与后续随机流；A2/B2 用同一头构造策略。

验证与最终测试列表、指标函数和选优方式三组一致：Dice、HD95、ASD 与 1 voxel Boundary F1。距离单位遵循原体素坐标，不声称已换算为物理毫米。LA 两次评估同一 test.list，不能称为独立测试泛化；列表来源和哈希见 [审计](AUDIT_LEGACY.md) 与 [验证 JSON](artifacts/synthetic_boundary_tests.json)。

## 启动与复核

在仓库根目录使用 SSL29 环境。以下三个命令**仅打印**各自完整正式训练命令，不会启动训练：

```bash
cd /aita_data/home/aita_duanxuelong/BCSI_Baseline_Multi-scale_Boundary-Guided_BFA_Refactor

/aita_data/home/aita_duanxuelong/.conda/envs/SSL29/bin/python experiments/boundary_refactor/run_experiment.py --experiment BFA_Deep_A2 --stage formal --device cuda:0 --boundary_weight 0.1 --diagnostic_interval 200 --output_dir Results

/aita_data/home/aita_duanxuelong/.conda/envs/SSL29/bin/python experiments/boundary_refactor/run_experiment.py --experiment BFA_Local_B2 --stage formal --device cuda:0 --boundary_weight 0.1 --diagnostic_interval 200 --output_dir Results

/aita_data/home/aita_duanxuelong/.conda/envs/SSL29/bin/python experiments/boundary_refactor/run_experiment.py --experiment Decoder_DS_C --stage formal --device cuda:0 --decoder_aux_weight 0.1 --output_dir Results
```

后续独立开展正式实验时，在所选命令末尾显式追加 `--execute`。三组是独立初始化与独立目录，不使用 A2 的训练权重初始化 B2/C。可指定 `--data_path` 适配数据位置，数据列表内容仍须与本次审计一致。launcher 在执行前重跑两个新测试套件，失败即拒绝启动；已有目标目录即拒绝覆盖。`--stage formal` 固定 30000 步；默认 `--stage smoke` 为 2 步，Smoke 步数最多 10。

例如，重新运行有限 Smoke 时使用一个尚不存在的新输出根目录：

```bash
/aita_data/home/aita_duanxuelong/.conda/envs/SSL29/bin/python experiments/boundary_refactor/run_experiment.py --experiment BFA_Local_B2 --stage smoke --smoke_iterations 2 --device cuda:0 --output_dir /tmp/bcsi_bfa_local_smoke_review --execute
```

只重跑测试和标签审计：

```bash
/aita_data/home/aita_duanxuelong/.conda/envs/SSL29/bin/python experiments/boundary_supervision/test_boundary_supervision.py --threads 4 --report /tmp/bcsi_legacy_check.json
/aita_data/home/aita_duanxuelong/.conda/envs/SSL29/bin/python experiments/boundary_refactor/test_boundary_refactor.py --report /tmp/bcsi_boundary_refactor_check.json
/aita_data/home/aita_duanxuelong/.conda/envs/SSL29/bin/python experiments/boundary_refactor/test_decoder_supervision.py --report /tmp/bcsi_decoder_check.json
/aita_data/home/aita_duanxuelong/.conda/envs/SSL29/bin/python experiments/boundary_refactor/audit_labels.py --data_path /aita_data/home/aita_duanxuelong/Dataset/LA --output /tmp/bcsi_label_audit_review
```

标签审计的输出目录也必须尚不存在，以保留已有产物。

## 输出、推理与旧结果保护

正式默认目录分别是：

```text
Results/BFA_Deep_A2/seed_42/result_LA_10l/fold_0/
Results/BFA_Local_B2/seed_42/result_LA_10l/fold_0/
Results/Decoder_DS_C/seed_42/result_LA_10l/fold_0/
```

每组写独立 `experiment_config.json`、源码快照、`run_status.json`、训练 CSV、`validation_metrics.csv`、`best_metrics.json`；A2/B2 另写 `gradient_diagnostics.csv`。保留每层损失、目标比例及形状、总损失、学习率；C 分别记录三侧损失和最终监督权重。

最佳 Teacher 保存为原格式 `Model_iter_*.pth`，仍是纯 VNet state_dict；原 `prediction.py` 接口和旧 VNet checkpoint 严格加载保持兼容。边界头与解码器头训练状态独立保存为 `BoundaryHeads_iter_*.pth` / `DecoderHeads_iter_*.pth`；latest training state 另含 Student、Teacher、辅助模块、优化器、调度器与配置，用于复现和分析。当前未提供逐位精确的完整状态恢复入口，新辅助实验拒绝 model-only resume，不能把其称为精确续训。

旧 `Results/Boundary_A`、`Results/Boundary_B`、原 baseline checkpoint 与日志均保留。旧模式源文件和快照不重新解释成 A2/B2，辅助实验目录使用拒绝覆盖检查。本目录的新开发与 PR 不合并覆盖 main，也不自动推送新训练结果。

所有已执行验证与尚未完成项目，以 [VALIDATION.md](VALIDATION.md)、[validation_summary.json](artifacts/validation_summary.json) 和 [GRADIENT_DIAGNOSTICS.md](GRADIENT_DIAGNOSTICS.md) 的最终证据为准；正式性能、最优权重及研究假设仍待后续完整实验。
