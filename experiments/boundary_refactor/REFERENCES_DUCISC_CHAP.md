# DuCiSC / CHAP 真实源码参考审计

审计日期：2026-10-20（Asia/Shanghai）。本报告先阅读上游真实源码，再给出适配方案；不以旧 A/B 的性能下降推断实现错误。源码通过只读 shallow clone 获取，未将其他本地改写版本当作上游。精确 commit、文件 SHA256 与永久行链接见 `references_ducisc_chap.json`。

## DuCiSC：一条解码路径的四尺度分割监督

固定版本：`cc77a8308d0aa1c59a7c29556478bfd33d804a1a`。

[`code/networks/VNet.py:283–320`](https://github.com/ShanghaiTech-IMPACT/DuCiSC/blob/cc77a8308d0aa1c59a7c29556478bfd33d804a1a/code/networks/VNet.py#L283-L320) 中 `DuCiSC` 定义一条解码路径、一个最终 `out_conv` 和三个 `1×1×1` 分割侧头，输出通道均为 `n_classes`。它没有在五个编码阶段建立边界预测头。

[`decoder():344–384`](https://github.com/ShanghaiTech-IMPACT/DuCiSC/blob/cc77a8308d0aa1c59a7c29556478bfd33d804a1a/code/networks/VNet.py#L344-L384) 的真实接入位置如下。这里分辨率相对于原始输入，默认 `n_filters=16`：

| 输出顺序 | 上游输入特征与接入位置 | 输入通道 | 输出空间尺寸，LA 112×112×80 | 输出通道 |
|---|---|---:|---|---:|
| 0 | `x5_up = block_five_up(x5) + x4`，在 `block_six` **之前** | 128 | 14×14×10 | 2 |
| 1 | `x6_up = block_six_up(x6) + x3`，在 `block_seven` **之前** | 64 | 28×28×20 | 2 |
| 2 | `x7_up = block_seven_up(x7) + x2`，在 `block_eight` **之前** | 32 | 56×56×40 | 2 |
| 3 | `x9 = block_nine(block_eight_up(x8) + x1)`，可选原有 dropout 之后 | 16 | 112×112×80 | 2 |

[`forward():387–391`](https://github.com/ShanghaiTech-IMPACT/DuCiSC/blob/cc77a8308d0aa1c59a7c29556478bfd33d804a1a/code/networks/VNet.py#L387-L391) 在 `deepsup=True` 时返回四个输出；否则仅返回最终输出，同时另返回特征列表。不能将 DuCiSC 的默认 `forward` tuple 接口直接替换当前项目 VNet 的单张量推理接口。

[`train_DuCiSC.py:get_loss():117–153`](https://github.com/ShanghaiTech-IMPACT/DuCiSC/blob/cc77a8308d0aa1c59a7c29556478bfd33d804a1a/code/train_DuCiSC.py#L117-L153) 对真实标签使用 nearest 缩放，在各个输出上计算 CE 与**前景二值 Dice**，分别对四尺度平均；[`279–280`](https://github.com/ShanghaiTech-IMPACT/DuCiSC/blob/cc77a8308d0aa1c59a7c29556478bfd33d804a1a/code/train_DuCiSC.py#L275-L280) 再取 `0.5×(mean_CE + mean_Dice)`。因此最终输出的直接监督系数是四分之一，原版并没有先保留一个完整 final loss 再叠加四尺度平均。它的 [`Binary_dice_loss`](https://github.com/ShanghaiTech-IMPACT/DuCiSC/blob/cc77a8308d0aa1c59a7c29556478bfd33d804a1a/code/utils/losses.py#L4-L8) 使用线性概率和，当前 MT 的 `utils.losses.DiceLoss` 则对全部类别取平均且分母使用概率平方；这也是需要显式保留的适配差异。

上游完整训练还在 [`320–331`](https://github.com/ShanghaiTech-IMPACT/DuCiSC/blob/cc77a8308d0aa1c59a7c29556478bfd33d804a1a/code/train_DuCiSC.py#L320-L331) 将各类损失除以其 detached 数值，包含伪标签、原型和混合体积目标。本项目 C 只借鉴解码器多尺度分割监督，**不迁移**这些归一化、原型、混合体积和伪标签机制。上游训练验证还直接读取 `test.list`（[`test_patch.py:18–29`](https://github.com/ShanghaiTech-IMPACT/DuCiSC/blob/cc77a8308d0aa1c59a7c29556478bfd33d804a1a/code/utils/test_patch.py#L18-L29)）；这一点只是上游事实，不授权修改当前项目的数据划分。

## Decoder_DS_C 的明确适配选择

适配决策：把当前 VNet 原解码器一次前向中的 `x5_up = block_five_up(x5) + x4`、`x6_up = block_six_up(x6) + x3`、`x7_up = block_seven_up(x7) + x2` 以可选接口暴露出来，使用三个独立 `1×1×1` 侧头和已有最终分割输出，形成 `[side_1_8, side_1_4, side_1_2, final]`。侧头位置与上游精确一致，均在 skip-add 后、后续 `block_six/seven/eight` 之前；三个新头置于 Student 的独立辅助模块，EMA Teacher 继续只有原始 VNet 参数。保留当前 VNet 的 residual、dropout 和主解码器，不迁移完整 DuCiSC 训练方法。

令 `S(z,y) = 0.5×(CE(z,y) + 原MT Dice(z,y))`，只在 labeled batch 计算：

```text
L_MT = S(final, Y_l) + consistency_weight × L_consistency
L_decoder_aux = mean(S(side_1_8, nearest(Y_l)),
                     S(side_1_4, nearest(Y_l)),
                     S(side_1_2, nearest(Y_l)))
L_C = L_MT + decoder_aux_weight × L_decoder_aux
```

默认 `decoder_aux_weight=0.1`，这是预注册的初始对照权重，不宣称最优。最终输出的监督系数保持原 MT 的 1；四输出中的 final 只作记录和复用，不能再进入 `L_decoder_aux`。每个侧头系数明确为 `decoder_aux_weight / 3`。这与上游“四输出等权平均”不同，目的是隔离辅助深度监督且保持原 MT 主损失定义。A2/B2/C 使用同一原 MT 主损失、划分与模型选优流程，C 默认不能与编码器边界辅助监督叠加。新侧头初始化应在原 Student 与 Teacher 构造完成之后进行，以免改变原 VNet 初始化的随机数序列；两个辅助分支需保存独立状态。

默认推理 `VNet.forward` 仍返回已有 final tensor，原 checkpoint 键保持不变；额外接口仅在 C 的 Student 训练中启用。显式暴露中间张量优于通过 forward hooks 隐式修改返回值；启用或禁用该接口必须复现同一最终 logits 和 dropout 调用序列。

## CHAP：可借鉴的梯度分析，而非固定 Top 25% 筛选

固定版本：`8754bfa5e5d269719b0ca45345c8a28742b13c0d`。

[`utils/grad.py:176–219`](https://github.com/gardnerzhou/CHAP/blob/8754bfa5e5d269719b0ca45345c8a28742b13c0d/utils/grad.py#L176-L219) 的 `GradSim_3d.compute_grad_convkernel` 先独立 backward 监督损失，再清 `.grad` 并 backward 无监督损失，提取五个编码 block 最后一个卷积核的权重梯度，展平为“输出通道 × 该通道卷积核参数”。这不是 `dL/dfeature`，也没有在这里对边界损失做分析。

[`168–174`](https://github.com/gardnerzhou/CHAP/blob/8754bfa5e5d269719b0ca45345c8a28742b13c0d/utils/grad.py#L168-L174) 计算各通道的梯度余弦相似度，再乘以无监督梯度范数相对最大值；[`138–148`](https://github.com/gardnerzhou/CHAP/blob/8754bfa5e5d269719b0ca45345c8a28742b13c0d/utils/grad.py#L138-L148) 对历史梯度做余弦调度动量平滑。上游在这些位置没有显式处理全零梯度的 `norm/max_norm` 除零，本项目诊断应对缺失、零范数及 NaN/Inf 给出独立状态和 JSON `null`，不能继承不稳定除法。

[`train_ours_3D.py:269–303`](https://github.com/gardnerzhou/CHAP/blob/8754bfa5e5d269719b0ca45345c8a28742b13c0d/train_ours_3D.py#L269-L303) 使用历史相似度触发通道扰动，再做梯度分析，最终重新 zero_grad/backward/step；其 [`networks/FilterDropout.py:95–146`](https://github.com/gardnerzhou/CHAP/blob/8754bfa5e5d269719b0ca45345c8a28742b13c0d/networks/FilterDropout.py#L95-L146) 对无标签特征做通道 mask，[`175–218`](https://github.com/gardnerzhou/CHAP/blob/8754bfa5e5d269719b0ca45345c8a28742b13c0d/networks/FilterDropout.py#L175-L218) 把梯度相似度与池化激活组合后以 sigmoid 概率做 Bernoulli 采样。这条实际 3D 训练路径不是固定选择边界梯度 Top 25% 通道。`utils/get_gradient.py` 另有取某卷积权重梯度绝对值平均的辅助函数，也不能视为特征梯度接口。

本阶段只借鉴“把不同损失的梯度分开观察”这一思想：按固定 iteration 间隔，对五个**完整阶段参数组**分别用 `torch.autograd.grad(..., retain_graph=True, allow_unused=True)` 计算 `L_MT` 和加权边界损失的梯度，输出范数、比值、余弦与显式异常状态。B2 必须分析**实际路由后**边界贡献（每阶段只含其自身边界项），不能把普通深度边界总损失的天然跨层梯度冒充局部模式的实际更新。可另标记记录未路由数学梯度作研究对照，二者不得混用。

`grad(L_boundary_i, feature_i)` 应作为独立可选诊断接口保留，返回与 feature 同形状的张量；它不等于上游 CHAP 的卷积核梯度。不写入参数 `.grad`、不重新前向、不改变 optimizer/EMA/BN/dropout 状态，诊断前后训练参数更新须数值一致。当前不实现通道排名、筛选、mask、特征交换或双向交互。

## 结论边界

已确认的是上游头位置、目标类型、损失聚合和梯度分析代码的事实，以及上面列明的适配差异。辅助监督是否有助于 LA 分割、不同梯度范围是否缓解冲突、任何诊断指标能否预测性能，都仍需后续正式实验；这些不是源码审计能够证明的结论。
