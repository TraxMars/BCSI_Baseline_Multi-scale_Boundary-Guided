# BFA 真实源码审计与本实验的边界

审计日期：2026-10-20（Asia/Shanghai）。审计对象为官方仓库 `hyg902/BFA-main` 的 commit [`74c2f2da2c2a399148dfcbb4750ea016b1941cbe`](https://github.com/hyg902/BFA-main/commit/74c2f2da2c2a399148dfcbb4750ea016b1941cbe)。通过 GitHub connector 读取原始文件并核对 Git blob SHA；本地临时原文位于 `/tmp/bcsi-boundary-upstream/BFA-main/`，并非训练输入。持久来源与 SHA-256 记录见 `references_bfa.json`。本文依据实际代码，不依据论文摘要。

## 可核实的上游行为

| 内容 | 源码依据 | 审计结论 |
|---|---|---|
| 边界头 | [VNet_BFA.py:9–51](https://github.com/hyg902/BFA-main/blob/74c2f2da2c2a399148dfcbb4750ea016b1941cbe/code/networks/VNet_BFA.py#L9-L51) | 3×3×3 `C→16` 卷积（无 bias）、BatchNorm3d、ReLU、1×1×1 `16→1` 卷积（无 bias）；两次卷积均采用 Kaiming normal、`a=1`。头输出 logits，没有 sigmoid。 |
| 层数与上采样 | [VNet_BFA.py:310–313](https://github.com/hyg902/BFA-main/blob/74c2f2da2c2a399148dfcbb4750ea016b1941cbe/code/networks/VNet_BFA.py#L310-L313)、[351–365](https://github.com/hyg902/BFA-main/blob/74c2f2da2c2a399148dfcbb4750ea016b1941cbe/code/networks/VNet_BFA.py#L351-L365) | 对 x1、x2、x3、x4 设置四头，输入通道 16、32、64、128。x2–x4 的 logits 用 trilinear、`align_corners=True` 放大到最终分割输出分辨率；没有 x5 边界头。 |
| 标签生成 | [bfa_3D.py:29–50](https://github.com/hyg902/BFA-main/blob/74c2f2da2c2a399148dfcbb4750ea016b1941cbe/code/BFA_loss/bfa_3D.py#L29-L50) | 使用中心 26、其余 26 位置 −1 的 3×3×3 核；`conv3d(padding=1)` 使用零填充；随后 clamp 到非负并以 `>0.1` 二值化。原始 GT 直接进入边界构造，无各层 GT 下采样。源码同时构造的中心 6 核没有进入此 forward。 |
| 边界损失 | [bfa_3D.py:8–16](https://github.com/hyg902/BFA-main/blob/74c2f2da2c2a399148dfcbb4750ea016b1941cbe/code/BFA_loss/bfa_3D.py#L8-L16)、[40–51](https://github.com/hyg902/BFA-main/blob/74c2f2da2c2a399148dfcbb4750ea016b1941cbe/code/BFA_loss/bfa_3D.py#L40-L51) | BCEWithLogits 接受 logits；Dice 内只应用一次 sigmoid，按样本 flatten，分母为概率和与目标和，smooth=1。函数分别返回 BCE 与 Dice。fallback 上采样仅检查最后一个维度，移植时应检查全部三个空间维度。 |
| 有标签预训练 | [LA_BFA_train.py:161–199](https://github.com/hyg902/BFA-main/blob/74c2f2da2c2a399148dfcbb4750ea016b1941cbe/code/LA_BFA_train.py#L161-L199) | 整个输入切为 labeled 子批，再前向；四层 BCE 与 Dice 直接相加到 0.5×(CE+Dice) 上。普通 backward。 |
| 自训练路径 | [LA_BFA_train.py:269–338](https://github.com/hyg902/BFA-main/blob/74c2f2da2c2a399148dfcbb4750ea016b1941cbe/code/LA_BFA_train.py#L269-L338) | Student 输入是 labeled 加强扰动后的 unlabeled；全部输入先通过边界头，之后才按 labeled/unlabeled 切输出，因此头内 BN 统计跨两类样本。Teacher 对 unlabeled 生成带最大连通域处理的伪标签；无标签边界使用 LoG。总损失含监督分割、无标签 CE/余弦项和两组边界项。 |
| LoG 与训练控制 | [bfa_3D.py:59–90](https://github.com/hyg902/BFA-main/blob/74c2f2da2c2a399148dfcbb4750ea016b1941cbe/code/BFA_loss/bfa_3D.py#L59-L90)、[LA_BFA_train.py:335–376](https://github.com/hyg902/BFA-main/blob/74c2f2da2c2a399148dfcbb4750ea016b1941cbe/code/LA_BFA_train.py#L335-L376) | LoG 为 size=3、sigma=0.3 的 Gaussian 再接该 Laplacian。原版不是本仓库的 MT MSE 训练：EMA=0.99，预训练后自训练；验证 Student。原版计算的 consistency_weight 仅写日志，未乘进其最终 loss。 |

## 方法意义与 crop padding 验证

二值标签上，Laplacian clamp 后保留的是**前景内侧边界**：当前体素为前景且 26 邻域至少存在背景时为 1。背景体素不会被标为边界。这与旧版 dilation−erosion 的双侧边界不同；旧版再按尺度缩小标签，会令固定 3×3×3 形态学算子的原图等效厚度随层加大。旧法是先前实验的设计，不因其性能下降而自动构成梯度代码错误。

参考模式应默认保留上游 zero padding。若前景碰到裁剪面，外部虚构为背景会在该面生成目标边界；对全前景体积，正体素数为 `D*H*W-(D-2)*(H-2)*(W-2)`（各维至少为 3）。这是须记录的监督设计取舍。可提供 `replicate` 显式选项用于验证：全前景体积的边界应为 0；完全处于内部的物体，两种 padding 的目标应一致。zero 与 replicate 的差异只可能在体积最外一层，且 Laplacian 两种模式都只标前景体素。不得为 A2/B2 单独选择不同 padding 或根据验证 Dice 静默切换。

## A2/B2 有意采用的适配

新实验是 **BFA 参考的编码器边界监督**，并非完整复现 BFA：保持本仓库原始 MT 的 MSE、一致性调度、EMA、SGD、验证 Teacher 与推理接口；不移植 BFA 的预训练阶段、强扰动、无标签 CE/余弦、LoG、最大连通域伪标签处理或附加投影模块。边界头应为 Student 独立模块，使用 labeled 特征子批；这与 BFA 自训练的整批 BN 路径存在明确差异，但 A2/B2 两者一致。

新目标只构造一张原始 GT 分辨率边界并共享至四头。每层采用 `0.5*BCEWithLogits + 0.5*SoftDice`，四层求均值，再乘初始可配权重 0.1。原版使用跨层 BCE+Dice 直接求和，还包含无标签边界项，因此两者的损失绝对尺度不同。若采用上游 smooth=1，应在新配置中固定并记录；不能只对某一组改变。设备相关 `.type(torch.cuda.FloatTensor)` 应由设备无关 buffer 实现替换，阈值、核和 padding 的参考语义保持明确。

A2 使用普通 backward，因此第 i 个边界损失可到达 E1…Ei；B2 使用定向 autograd.grad，只累加 Ei 与第 i 个头。两者保留同一 forward、初始权重、目标、损失权重与样本流。该局部路由不是 BFA 原版行为，而是本研究的受控变量。性能是否改善、梯度是否与 MT 冲突及 padding 的实际影响均需要独立数值诊断和正式实验，不能由旧 A/B 的性能数值直接判定。
