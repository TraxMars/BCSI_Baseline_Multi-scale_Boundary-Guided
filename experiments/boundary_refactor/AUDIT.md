# 多尺度监督审计：实现前结论

审计基线：远程 main `995c354661da83a8b886d499a02ab98f7cd550ee`。
开发分支：`experiment/bfa-a2-b2-decoder-ds`；独立 worktree 不重置原 main 工作区。
本报告在新实现开始前形成。新实验测试结果另记于验证报告，不能反向改写旧实验定义。

## 结论分类

### 已确认的代码错误

没有证据表明旧 A/B 的边界梯度范围或 MT 主损失实现错误。
旧梯度测试真实复跑 10/10 通过，局部实现对自己的 Ei+Head_i 获取梯度再与 MT 梯度相加。
旧标签函数正确实现了当时约定的 nearest 下采样后形态学梯度，不能将设计需要调整写成代码 bug。
移植上游时需要避免的具体问题是 BFA 的 CUDA 类型硬编码及仅比较最后一个空间维的插值判断；
这些是上游兼容性风险，不是旧 A/B 性能下降的已证实原因。

### 已确认的方法设计差异及协议限制

1. **MT/EMA**：原 baseline 与旧实验的 CE+Dice、有无标签 MSE、rampup、SGD、
   8000 周期 scheduler 和参数 EMA 一致。Teacher 训练前向仍为 train 模式，
   BN buffer 自身更新且 dropout 开启；参数 EMA 不处理 BN buffer。这是原 baseline 行为，本次不更改。
2. **A/B 路由**：A 的 Li 作用于 E1…Ei，B 的 Li 仅给 Ei 和自己的 head 贡献参数梯度。
   监督和一致性损失仍更新整个 Student。两组源码哈希相同、配置仅 mode 不同，
   30000 行目标记录全部一致，未确认“局部实现失败”。
3. **边界膨胀**：旧方法在每层缩小 GT 后使用同一 3³ 形态学核，物理厚度随尺度增加。
   60000 个有标签 patch 的 E1…E5 平均覆盖率为
   5.0116%、9.9077%、19.0467%、34.9200%、55.8672%。
   E5 为 7×7×5，单 patch 最大 87.3469%，1355/60000（2.2583%）达到 ≥80%；
   未观察到空边界或前景消失。这是设计上的粗尺度厚边界问题。
4. **与 BFA 不同**：旧版为五个 1³ 线性头、逐层不同的对称形态学目标；BFA 只用 x1…x4，
   3³ C→16 + BN + ReLU + 1³ 16→1，以原图 Laplacian 前景内侧边界监督上采样预测。
5. **新方案仍是适配**：A2/B2 只将 labeled 特征送入辅助头，避免头 BN 混入无标签样本；
   编码器 BN 的原始混合 batch 依赖仍保留。上游 BFA self-train 先对全 batch 运行 heads，
   还包含强增强和无标签 LoG，且边界损失四层求和；本次均不迁移，只使用指定的四层平均×0.1。
   Soft Dice 固定采用 BFA 的 smooth=1；旧 deep/local 的 smooth=1e-5 不变。
6. **DuCiSC 对照**：真实侧头位于 skip-add 后、后续 decoder block 前，尺度 /8、/4、/2、1。
   新 C 保留相同侧头位置和原最终预测，GT nearest 缩放，保留当前 MT 的 CE+Dice。
   final 主监督权重默认 1，三侧头损失均值总权重默认 0.1，不重复计 final。
   不迁移上游四输出平均、自归一化、原型、mix 或多尺度伪标签机制。
7. **验证不独立**：LA 的 validation 和 final test 均读取同一 test.list（20例）；
   train.list 为80例、前8例有标签。当前任务不改划分，只明确记录这项协议限制。
   BraTS 分别来自 val.list/test.list。所有新实验沿用原函数和选优策略。
8. **参照基线混杂**：匹配旧 A/B 的 warm-restart baseline 最佳日志 Dice=0.873561@27000，
   HD95=8.888493。独立 fixed_lr_0p01 结果使用另一个 scheduler，并存在 model-only 恢复记录，
   不能当作仅边界因素变化的对照。源码/命令以匹配的 warm-restart 原 MT 为准。

### 尚待实验验证的假设

- 粗尺度膨胀、E5 监督、头表达能力或参数梯度冲突是否解释旧 A 与 B 的差异，均不是性能数字可证明的因果关系。
- A2/B2 可能改善边界表达，但其性能及最优边界权重需要新正式实验验证。
- C 用于区分普通多尺度分割监督与显式边界学习的作用，不能预先认定哪种更好。
- 梯度余弦为负只代表某个 batch/阶段的局部梯度方向；短程初始化诊断不能证明完整训练性能的原因。
- 单 seed、同验证/测试列表的结果不构成独立测试泛化或统计显著性结论。

## 新实现前固定的选择

- 保留 old `deep/local` 函数、代码快照及 Results/Boundary_A、Boundary_B，新增 `bfa_deep/bfa_local`。
- A2/B2：同四头、同 Kaiming(a=1) 初始化、同 full-resolution Laplacian 目标、同损失/权重/数据。
- Laplacian kernel 用 buffer，默认 `zero` 保留 BFA reference；显式 `replicate` 仅供边缘验证，不能单独调一组。
- 实际 GT 只提取一次目标，四预测按全部空间维判断并 trilinear/align_corners=True 恢复原尺寸。
- E5 编码和解码照常训练，但无第五边界头或损失；无无标签 LoG。
- C 与边界模式互斥，独立 auxiliary_mode；原最终损失不重复计算。
- 梯度诊断用 autograd.grad、retain_graph，不重跑前向、不写 .grad、不用额外随机数。
  B2 同时区分完整边界标量的 raw gradient 和实际路由的 applied gradient，避免错误解读。
- 不筛通道、不交换特征、不做双向交互。只允许短程 Smoke Test，完整训练留给独立后续实验。

## 可复核证据

- [旧实现及全部30000行审计](AUDIT_LEGACY.md)，[机器可读证据](legacy_audit.json)。
- [BFA 源码、commit及精确行引用](REFERENCES_BFA.md)。
- [DuCiSC、CHAP 源码、commit及精确行引用](REFERENCES_DUCISC_CHAP.md)。
- BFA：`74c2f2da2c2a399148dfcbb4750ea016b1941cbe`。
- DuCiSC：`cc77a8308d0aa1c59a7c29556478bfd33d804a1a`。
- CHAP：`8754bfa5e5d269719b0ca45345c8a28742b13c0d`。

这些源码已经实际获取和阅读；不是依据论文摘要推断。CHAP 的参数梯度诊断只提供研究启发，
不将其概率扰动写成“Top25%复现”，也不迁移其算法到本阶段。
