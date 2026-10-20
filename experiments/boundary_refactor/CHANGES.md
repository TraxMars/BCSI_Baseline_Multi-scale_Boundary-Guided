# 修改文件与实现边界

| 文件 | 改动与目的 |
| --- | --- |
| `model/vnet.py` | Decoder 新增可选 `return_features`，暴露原 skip-add 张量；默认前向、参数、拓扑与 checkpoint key 不变。 |
| `trainer.py` | 新四头工厂、A2 普通反向/B2 定向路由、独立 C 损失、反向之后的只读诊断、独立辅助状态及训练记录。MT、SGD、参数 EMA、scheduler 和 Teacher 选优仍沿用原逻辑。 |
| `train.py` | 新模式/权重/诊断参数、独立拒绝覆盖目录、方法配置/源码快照、明确评估列表来源。 |
| `utils/bfa_supervision.py` | labeled-only 四个 BFA 头、一次原图 Laplacian 目标、trilinear logits、四层 BCE+Dice 均值及保图特征梯度接口。 |
| `utils/decoder_supervision.py` | 原解码器三个分割侧头，nearest GT，原 CE+Dice，辅助均值不包含已计算的 final。 |
| `utils/gradient_diagnostics.py` | 主/边界梯度范数、比值、余弦及显式异常状态；区分 raw 与 local 缓存贡献，不修改 `.grad`。 |
| `.gitignore` | 仅允许本次固定 patch 的四张审计 PNG；checkpoint、数据和其它大文件仍沿用忽略规则。 |
| `experiments/boundary_refactor/` | 审计、三个真实源码来源报告、修改前快照、34项自动化测试证据、真实标签图/统计、有限GPU诊断及Smoke记录、启动器和正式命令预览。 |
| `Results/refactor_smoke_20261020/` | 首轮 A2/B2/C 各两步训练、20例验证及当时源码快照；权重留在本地。 |
| `Results/refactor_smoke_postdiag_20261020/` | 诊断时序调整后的最终 A2/B2 各两步训练与20例验证，独立新目录。 |

`prediction.py`、`utils/losses.py`、`utils/utils.py`、数据加载器及旧 `utils/boundary_supervision.py` 未修改。
后者旧注释仍描述五层的 `/5`；实际函数通过 `len(result.losses)` 归一化，四层模式正确为 `/4`，已独立验证。
旧 `deep/local`、原模式代码快照和所有既有结果均保留；18个旧关键产物的保护哈希再次一致。

新增接口的原因是读取现有 decoder 中间张量和独立管理辅助模块，而不是改变 VNet 主体计算。
自动化测试检验默认前向、反向、连续MT更新、BN/RNG、原 checkpoint 严格加载及推理一致。
`train.py` 最后只清理新增行的 CR 行尾；AST 校验与 Smoke 快照相同，不重写历史快照。

新增GPU诊断的初版跨运行严格一致性检查失败；记录保留且未放宽容差。
最终诊断时序为实际反向和 local 累加之后、SGD之前。同图调用前后已有状态逐位不变的独立检查通过。
必需CUDA三线性插值反向已测出非确定性；不把其当作所有运行间误差的已证实唯一原因。
这些限制和 LA 验证/测试共用列表均见 [验证报告](VALIDATION.md)。

本次尚未进行正式性能对照、权重搜索、独立测试或多seed统计；研究假设仍待正式实验。
没有加入无标签LoG、通道筛选、交换、原型、双向交互，也没有改选优策略。
