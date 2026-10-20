# 执行验证记录

更新：2026-10-20T15:17:29.597898+08:00。以下为实际执行证据，不包含新正式训练结果。

| 测试套件 | 执行数 | 失败 | 错误 | 跳过 |
| --- | ---: | ---: | ---: | ---: |
| synthetic_boundary_tests | 16 | 0 | 0 | 0 |
| synthetic_decoder_tests | 8 | 0 | 0 | 0 |
| legacy_mode_regression | 10 | 0 | 0 | 0 |

共 34 项自动化测试通过；真实GPU同图诊断前后状态隔离检查通过（零optimizer更新）。

**GPU跨运行严格一致性检查未通过**：保留原失败记录。必需的CUDA三线性插值反向算子已测出非确定性，strict deterministic会报错；不宣称GPU逐位复现或全部误差的唯一成因。

本报告引用的A2、B2、C每个Smoke均为两次训练更新和一次原始20例验证。Teacher序列化后严格加载通过，
loss、梯度和验证指标均有限，配置与快照哈希已复核。Smoke Dice 不用于评价新方法。

A2/B2 方法配置只有 boundary_mode 不同；有限Smoke另有GPU编号cuda:5/cuda:3差异，均为A100。源码哈希、数据 manifest 相同。
C 的边界目标/损失/头明确为none，final权重1且只算一次，三个侧头均值总权重0.1。
A2/B2使用最终post-backward诊断版本，最初A2/B2 Smoke仍保留在refactor_smoke_20261020。C此前完成，边界互斥分支外的Trainer及VNet代码经AST比较相同；无需重跑未改动的C训练路径。

另外保留GPU诊断启停比较的8次真实更新；两项算子/同图隔离探针均为零optimizer更新。这些均为有限检查。

验证/最终测试仍使用原 LA/test.list，未改成独立划分。原 A/B 的18个关键产物 SHA256 保持不变。
新三组完整30000次训练均未启动。正式命令仅输出供后续独立开展，不自动执行。

[机器可读完整报告](artifacts/validation_summary.json)；[梯度诊断](GRADIENT_DIAGNOSTICS.md)；
[固定真实标签比较](artifacts/label_audit_seed42/label_audit_report.md)。
