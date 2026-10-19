# BCSI Baseline：固定 lr=0.01，seed=42，10% / 20%

这是一组独立的 Mean-Teacher baseline 对照实验，共六次训练。
训练源代码来自 `Baseline/BCSI_Baseline`，结果写入
`Results/fixed_lr_0p01_seed42/`。主目录的训练代码及已有结果不做修改。

| 数据集 | 训练病例总数 | 10% 标注病例 | 20% 标注病例 | patch | GPU |
| --- | ---: | ---: | ---: | --- | ---: |
| LA | 80 | 8 | 16 | 112 × 112 × 80 | 2 |
| Pancreas | 62 | 6 | 12 | 96 × 96 × 96 | 4 |
| BraTS2019 | 250 | 25 | 50 | 96 × 96 × 96 | 6 |

唯一的训练方法变化是：代码快照中的 `trainer.py` 将
`PolyWarmRestartScheduler` 实例替换为 `LambdaLR(lr_lambda=lambda step: 1.0)`，
使 SGD 的学习率从初始化到第 30,000 次更新始终为 0.01，取消原有的
8,000 次周期衰减和重启。原来的调度器类仍保留于快照中，但不再实例化。
预备阶段记录原始源码和实际训练快照的 SHA-256；启动时核对快照和数据划分。

其他设置沿用 baseline：seed=42、fold=0、30,000 次迭代、batch size=4、
labeled batch size=2、SGD momentum=0.9、weight decay=1e-5、EMA decay=0.9、
consistency=0.1、consistency rampup=200；每 200 次迭代验证 EMA teacher。
标注样本仍是原 `train.list` 的前 N 个病例，不重新洗牌。
LA/Pancreas 使用原有 test.list 进行训练期验证；BraTS2019 使用 val.list。

每张 GPU 同时运行该数据集的 10% 和 20% 两组。每组训练完成后，自动用
最佳验证 Dice 的 EMA checkpoint 执行原 `prediction.py`，保留原有滑窗步长和
最大连通域后处理；LA/Pancreas 最后评估与训练期验证使用相同病例，
BraTS2019 最后评估使用独立 test.list。

```bash
cd /home/duanxuelong/Baseline/BCSI_Baseline
conda run -n SSL python -u -B experiments/fixed_lr_0p01_seed42/run_experiment.py --prepare --gpus 2 4 6
conda run -n SSL python -u -B experiments/fixed_lr_0p01_seed42/verify_experiment.py
# 后台运行时应在 tmux 内执行下一条命令。
conda run -n SSL python -u -B experiments/fixed_lr_0p01_seed42/run_experiment.py --run
```

`--prepare` 拒绝覆盖已有实验目录；`--run` 仅允许启动 `prepared` 状态的实验，
并持有文件锁防止重复启动。`--run` 使用 metadata 中已保存的 GPU 分配。
运行需要可以访问 CUDA 的宿主机环境。

结果目录包含：

- `metadata.json`：六组命令、原始/快照源码哈希、数据划分和标注病例 ID。
- `source_snapshot/`：实际执行的独立源码。
- `experiment_source/`：本实验启动器、说明和验证脚本的副本。
- `data_splits/`：原始列表副本；实际训练使用 metadata 指定的数据路径。
- `verification.json`：30,000 次调度更新、数据加载和隔离检查结果。
- `run_status.json`：每 10 秒更新进度、GPU、PID、lr、loss、验证 Dice/HD95。
- `summary.csv`：六组状态、最佳验证 Dice 和最终测试 Dice/IoU/HD95/ASD。
- `seed_42_10pct/` 和 `seed_42_20pct/`：分组日志及模型。

每组的 `result_<dataset>_<percent>l/fold_0/` 保留原始训练日志、code、最佳模型，
最终还包含 `per_case_metrics.csv`、`test_metrics.json` 和 `test_command.json`。
所有 Dice/IoU 都是 0–1 的原始值。若发现非有限 loss/metric 或 lr 偏离 0.01，
启动器将该组标记失败，并继续运行其他组；详情见对应 console/test 日志。
