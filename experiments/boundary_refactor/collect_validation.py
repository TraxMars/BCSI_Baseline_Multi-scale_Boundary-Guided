"""Collect completed checks and finite real smoke evidence without efficacy claims."""
import ast
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch
from model.vnet import VNet


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def float_or_none(value):
    return None if value == '' else float(value)


def main():
    report = dict(generated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        base_commit='995c354661da83a8b886d499a02ab98f7cd550ee',
        development_branch='experiment/bfa-a2-b2-decoder-ds', tests={}, smoke={},
        formal_training_started=False)
    for name in ('synthetic_boundary_tests', 'synthetic_decoder_tests', 'legacy_mode_regression'):
        value = read_json(HERE / 'artifacts' / (name + '.json'))
        assert value['passed'] and value['failures'] == 0 and value['errors'] == 0
        report['tests'][name] = {key: value.get(key, 0) for key in
                                 ('tests_run', 'failures', 'errors', 'skipped', 'passed')}
    report['tests_run'] = sum(value['tests_run'] for value in report['tests'].values())
    cross_run = read_json(HERE / 'artifacts/gpu_diagnostic_invariance.json')
    report['gpu_diagnostic_invariance'] = dict(passed=cross_run['passed'],
        bitwise_equal=cross_run['bitwise_equal'], rtol=cross_run['rtol'], atol=cross_run['atol'],
        evidence='artifacts/gpu_diagnostic_invariance.json')
    observation = read_json(HERE / 'artifacts/gpu_diagnostic_observation.json')
    assert observation['passed']
    report['gpu_diagnostic_observation'] = dict(passed=True, optimizer_updates=0,
        modes={mode: value['comparison']['bitwise_equal'] for mode, value in observation['modes'].items()},
        evidence='artifacts/gpu_diagnostic_observation.json')
    for relative, expected in observation['source_sha256'].items():
        assert digest(ROOT / relative) == expected
    report['gpu_trilinear_reproducibility'] = read_json(HERE / 'artifacts/gpu_trilinear_reproducibility.json')
    report['limitations'] = [
        'Strict cross-run GPU diagnostic on/off comparison failed its original tolerances; retained without relaxation.',
        'CUDA trilinear backward is empirically nondeterministic; strict deterministic algorithms reject it.',
        'The observed operator nondeterminism is not proof of the cause of every cross-run discrepancy.',
        'LA validation and final test reuse the same original test.list; no independent test split.',
        'Formal A2/B2/C training and efficacy comparisons have not been performed.']
    folders = {}
    names = ('BFA_Deep_A2', 'BFA_Local_B2', 'Decoder_DS_C')
    for name in names:
        smoke_root = ('Results/refactor_smoke_20261020' if name == 'Decoder_DS_C'
                      else 'Results/refactor_smoke_postdiag_20261020')
        fold = ROOT / smoke_root / name / 'seed_42/result_LA_10l/fold_0'
        folders[name] = fold
        status = read_json(fold / 'run_status.json')
        assert status['state'] == 'completed' and status['iteration'] == 2
        config = read_json(fold / 'experiment_config.json')
        train = read_csv(fold / 'training_metrics.csv')
        validation = read_csv(fold / 'validation_metrics.csv')
        assert [int(row['iteration']) for row in train] == [0, 1]
        assert len(validation) == 1 and int(validation[0]['iteration']) == 2
        assert all(math.isfinite(float(v)) for v in validation[0].values())
        for row in train:
            assert all(math.isfinite(float(row[key])) for key in
                       ('supervised_loss', 'consistency_loss', 'consistency_weight', 'total_loss'))
            expected = float(row['supervised_loss']) + float(row['consistency_weight']) * float(row['consistency_loss'])
            if name == 'Decoder_DS_C':
                assert float(row['final_supervision_weight']) == 1
                expected += 0.1 * float(row['total_decoder_aux_loss'])
            else:
                expected += 0.1 * float(row['total_boundary_loss'])
                assert all(math.isfinite(float(row[f'boundary_loss_{i}'])) for i in range(1, 5))
                assert len({row[f'boundary_positive_ratio_{i}'] for i in range(1, 5)}) == 1
            assert math.isclose(float(row['total_loss']), expected, abs_tol=1e-6)
        for relative, expected in config['source_sha256'].items():
            assert digest(fold / 'code' / relative) == expected
        model = VNet()
        model.load_state_dict(torch.load(fold / 'Model_iter_2.pth', map_location='cpu', weights_only=True), strict=True)
        assert all(torch.isfinite(p).all() for p in model.parameters())
        manifest = read_json(fold / 'data_split_manifest.json')
        assert manifest['evaluation']['validation_test_same_list']
        assert manifest['evaluation']['validation_list'].endswith('/LA/test.list')
        assert manifest['evaluation']['validation_sha256'] == manifest['evaluation']['test_sha256']
        assert len(manifest['splits']['train']['ids']) == 80 and len(manifest['splits']['test']['ids']) == 20
        assert len(manifest['labeled_indices']) == 8
        report['smoke'][name] = dict(path=str(fold.relative_to(ROOT)), status=status,
            train_iterations=2, validation_cases=20, validation=validation[0],
            teacher_checkpoint_strict_load=True, data_evaluation=manifest['evaluation'],
            source_snapshots_verified=True,
            note='Smoke metrics test the pipeline only; they are not formal experimental results.')
        assert not (ROOT / 'Results' / name).exists(), 'Unexpected formal output directory'
    a, b = [read_json(folders[name] / 'experiment_config.json') for name in names[:2]]
    differences = [key for key in a['args'] if a['args'][key] != b['args'][key]]
    assert set(differences) == {'boundary_mode', 'device'}
    assert all(a['args'][key] == b['args'][key] for key in a['args']
               if key not in ('boundary_mode', 'device'))
    assert a['source_sha256'] == b['source_sha256']
    final_source = {}
    for relative, expected in a['source_sha256'].items():
        current = ROOT / relative
        snapshot = folders[names[0]] / 'code' / relative
        same_bytes = digest(current) == expected
        same_ast = ast.dump(ast.parse(current.read_text())) == ast.dump(ast.parse(snapshot.read_text()))
        assert same_ast
        final_source[relative] = dict(byte_identical=same_bytes, ast_identical=same_ast)
    assert all(value['byte_identical'] for key, value in final_source.items() if key != 'train.py')
    report['final_source_vs_completed_smoke'] = final_source
    assert read_json(folders[names[0]] / 'data_split_manifest.json') == read_json(
        folders[names[1]] / 'data_split_manifest.json')
    report['AB_fairness'] = dict(configuration_differences=differences,
        scientific_configuration_differences=['boundary_mode'],
        device_placement=dict(A2=a['args']['device'], B2=b['args']['device']),
        source_hashes_identical=True, data_manifests_identical=True,
        initial_forward_exactly_equal='verified in CPU synthetic tests')
    # C ran before the post-backward boundary-only diagnostic timing refinement.
    # Remove that mutually exclusive branch to verify every other Trainer path.
    class OmitBoundaryTraining(ast.NodeTransformer):
        def visit_If(self, node):
            if (ast.unparse(node.test) == 'self.boundary is not None' and
                    any(isinstance(child, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id == 'boundary_result'
                        for target in child.targets) for child in node.body)):
                node.body = [ast.Pass()]
            return self.generic_visit(node)
    code = {}
    for name in names:
        code[name] = ast.dump(OmitBoundaryTraining().visit(ast.parse(
            (folders[name] / 'code/trainer.py').read_text())))
    assert len(set(code.values())) == 1
    code = {name: ast.dump(ast.parse((folders[name] / 'code/model/vnet.py').read_text())) for name in names}
    assert len(set(code.values())) == 1
    c = read_json(folders[names[2]] / 'experiment_config.json')
    assert c['boundary_target'] == c['boundary_loss'] == c['boundary_head'] == 'none'
    assert c['loss_weights'] == dict(final_segmentation=1.0, boundary_total=0.0,
                                    decoder_sides_total=0.1, decoder_side_count=3)
    report['C_independent'] = dict(boundary_mode='none', final_loss_count=1,
        final_weight=1, side_count=3, side_total_weight=0.1,
        nonboundary_trainer_and_vnet_math_identical_across_smokes=True,
        source_note='C smoke preceded boundary-only post-backward diagnostic timing change; C code path unchanged.')
    diagnostics = {}
    for name in names[:2]:
        rows = read_csv(folders[name] / 'gradient_diagnostics.csv')
        assert len(rows) == 10
        diagnostics[name] = []
        for stage in range(1, 6):
            samples = [row for row in rows if int(row['stage']) == stage]
            assert len(samples) == 2
            cosines = [float_or_none(row['cosine']) for row in samples]
            valid_cosines = [v for v in cosines if v is not None]
            assert all(math.isfinite(float(row['main_norm'])) and math.isfinite(float(row['boundary_norm']))
                       for row in samples)
            if stage == 5:
                assert all(float(row['boundary_norm']) == 0 and row['cosine'] == '' for row in samples)
            diagnostics[name].append(dict(stage=stage, samples=2,
                main_norm_mean=sum(float(r['main_norm']) for r in samples)/2,
                weighted_boundary_norm_mean=sum(float(r['boundary_norm']) for r in samples)/2,
                ratio_mean=sum(float(r['ratio']) for r in samples)/2,
                cosine_mean=sum(valid_cosines)/len(valid_cosines) if valid_cosines else None,
                opposite_count=sum(row['opposite'] == 'True' for row in samples),
                significant_reverse_count=sum(row['reverse'] == 'True' for row in samples),
                raw_boundary_norm_mean=sum(float(r['raw_boundary_norm']) for r in samples)/2,
                statuses=sorted({row['status'] for row in samples})))
    report['gradient_diagnostics'] = diagnostics
    protection = read_json(HERE / 'artifacts/legacy_protection_check.json')
    for row in protection['checks']:
        assert digest(Path(row['path'])) == row['expected_sha256'], row['path']
    report['historical_artifacts_unchanged'] = dict(checked_files=len(protection['checks']), passed=True)
    report['required_automated_tests_passed'] = True
    report['status'] = 'completed_with_disclosed_gpu_reproducibility_limitation'
    (HERE / 'artifacts/validation_summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    lines = ['# 梯度诊断：真实 LA 两步 Smoke Test', '',
        '采样 iteration 0、1；原 patch 112×112×80、batch 4/2、seed42、boundary_weight=0.1。',
        '主损失包括原 supervised CE+Dice 与原权重的 MT 一致性损失。',
        'B2使用缓存并实际累加的局部边界梯度；A2使用同一目标重新求导的边界贡献估计。',
        '由于CUDA反向非确定，A2估计不声称逐位等于combined backward中的实际边界分量。B2另保留未路由raw梯度。', '',
        '| 实验 | 阶段 | 主梯度范数均值 | 加权边界范数均值 | 范数比均值 | 余弦均值 | 负方向次数 | 显著反向次数 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, stages in diagnostics.items():
        for row in stages:
            cosine = '未定义（无边界梯度）' if row['cosine_mean'] is None else f"{row['cosine_mean']:.6f}"
            lines.append(f"| {name} | E{row['stage']} | {row['main_norm_mean']:.6f} | "
                f"{row['weighted_boundary_norm_mean']:.6f} | {row['ratio_mean']:.6f} | {cosine} | "
                f"{row['opposite_count']}/2 | {row['significant_reverse_count']}/2 |")
    lines += ['', '显著反向的预先固定定义为 cosine≤−0.2 且加权边界/主梯度范数比≥0.1；未按结果修改阈值。',
        'E5 无边界损失，因此范数为0、余弦未定义，CSV 中有显式 missing_boundary_gradient 状态。',
        '这些是初始化附近两个 batch 的观察，不能证明完整训练有无梯度冲突，更不能据此解释旧 A/B 的性能因果。',
        '诊断安排在实际优化梯度计算及local累加之后、SGD之前。真实GPU同图检查确认调用前后已有.grad、模型/头/Teacher状态、RNG及优化器状态逐位相同；见 gpu_diagnostic_observation.json。',
        'CPU诊断启停完整状态回归通过。独立GPU两次运行的严格比较超过原容差，gpu_diagnostic_invariance.json明确保留failed；没有放宽容差或将其改写成通过。',
        '同一CUDA trilinear图重复backward已测出差异，strict deterministic报无实现，见gpu_trilinear_reproducibility.json；这说明GPU严格重复性受限，但不能证明每个跨运行差异均由该算子导致。',
        '当前没有据此做通道选择、梯度投影或修改优化。', '']
    (HERE / 'GRADIENT_DIAGNOSTICS.md').write_text('\n'.join(lines))
    lines = ['# 执行验证记录', '',
        f"更新：{report['generated_at']}。以下为实际执行证据，不包含新正式训练结果。", '',
        '| 测试套件 | 执行数 | 失败 | 错误 | 跳过 |', '| --- | ---: | ---: | ---: | ---: |']
    for name, value in report['tests'].items():
        lines.append(f"| {name} | {value['tests_run']} | {value['failures']} | {value['errors']} | {value['skipped']} |")
    lines += ['', f"共 {report['tests_run']} 项自动化测试通过；真实GPU同图诊断前后状态隔离检查通过（零optimizer更新）。", '',
        '**GPU跨运行严格一致性检查未通过**：保留原失败记录。必需的CUDA三线性插值反向算子已测出非确定性，strict deterministic会报错；不宣称GPU逐位复现或全部误差的唯一成因。', '',
        '本报告引用的A2、B2、C每个Smoke均为两次训练更新和一次原始20例验证。Teacher序列化后严格加载通过，',
        'loss、梯度和验证指标均有限，配置与快照哈希已复核。Smoke Dice 不用于评价新方法。', '',
        'A2/B2 方法配置只有 boundary_mode 不同；有限Smoke另有GPU编号cuda:5/cuda:3差异，均为A100。源码哈希、数据 manifest 相同。',
        'C 的边界目标/损失/头明确为none，final权重1且只算一次，三个侧头均值总权重0.1。',
        'A2/B2使用最终post-backward诊断版本，最初A2/B2 Smoke仍保留在refactor_smoke_20261020。C此前完成，边界互斥分支外的Trainer及VNet代码经AST比较相同；无需重跑未改动的C训练路径。', '',
        '另外保留GPU诊断启停比较的8次真实更新；两项算子/同图隔离探针均为零optimizer更新。这些均为有限检查。', '',
        '验证/最终测试仍使用原 LA/test.list，未改成独立划分。原 A/B 的18个关键产物 SHA256 保持不变。',
        '新三组完整30000次训练均未启动。正式命令仅输出供后续独立开展，不自动执行。', '',
        '[机器可读完整报告](artifacts/validation_summary.json)；[梯度诊断](GRADIENT_DIAGNOSTICS.md)；',
        '[固定真实标签比较](artifacts/label_audit_seed42/label_audit_report.md)。', '']
    (HERE / 'VALIDATION.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status=report['status'], tests_run=report['tests_run'], real_smoke=list(report['smoke']),
                         gpu_same_graph_nonmutation=True,
                         strict_gpu_cross_run_invariance=report['gpu_diagnostic_invariance']['passed'],
                         formal_training_started=False, legacy_files_unchanged=len(protection['checks'])), indent=2))


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
