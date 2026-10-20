"""Observational gradient diagnostics; never write .grad or rerun a forward.

The raw scalar boundary gradient and the actual routed local contribution are
different quantities. Both are retained explicitly for local-mode analysis.
"""
import math

import torch

from utils.boundary_supervision import local_boundary_gradients


def _derivatives(loss, parameters):
    parameters = tuple(parameters)
    active = [p for p in parameters if p.requires_grad]
    if not active or not loss.requires_grad:
        return tuple(None for _ in parameters)
    values = torch.autograd.grad(loss, active, retain_graph=True, allow_unused=True)
    mapping = {id(p): g for p, g in zip(active, values)}
    return tuple(mapping.get(id(p)) for p in parameters)


def gradient_pair_statistics(main_gradients, boundary_gradients,
                             cosine_threshold=-0.2, ratio_threshold=0.1):
    """Norms and cosine on a stage, treating unused parameters as zero vectors.

    Undefined ratios/cosines are None, with an explicit status. Nonfinite values
    are reported as None rather than emitted as invalid JSON NaN or Infinity.
    Double-precision scalar reductions avoid overflowing ordinary FP32 norms.
    """
    if not -1 <= cosine_threshold <= 0 or not math.isfinite(cosine_threshold):
        raise ValueError('cosine_threshold must be finite and in [-1, 0]')
    if ratio_threshold < 0 or not math.isfinite(ratio_threshold):
        raise ValueError('ratio_threshold must be finite and nonnegative')
    if len(main_gradients) != len(boundary_gradients):
        raise ValueError('Gradient lists must use the same stage parameter order')
    main_present = any(g is not None for g in main_gradients)
    boundary_present = any(g is not None for g in boundary_gradients)
    main_finite = all(bool(torch.isfinite(g).all()) for g in main_gradients if g is not None)
    boundary_finite = all(bool(torch.isfinite(g).all()) for g in boundary_gradients if g is not None)
    status = []
    if not main_present:
        status.append('missing_main_gradient')
    if not boundary_present:
        status.append('missing_boundary_gradient')
    if not main_finite:
        status.append('nonfinite_main_gradient')
    if not boundary_finite:
        status.append('nonfinite_boundary_gradient')

    def norm(gradients, finite):
        if not finite:
            return None
        sums = [g.detach().double().square().sum() for g in gradients if g is not None]
        return math.sqrt(sum(s.item() for s in sums)) if sums else 0.0

    main_norm, boundary_norm = norm(main_gradients, main_finite), norm(boundary_gradients, boundary_finite)
    if main_norm == 0 and main_present:
        status.append('zero_main_norm')
    if boundary_norm == 0 and boundary_present:
        status.append('zero_boundary_norm')
    ratio = (boundary_norm / main_norm if main_norm and boundary_norm is not None else None)
    cosine = None
    if main_norm and boundary_norm:
        products = [(a.detach().double() * b.detach().double()).sum().item()
                    for a, b in zip(main_gradients, boundary_gradients)
                    if a is not None and b is not None]
        cosine = max(-1.0, min(1.0, sum(products) / (main_norm * boundary_norm)))
    return dict(main_norm=main_norm, boundary_norm=boundary_norm, ratio=ratio,
                cosine=cosine, opposite=bool(cosine is not None and cosine < 0),
                reverse=bool(cosine is not None and ratio is not None and
                             cosine <= cosine_threshold and ratio >= ratio_threshold),
                status=';'.join(status) if status else 'ok',
                main_parameters_with_gradient=sum(g is not None for g in main_gradients),
                boundary_parameters_with_gradient=sum(g is not None for g in boundary_gradients))


def boundary_gradient_diagnostics(main_loss, result, stage_parameters, heads,
                                  boundary_weight, mode, cosine_threshold=-0.2,
                                  ratio_threshold=0.1, local_gradients=None):
    """Observe all five stages, including E5's missing boundary contribution.

    Local mode's boundary_norm/cosine describe the routed gradients actually
    added by the optimizer path. raw_* describes d(weight*mean(Li))/d(Ej).
    autograd.grad retains the graph and never populates any parameter's .grad.
    """
    parameters = tuple(p for stage in stage_parameters for p in stage)
    main = _derivatives(main_loss, parameters)
    raw = _derivatives(boundary_weight * result.total_loss, parameters)
    main_map = {id(p): g for p, g in zip(parameters, main)}
    raw_map = {id(p): g for p, g in zip(parameters, raw)}
    local = mode in ('local', 'bfa_local')
    if local:
        if local_gradients is None:
            local_gradients = local_boundary_gradients(result, stage_parameters,
                                                       heads, boundary_weight)
        applied_map = {id(p): g for p, g in local_gradients}
    else:
        applied_map = raw_map
    rows = []
    for index, stage in enumerate(stage_parameters, 1):
        main_stage = tuple(main_map[id(p)] for p in stage)
        applied = tuple(applied_map.get(id(p)) for p in stage)
        raw_stage = tuple(raw_map[id(p)] for p in stage)
        stats = gradient_pair_statistics(main_stage, applied, cosine_threshold, ratio_threshold)
        raw_stats = gradient_pair_statistics(main_stage, raw_stage, cosine_threshold, ratio_threshold)
        rows.append(dict(stage=index, **stats,
            raw_boundary_norm=raw_stats['boundary_norm'], raw_ratio=raw_stats['ratio'],
            raw_cosine=raw_stats['cosine'], raw_status=raw_stats['status'],
            boundary_contribution='routed' if local else 'ordinary',
            boundary_weight=boundary_weight, cosine_threshold=cosine_threshold,
            ratio_threshold=ratio_threshold))
    return rows
