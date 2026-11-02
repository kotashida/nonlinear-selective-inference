"""Small analytic-region check on all eight unjittered real designs.

Evaluates selection only: it does not run MCMC or create production outputs.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from benchmarks import run_publication_mcmc as e
from si_shap.inference import _spline_exact_set_intervals


def main():
    m = e.read_config()
    records = []
    for c in e.conditions(m):
        if c['study'] != 'real' or c['mode'] != 'gaussian_null':
            continue
        values, ident = e.generated(m, c, 0)
        X, y = values['X'], values['y']
        selector = e.selector(m, 'spline_exact')
        selected = sorted(map(int, selector.select(X, y, 2).selected_features))
        target = e.target_from_selected_set(selected, ident['auxiliary_u'])
        basis = e._spline_effect_basis(X[:, target])
        projected = basis @ (basis.T @ y)
        radius = np.linalg.norm(projected)
        direction = projected / radius
        orthogonal = y - projected
        intervals = _spline_exact_set_intervals(selector._bases(X), orthogonal, direction, 1., selected)
        grid = np.r_[np.linspace(.001, 8., 200), radius]
        for z in grid:
            actual = sorted(map(int, selector.select(X, orthogonal+direction*z, 2).selected_features)) == selected
            analytic = any(lower <= z <= upper for lower, upper in intervals)
            e.require(actual == analytic, f'Analytic spline event mismatch: {c["dataset"]}, radius {z}')
        serial_values = {key: value.tolist() for key, value in values.items()}
        records.append(dict(dataset=c['dataset'], identity=ident, input_values_sha256=e.value_hash(serial_values),
                            target=target, rank=basis.shape[1], checks=len(grid), status='passed'))
    report = dict(checker_sha256=e.digest(Path(__file__)), manifest_sha256=e.value_hash(m),
                  datasets=len(records), total_ray_checks=sum(x['checks'] for x in records),
                  full_benchmarks_executed=False, records=records)
    e.atomic(ROOT/'docs/publication_mcmc_spline_ray_smoke.json', report)
    print(f'{len(records)} real designs; {report["total_ray_checks"]} analytic/selector ray checks passed')


if __name__ == '__main__':
    main()
