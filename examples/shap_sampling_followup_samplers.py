"""Experiment-only AIS ablations and reversible rank tests.

Production si_shap inference is deliberately unchanged. All tuning is fixed in
the manifest. Serial ranks use a uniform observation position, not a chain tail
frequency from a chain initialized at the observation.
"""
from __future__ import annotations

import json
import numpy as np
from scipy import stats
from scipy.special import logsumexp
from si_shap.inference import (
    _component_mixture_logpdf, _deterministic_component_schedule,
    _effective_sample_size, _refinement_components, _sample_component_labels,
    _weighted_location_scale, _weight_gini,
)


def preserved_islands(pilot, rank, max_components=64):
    """Keep runs separated by >=3 rejected points; cap by raw-radius gaps.

Unlike reference consolidation, no absolute chi-probability gap is required
and overlapping proposal cores cannot erase these observed separations. The
64-component cap can still merge groups; this is recorded, not certification.
"""
    z, states, lw = (np.asarray(pilot[k]) for k in ('z_values', 'selected', 'log_weights'))
    order = np.argsort(z, kind='stable')
    z, states, lw = z[order], states[order], lw[order]
    indices = np.flatnonzero(states & np.isfinite(lw))
    if not len(indices):
        return (), 0
    cuts = [i for i in range(1, len(indices))
            if np.sum(~states[indices[i-1]+1:indices[i]]) >= 3]
    groups = list(np.split(indices, cuts))
    before = len(groups)
    while len(groups) > max_components:
        join = int(np.argmin([z[b[0]]-z[a[-1]] for a, b in zip(groups, groups[1:])]))
        groups[join] = np.concatenate((groups[join], groups.pop(join+1)))
    masses = np.array([logsumexp(lw[g]) for g in groups])
    masses = .8*np.exp(masses-logsumexp(masses)) + .2/len(groups)
    islands = []
    for g, mass in zip(groups, masses):
        mean, sd = _weighted_location_scale(z[g], lw[g], sd_floor=.05)
        lo = np.flatnonzero(~states[:g[0]])
        hi = np.flatnonzero(~states[g[-1]+1:])
        lower = (z[lo[-1]]+z[g[0]])/2 if len(lo) else max(0., z[g[0]]-sd)
        upper = (z[g[-1]]+z[g[-1]+1+hi[0]])/2 if len(hi) else z[g[-1]]+sd
        islands.append(dict(mean=float(mean), sd=float(sd), mass=float(mass),
                            lower_z=float(lower), upper_z=float(upper),
                            lower_u=float(stats.chi.cdf(lower, rank)),
                            upper_u=float(stats.chi.cdf(upper, rank))))
    return tuple(islands), before


def fit_components(t_obs, rank, pilot, variant, allocation_cap, draw_floor=128):
    if variant in ('reference', 'stable'):
        components, _, _, islands, before = _refinement_components(
            t_obs, rank, pilot, interval=False, denominator=False,
            draw_floor=draw_floor, max_final_samples=allocation_cap,
            tail_stable=variant == 'stable')
        return components, islands, before
    if variant != 'preserve':
        raise ValueError(f'Unknown AIS variant: {variant}')
    islands, before = preserved_islands(pilot, rank)
    if not islands:
        # Match the reference fallback without inventing a selected interval.
        return fit_components(t_obs, rank, pilot, 'stable', allocation_cap, draw_floor)
    components = [('chi', 0., 1., .25),
                  ('tn', float(t_obs), max(.75, .25*t_obs+.5), .375)]
    components += [('tn', i['mean'], i['sd'], .375*i['mass']) for i in islands]
    floor = draw_floor/allocation_cap
    deficit = sum(max(0., floor-c[3]) for c in components[2:])
    if deficit > .325:
        raise ValueError('Component floor exceeds available observed-component mass')
    components[1] = (*components[1][:3], .375-deficit)
    components[2:] = [(*c[:3], max(c[3], floor)) for c in components[2:]]
    return tuple(components), islands, before


def run_ais(t_obs, rank, oracle, rng, pilot, arm, settings):
    allocation_cap = settings['allocation_cap']
    components, islands, before = fit_components(
        t_obs, rank, pilot, arm['variant'], allocation_cap, settings['draw_floor'])
    schedule = _deterministic_component_schedule(rng, allocation_cap, components)
    allocation = np.bincount(schedule, minlength=len(components))/allocation_cap
    components = tuple((*c[:3], float(w)) for c, w in zip(components, allocation))
    values, logs, selected_labels, checkpoints = [], [], [], []
    threshold_d, threshold_t = settings['min_denominator_ess'], settings['min_tail_ess']
    count = 0
    first_met = None
    while count < arm['final_cap']:
        size = min(settings['batch_size'], arm['final_cap']-count)
        labels = schedule[count:count+size]
        z = _sample_component_labels(rng, labels, rank, components)
        selected = np.fromiter((oracle(float(v)) for v in z), bool, size)
        values.extend(z[selected])
        selected_labels.extend(labels[selected])
        logs.extend(stats.chi.logpdf(z[selected], rank)
                    - _component_mixture_logpdf(z[selected], rank, components))
        count += size
        zz, ll = np.asarray(values), np.asarray(logs)
        finite = np.isfinite(ll)
        zz, ll = zz[finite], ll[finite]
        weights = np.exp(ll-np.max(ll)) if len(ll) else np.array([])
        tail = zz >= t_obs
        de, te = _effective_sample_size(weights), _effective_sample_size(weights[tail])
        estimate = float(np.sum(weights*tail)/np.sum(weights)) if len(weights) else np.nan
        met = de >= threshold_d and te >= threshold_t
        if met and first_met is None:
            first_met = count
        checkpoints.append([count, estimate, de, te, float(met)])
        if met and arm['early_stop']:
            break
    resolved = de >= threshold_d and te >= threshold_t
    normalized = weights/weights.sum() if len(weights) else weights
    se = float(np.sqrt(np.sum(weights**2*(tail-estimate)**2)/weights.sum()**2)) if len(weights) else np.nan
    diag = dict(p_value=estimate if resolved else np.nan, estimate_p_value=estimate,
                status='ok' if resolved else ('insufficient_ess' if len(weights) else 'no_selected_samples'),
                resolution_status='resolved' if resolved else 'unresolved',
                proposals=count, selected_samples=len(zz), tail_samples=int(tail.sum()),
                denominator_ess=de, tail_ess=te, mc_se=se,
                first_ess_met_draws=first_met, finite_sample_valid=False,
                p_value_method='self_normalized_importance_sampling_estimate',
                max_normalized_weight=float(normalized.max()) if len(weights) else np.nan,
                weight_gini=_weight_gini(weights), pre_merge_cluster_count=before,
                post_merge_cluster_count=len(islands),
                component_parameters_json=json.dumps(components),
                island_parameters_json=json.dumps(islands))
    trace = dict(selected_z=zz, selected_log_weights=ll,
                 selected_labels=np.asarray(selected_labels, dtype=np.int16)[finite],
                 checkpoints=np.asarray(checkpoints), schedule=schedule[:count])
    return diag, trace


def transition(initial, rank, oracle, rng, steps, scales, jump, trace=False):
    """Same fixed reversible kernel and RNG order as si_shap._mcmc_transition."""
    current = float(initial)
    density = float(stats.chi.logpdf(current, rank))
    counters = np.zeros(4, dtype=int)  # local attempts/accepts, global attempts/accepts
    path = []
    for _ in range(steps):
        if rng.random() < jump:
            counters[2] += 1
            proposal = float(stats.chi.rvs(rank, random_state=rng))
            if oracle(proposal):
                current = proposal
                density = float(stats.chi.logpdf(current, rank))
                counters[3] += 1
        else:
            counters[0] += 1
            scale = float(scales[rng.integers(len(scales))])
            proposal = abs(float(current+rng.normal(scale=scale)))
            if oracle(proposal):
                proposed_density = float(stats.chi.logpdf(proposal, rank))
                if np.log(rng.random()) <= min(0., proposed_density-density):
                    current, density = proposal, proposed_density
                    counters[1] += 1
        if trace:
            path.append(current)
    return current, counters, path


def run_rank(t_obs, rank, oracle, rng, arm, settings, trace=False):
    replicas, steps = arm['replicas'], arm['steps']
    if replicas < 1 or steps < 1 or not oracle(t_obs):
        raise ValueError('Positive replica/step counts and a selected observation are required')
    scales = np.asarray(settings['step_scales'], dtype=float)
    jump = settings['global_jump_probability']
    if not np.all(scales > 0) or not 0 <= jump <= 1:
        raise ValueError('Invalid fixed MCMC kernel')
    kwargs = dict(rank=rank, oracle=oracle, rng=rng, steps=steps,
                  scales=scales, jump=jump, trace=trace)
    counters = np.zeros(4, dtype=int)
    paths = []
    if arm['construction'] == 'parallel':
        hub, counts, hub_path = transition(t_obs, **kwargs)
        counters += counts
        endpoints = np.empty(replicas)
        for i in range(replicas):
            endpoints[i], counts, path = transition(hub, **kwargs)
            counters += counts
            paths.extend(path)
        p = float((1+np.sum(endpoints >= t_obs))/(replicas+1))
        proposals = (replicas+1)*steps
        extra = dict(mcmc_hub=hub, mcmc_observation_position=None,
                     mcmc_moved_replicas=int(np.sum(endpoints != hub)),
                     mc_se_conditional_hub=float(np.sqrt(
                         np.mean(endpoints >= t_obs)*(1-np.mean(endpoints >= t_obs))/replicas)))
    elif arm['construction'] == 'serial':
        # Uniform insertion is essential. K is its own reversal. The rank is
        # permutation invariant, so explicit output permutation is unnecessary.
        position = int(rng.integers(replicas+1))
        chain = np.empty(replicas+1)
        chain[position] = t_obs
        for i in range(position-1, -1, -1):
            chain[i], counts, path = transition(chain[i+1], **kwargs)
            counters += counts
            paths.extend(path)
        for i in range(position+1, replicas+1):
            chain[i], counts, path = transition(chain[i-1], **kwargs)
            counters += counts
            paths.extend(path)
        endpoints = np.delete(chain, position)
        p = float(np.sum(chain >= t_obs)/(replicas+1))
        proposals = replicas*steps
        hub_path = []
        extra = dict(mcmc_hub=None, mcmc_observation_position=position,
                     mcmc_moved_replicas=None, mc_se_conditional_hub=None)
    else:
        raise ValueError('Unknown rank construction')
    diag = dict(p_value=p, estimate_p_value=None, status='ok', resolution_status='resolved',
                proposals=proposals, selected_samples=replicas,
                tail_samples=int(np.sum(endpoints >= t_obs)),
                denominator_ess=None, tail_ess=None, mc_se=None,
                finite_sample_valid=True,
                p_value_method=f"{arm['construction']}_reversible_mcmc_rank",
                minimum_attainable_p_value=1/(replicas+1),
                mcmc_unique_replicas=int(len(np.unique(endpoints))),
                mcmc_replica_count=replicas, mcmc_steps=steps,
                mcmc_local_attempts=int(counters[0]), mcmc_local_accepts=int(counters[1]),
                mcmc_global_attempts=int(counters[2]), mcmc_global_accepts=int(counters[3]),
                **extra)
    return diag, dict(endpoints=endpoints, hub_path=np.asarray(hub_path),
                      transition_states=np.asarray(paths))


def log_interval_mass(lower, upper, rank):
    if upper <= lower:
        return -np.inf
    a = float(stats.chi.logsf(lower, rank))
    b = float(stats.chi.logsf(upper, rank))
    if np.isneginf(b):
        return a
    return a+np.log(-np.expm1(b-a)) if b < a else -np.inf


def map_regions(oracle, t_obs, rank, budget=4096, anchors=()):
    """Bounded finite-grid diagnostic. Never certifies undiscovered islands."""
    zmax = max(t_obs+6., float(stats.chi.isf(1e-14, rank)))
    n = max(12, budget//12)
    coarse = np.unique(np.r_[0., t_obs, np.linspace(0., zmax, n),
        stats.chi.ppf((np.arange(n)+.5)/n, rank),
        stats.chi.isf(np.exp(np.linspace(np.log(.5), stats.chi.logsf(zmax, rank), n)), rank)])
    valid_anchors = np.asarray(anchors, dtype=float)
    valid_anchors = valid_anchors[(valid_anchors >= 0)&(valid_anchors <= zmax)]
    # Spread anchor coverage over the ray; reserve the remaining budget for
    # midpoint checks and boundary refinement rather than using every pilot.
    valid_anchors = np.sort(valid_anchors)[::max(1, len(valid_anchors)//n+1)]
    coarse = np.unique(np.r_[coarse, valid_anchors])
    states = {float(z): bool(oracle(float(z))) for z in coarse}
    intervals_by_level = []
    level_counts = []
    for level in range(2):
        level_budget = budget//2 if level == 0 else budget
        grid = np.array(sorted(states))
        if level:
            for z in (grid[:-1]+grid[1:])/2:
                if len(states) >= level_budget:
                    break
                states[float(z)] = bool(oracle(float(z)))
        grid = np.array(sorted(states))
        boundaries = []
        for left, right in zip(grid[:-1], grid[1:]):
            if states[left] == states[right]:
                continue
            lo, hi = left, right
            for _ in range(24):
                if hi-lo <= 1e-6 or len(states) >= level_budget:
                    break
                mid = float((lo+hi)/2)
                states[mid] = bool(oracle(mid))
                if states[mid] == states[left]:
                    lo = mid
                else:
                    hi = mid
            boundaries.append(float((lo+hi)/2))
        edges = [0., *boundaries, zmax]
        selected = states[0.]
        intervals = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            if selected:
                intervals.append((lo, hi))
            selected = not selected
        ld = float(logsumexp([log_interval_mass(a,b,rank) for a,b in intervals]))
        ln = float(logsumexp([log_interval_mass(max(a,t_obs),b,rank)
                             for a,b in intervals if b > t_obs]))
        intervals_by_level.append((intervals, ld, float(np.exp(ln-ld)) if np.isfinite(ld) else np.nan))
        level_counts.append(len(states))
    _, ld0, p0 = intervals_by_level[0]
    intervals, ld, p = intervals_by_level[-1]
    relative_change = float(abs(np.expm1(ld0-ld))) if np.isfinite(ld0) and np.isfinite(ld) else np.inf
    tail_bound = float(np.exp(stats.chi.logsf(zmax,rank)-ld)) if np.isfinite(ld) else np.inf
    stabilized = (level_counts[1] > level_counts[0] and relative_change <= .05
                  and abs(p-p0) <= .002 and tail_bound <= 1e-4)
    return dict(intervals=intervals, log_selection_mass=ld, diagnostic_p=p,
                previous_diagnostic_p=p0, denominator_relative_change=relative_change,
                omitted_tail_relative_bound=tail_bound, z_max=zmax,
                grid_evaluations=len(states), stabilized=bool(stabilized),
                regions_certified=False,
                interpretation='finite_grid_diagnostic_only'), dict(
                    grid_z=np.array(sorted(states)),
                    grid_selected=np.array([states[z] for z in sorted(states)]))
