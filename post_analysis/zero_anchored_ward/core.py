"""Zero-anchored geometry and Euclidean Ward clustering, independent of legacy analyses."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)

FEATURES = ('sensory', 'action_history_total', 'wsls_history_total', 'dynamic_context_total')
LABELS = ('Sensory', 'Action history', 'WSLS history', 'Dynamic context')
FORMULAS = ('L(const) - L(S)', 'L(S) - L(S+AH)', 'L(S) - L(S+WH)', 'L(S) - L(S+Bdynamic)')
MODELS = ('m0_sensory', 'm1_action_kernel', 'm1_stimulus_kernel', 'm2_dynamic')
EPSILON = 1e-12
SEED = 20261104
KEYS = ['subject', 'session_id']


@dataclass
class Geometry:
    positive: np.ndarray
    scale: np.ndarray
    scaled: np.ndarray
    norm: np.ndarray
    z: np.ndarray
    eligible: np.ndarray


def normalize(gains, epsilon=EPSILON):
    """Global sample SD of clamped gains; no centering, then epsilon-regularized row L2.

    All-zero columns retain zero (denominator=1). Nonzero constant columns cannot
    be standardized and fail explicitly. All-zero rows have no direction and are
    excluded from Ward, retaining label zero in exports. No low-norm cutoff is used.
    """
    x = np.asarray(gains, dtype=float)
    if x.ndim != 2 or x.shape[1] != 4 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('Expected at least two rows of four finite signed gains')
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('epsilon must be finite and positive')
    positive = np.maximum(x, 0.)
    scale = positive.std(axis=0, ddof=1)
    # Repeated decimal constants can have a tiny nonzero floating-point SD.
    constant = np.ptp(positive, axis=0) == 0
    if np.any(constant & np.any(positive != 0, axis=0)):
        raise ValueError('Nonzero constant coordinate: specify a scientifically justified scale')
    scale[constant] = 1.
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Nonfinite or numerically underflowed feature scale')
    u = positive / scale
    norm = np.linalg.norm(u, axis=1)
    z = u / np.hypot(norm, epsilon)[:, None]
    return Geometry(positive, scale, u, norm, z, norm > 0)


def number_labels(labels):
    """Decreasing size, first input member breaks ties; never aligned to old labels."""
    groups = sorted(np.unique(labels), key=lambda c: (-np.sum(labels == c), np.flatnonzero(labels == c)[0]))
    mapping = {c: i+1 for i, c in enumerate(groups)}
    return np.array([mapping[c] for c in labels], dtype=int)


def ward_tree(z):
    x = np.asarray(z, float)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('Ward requires at least two finite coordinate vectors')
    return linkage(x, method='ward', metric='euclidean', optimal_ordering=True)


def partition(tree, k):
    n = len(tree)+1
    if not 2 <= k < n:
        raise ValueError('K must be at least 2 and smaller than sample size')
    # cut_tree gives exactly K groups, even when merge heights tie.
    return number_labels(cut_tree(tree, n_clusters=[k]).ravel())


def clustering(gains, subjects, replicates=100, k_values=range(2, 9)):
    """Descriptive K selection. Subjects are audited, never forced into each group."""
    g = normalize(gains)
    ix = np.flatnonzero(g.eligible)
    x, sub = np.asarray(gains)[ix], np.asarray(subjects)[ix]
    z = g.z[ix]
    if len(z) < 5:
        raise ValueError('At least five nonzero sessions required for K2/K3/K4 reports')
    tree = ward_tree(z)
    ks = [k for k in k_values if k < len(z) and k <= np.unique(z, axis=0).shape[0]]
    if not ks:
        raise ValueError('No candidate K with enough distinct nonzero profiles')
    labels = {k: partition(tree, k) for k in ks}
    rng = np.random.default_rng(SEED)
    stability = {k: [] for k in ks}
    # Refit SD per subset; keep complete-pipeline perturbation distinct from gain uncertainty.
    draws = []
    for b in range(replicates):
        take = np.sort(rng.choice(len(x), max(3, int(.8*len(x))), replace=False))
        try:
            geom = normalize(x[take])
            take = take[geom.eligible]
            zs = geom.z[geom.eligible]
            stree = ward_tree(zs)
        except ValueError:
            continue
        for k in ks:
            if k >= len(take) or k > len(np.unique(zs, axis=0)):
                continue
            score = adjusted_rand_score(labels[k][take], partition(stree, k))
            stability[k].append(score)
            draws.append(dict(replicate=b, K=k, ari=score, n_sessions=len(take)))
    rows, loo = [], []
    for k, lab in labels.items():
        counts = pd.crosstab(pd.Series(sub, name='subject'), pd.Series(lab, name='cluster')).reindex(
            index=np.unique(sub), columns=range(1, k+1), fill_value=0)
        a = np.asarray(stability[k])
        rows.append(dict(K=k, silhouette=silhouette_score(z, lab),
                         calinski_harabasz=calinski_harabasz_score(z, lab),
                         davies_bouldin=davies_bouldin_score(z, lab),
                         stability_ari_median=np.median(a) if len(a) else np.nan,
                         stability_ari_q10=np.quantile(a, .1) if len(a) else np.nan,
                         stability_ari_q90=np.quantile(a, .9) if len(a) else np.nan,
                         stability_valid_replicates=len(a),
                         min_mouse_cluster_count=int(counts.to_numpy().min()),
                         all_mice_in_every_cluster=bool(counts.to_numpy().min() > 0),
                         sizes='/'.join(str(np.sum(lab == c)) for c in range(1, k+1))))
        for mouse in np.unique(sub):
            take = np.flatnonzero(sub != mouse)
            try:
                gs = normalize(x[take])
                keep = take[gs.eligible]
                zl = gs.z[gs.eligible]
                if k >= len(zl) or k > len(np.unique(zl, axis=0)):
                    raise ValueError('Insufficient distinct retained profiles')
                ar = adjusted_rand_score(lab[keep], partition(ward_tree(zl), k))
                loo.append(dict(K=k, omitted_mouse=mouse, ari=ar, n_sessions=len(keep), status='ok'))
            except ValueError as exc:
                loo.append(dict(K=k, omitted_mouse=mouse, ari=np.nan, n_sessions=len(take), status=str(exc)))
    table = pd.DataFrame(rows).sort_values('K').reset_index(drop=True)
    optimal = int(table.loc[table.silhouette.idxmax(), 'K'])
    table['selected_descriptive'] = table.K.eq(optimal)
    full = {}
    for k, lab in labels.items():
        full[k] = np.zeros(len(gains), dtype=int)
        full[k][ix] = lab
    return g, tree, full, table, pd.DataFrame(draws), pd.DataFrame(loo), optimal


def distance_matrix(z):
    return squareform(pdist(z, metric='euclidean'))
