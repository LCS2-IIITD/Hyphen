"""
CoNLL-2011/2012 scores for coreference detection.


## References

- **Scoring Coreference Partitions of Predicted Mentions: A Reference Implementation.** Sameer
Pradhan, Xiaoqiang Luo, Marta Recasens, Eduard Hovy, Vincent Ng and Michael Strube. *Proceedings
of the 52nd Annual Meeting of the Association for Computational Linguistics*, Baltimore, MD,
June 2014. ([pdf](http://aclweb.org/anthology/P/P14/P14-2006.pdf))
- **BLANC: Implementing the Rand Index for Coreference Evaluation.** Marta Recasens and Eduard
Hovy In: *Natural Language Engineering* 17 (4). Cambridge University Press, 2011.
([pdf](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.300.9229&rep=rep1&type=pdf))
- **An Extension of BLANC to System Mentions.** Xiaoqiang Luo, Sameer Pradhan, Marta Recasens and
Eduard Hovy. *Proceedings of the 52nd Annual Meeting of the Association for Computational
Linguistics*, Baltimore, MD, June 2014. ([pdf](http://aclweb.org/anthology/P/P14/P14-2005.pdf))
The reference implementation : <https://github.com/conll/reference-coreference-scorers>
"""
import math
import typing as ty

from statistics import mean, harmonic_mean

import numpy as np

from scipy.optimize import linear_sum_assignment


def trace(cluster: ty.Set, partition: ty.Iterable[ty.Set]) -> ty.Iterable[ty.Set]:
    r"""
    Return the partition of `#cluster` induced by `#partition`, that is
    ```math
    \{C∩A|A∈P\} ∪ \{\{x\}|x∈C∖∪P\}
    ```
    Where `$C$` is `#cluster` and `$P$` is `#partition`.

    This assume that the elements of `#partition` are indeed pairwise disjoint.
    """
    remaining = set(cluster)
    for a in partition:
        common = remaining.intersection(a)
        if common:
            remaining.difference_update(common)
            yield common
    for x in sorted(remaining):
        yield set((x,))


class RemapClusteringsReturn(ty.NamedTuple):
    clusterings: ty.Sequence[ty.Sequence[ty.Sequence[int]]]
    elts_map: ty.Dict[ty.Hashable, int]


def remap_clusterings(
    clusterings: ty.Sequence[ty.Sequence[ty.Set[ty.Hashable]]],
) -> RemapClusteringsReturn:
    """Remap clusterings of arbitrary elements to clusterings of integers."""
    elts = set(e for clusters in clusterings for c in clusters for e in c)
    elts_map = {e: i for i, e in enumerate(elts)}
    res = []
    for clusters in clusterings:
        remapped_clusters = []
        for c in clusters:
            remapped_c = [elts_map[e] for e in c]
            remapped_clusters.append(remapped_c)
        res.append(remapped_clusters)
    return RemapClusteringsReturn(res, elts_map)


def muc(
    key: ty.Sequence[ty.Set], response: ty.Sequence[ty.Set]
) -> ty.Tuple[float, float, float]:
    r"""
    Compute the MUC `$(R, P, F₁)$` scores for a `#response` clustering given a `#key` clustering,
    that is
    ```math
    R &= \frac{∑_{k∈K}(\#k-\#p(k, R))}{∑_{k∈K}(\#k-1)}\\
    P &= \frac{∑_{r∈R}(\#r-\#p(r, K))}{∑_{r∈R}(\#r-1)}\\
    F &= 2*\frac{PR}{P+R}
    ```
    with `$p(x, E)=\{x∩A|A∈E\}$`.

    In the edge case where all clusters in either `#key` or `#response` are singletons, `$P$`, `$R$`
    and `$F$` are defined to be `$0$`, following the reference implementation (since singleton
    clusters where not considered in Vilain et al. (1995).

    Note: This implementation is significantly different from the reference one (despite
    implementing the formulae from Pradahan et al. (2014) in that the reference use the ordering of
    mentions in documents to consistently assign a non-problematic spanning tree (viz. a chain) to
    each cluster, thus avoiding the issues that led Vilain et al. (1995) to define MUC by the
    formulae above.
    """
    # Edge case
    if all(len(k) == 1 for k in key) or all(len(r) == 1 for r in response):
        return 0.0, 0.0, 0.0
    R = sum(len(k) - sum(1 for _ in trace(k, response)) for k in key) / sum(
        len(k) - 1 for k in key
    )
    P = sum(len(r) - sum(1 for _ in trace(r, key)) for r in response) / sum(
        len(r) - 1 for r in response
    )
    F = harmonic_mean((R, P))
    return R, P, F


def b_cubed(
    key: ty.Sequence[ty.Set], response: ty.Sequence[ty.Set]
) -> ty.Tuple[float, float, float]:
    r"""
    Compute the B³ `$(R, P, F₁)$` scores for a `#response` clustering given a `#key` clustering,
    that is
    ```math
    R &= \frac{∑_{k∈K}∑_{r∈R}\frac{(\#k∩r)²}{\#k}}{∑_{k∈K}\#k}\\
    P &= \frac{∑_{r∈R}∑_{k∈K}\frac{(\#r∩k)²}{\#r}}{∑_{r∈R}\#r}\\
    F &= 2*\frac{PR}{P+R}
    ```
    """
    if sum(len(k) for k in key) == 0:
        R = 0.0
    else:
        R = math.fsum(
            len(k.intersection(r)) ** 2 / len(k) for k in key for r in response
        ) / sum(len(k) for k in key)
    if sum(len(r) for r in response) == 0:
        P = 0.0
    else:
        P = math.fsum(
            len(r.intersection(k)) ** 2 / len(r) for r in response for k in key
        ) / sum(len(r) for r in response)
    F = harmonic_mean((R, P))
    return R, P, F


def ceaf(
    key: ty.Sequence[ty.Set],
    response: ty.Sequence[ty.Set],
    score: ty.Callable[[ty.Set, ty.Set], float],
) -> ty.Tuple[float, float, float]:
    r"""
    Compute the CEAF `$(R, P, F₁)$` scores for a `#response` clustering given a `#key` clustering
    using the `#score` alignment score function, that is
    ```math
    R &= \frac{∑_{k∈K}C(k, A(k))}{∑_{k∈K}C(k, k)}\\
    P &= \frac{∑_{r∈R}C(r, A⁻¹(r))}{∑_{r∈R}C(r, r)}\\
    F &= 2*\frac{PR}{P+R}
    ```
    Where `$C$` is `#score` and `$A$` is a one-to-one mapping from key clusters to response
    clusters that maximizes `$∑_{k∈K}C(k, A(k))$`.
    """
    if len(response) == 0 or len(key) == 0:
        return 0.0, 0.0, 0.0
    else:
        cost_matrix = np.array([[-score(k, r) for r in response] for k in key])
        # TODO: See https://github.com/allenai/allennlp/issues/2946 for ideas on speeding
        # the next line up
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        total_score = -cost_matrix[row_ind, col_ind].sum()
        R = total_score / math.fsum(score(k, k) for k in key)
        P = total_score / math.fsum(score(r, r) for r in response)
        F = harmonic_mean((R, P))
        return R, P, F


def ceaf_m(
    key: ty.Sequence[ty.Set], response: ty.Sequence[ty.Set]
) -> ty.Tuple[float, float, float]:
    r"""
    Compute the CEAFₘ `$(R, P, F₁)$` scores for a `#response` clustering given a `#key` clustering,
    that is the CEAF score for the `$Φ_3$` score function
    ```math
    Φ_3: (k, r) ⟼ \#k∩r
    ```
    """

    def Φ_3(k, r):
        return len(k.intersection(r))

    return ceaf(key, response, Φ_3)


def ceaf_e(
    key: ty.Sequence[ty.Set], response: ty.Sequence[ty.Set]
) -> ty.Tuple[float, float, float]:
    r"""
    Compute the CEAFₑ `$(R, P, F₁)$` scores for a `#response` clustering given a `#key`
    clustering, that is the CEAF score for the `$Φ₄$` score function (aka the Sørensen–Dice
    coefficient).
    ```math
    Φ₄: (k, r) ⟼ \frac{2×\#k∩r}{\#k+\#r}
    ```
    Note: this use the original (Luo, 2005) definition as opposed to Pradhan et al. (2014)'s one
    which inlines the denominators.
    """

    def Φ_4(k, r):
        return 2 * len(k.intersection(r)) / (len(k) + len(r))

    return ceaf(key, response, Φ_4)


# COMBAK: Check the numeric stability
def blanc(
    key: ty.Sequence[ty.Set], response: ty.Sequence[ty.Set], fast=True,
) -> ty.Tuple[float, float, float]:
    r"""
    Return the BLANC `$(R, P, F)$` scores for a `#response` clustering given a `#key` clustering.

    ## Notes

      - Mention identifiers have to be comparable
      - To ensure the compliance with the reference implementation, the edge cases results are
        those from Recasens and Hovy (2011) rather than from the more recent Luo et al. (2014) when
        those two disagree. This has an effect for the N-6 testcase, where according to Luo et al.
        (2014), BLANC should be `$\frac{0+F_n}{2}$` since `$C_k=∅$` and `$C_r≠∅$`, but according to
        Recasens and Hovy (2011), BLANC should be `$F_n$`.
    """
    if fast:
        C_score, N_score = fast_detailed_blanc(key, response)
    else:
        C_score, N_score = detailed_blanc(key, response)
    if C_score is None:
        assert N_score is not None  # nosec:B101
        return N_score
    if N_score is None:
        assert C_score is not None  # nosec:B101
        return C_score
    return ty.cast(
        ty.Tuple[float, float, float],
        tuple(np.mean((C_score, N_score), axis=0).tolist()),
    )


def links_from_clusters(
    clusters: ty.Iterable[ty.Set],
) -> ty.Tuple[
    ty.Set[ty.Tuple[ty.Hashable, ty.Hashable]],
    ty.Set[ty.Tuple[ty.Hashable, ty.Hashable]],
]:
    r"""
    Return a `(coreference_links, non-coreference_links)` tuple corresponding to a clustering.

    The links are given as sorted couples for uniqueness
    """
    clusters_lst = [list(c) for c in clusters]
    C = set()
    N = set()
    for i, c in enumerate(clusters_lst[:-1]):
        for j, e in enumerate(c[:-1]):
            # Since the links are symmetric, we only add the links between `e` and
            # the following mentions
            for f in c[j + 1 :]:
                C.add((e, f) if e <= f else (f, e))
        for other in clusters_lst[i + 1 :]:
            for e in c:
                for f in other:
                    N.add((e, f) if e <= f else (f, e))
    #  We missed the coreference links for the last cluster, add them here
    last_cluster = clusters_lst[-1]
    for j, e in enumerate(last_cluster):
        for f in last_cluster[j + 1 :]:
            C.add((e, f) if e <= f else (f, e))
    return C, N


def detailed_blanc(
    key: ty.Sequence[ty.Set], response: ty.Sequence[ty.Set]
) -> ty.Tuple[
    ty.Union[ty.Tuple[float, float, float], None],
    ty.Union[ty.Tuple[float, float, float], None],
]:
    """Return BLANC `$(R, P, F)$` scores for coreference and non-coreference respectively."""

    # Edge case : a single mention in both `key` and `response` clusters
    # in that case, `C_k`, `C_r`, `N_k` and `N_r` are all empty, so we need a separate examination
    # of the mentions to know if we are very good or very bad.
    if len(key) == len(response) == 1 and len(key[0]) == len(response[0]) == 1:
        if key[0] == response[0]:
            return ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
        else:
            return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    C_k, N_k = links_from_clusters(key)
    C_r, N_r = links_from_clusters(response)

    tp_c = len(C_k.intersection(C_r))
    tp_n = len(N_k.intersection(N_r))
    c_k, n_k = len(C_k), len(N_k)
    c_r, n_r = len(C_r), len(N_r)

    if not c_k and not c_r:
        R_c, P_c, F_c = (1.0, 1.0, 1.0)
    elif not c_k or not c_r:
        R_c, P_c, F_c = (0.0, 0.0, 0.0)
    else:
        R_c, P_c = tp_c / c_k, tp_c / c_r
        F_c = 2 * tp_c / (c_k + c_r)

    if not n_k and not n_r:
        R_n, P_n, F_n = (1.0, 1.0, 1.0)
    elif not n_k or not n_r:
        R_n, P_n, F_n = (0.0, 0.0, 0.0)
    else:
        R_n, P_n = tp_n / n_k, tp_n / n_r
        F_n = 2 * tp_n / (n_k + n_r)

    # Edge cases
    if not c_k:
        return (None, (R_n, P_n, F_n))
    if not n_k:
        return ((R_c, P_c, F_c), None)

    return ((R_c, P_c, F_c), (R_n, P_n, F_n))


class AdjacencyReturn(ty.NamedTuple):
    """Represents a clustering of integers as an adjacency matrix and a presence mask"""

    adjacency: np.ndarray
    presence: np.ndarray


def adjacency(clusters: ty.List[ty.List[int]], num_elts: int) -> AdjacencyReturn:
    adjacency = np.zeros((num_elts, num_elts), dtype=np.bool)
    presence = np.zeros(num_elts, dtype=np.bool)
    # **Note** The nested loop makes the complexity of this `$∑|c|²$` but we are only doing memory
    # access, which is really fast, so this is not really an issue. In comparison, doing it by
    # computing the Gram matrix one-hot elt-cluster attribution matrix was making `fast_blanc` 3×
    # slower than the naïve version.
    for c in clusters:
        # Note: don't be clever and use numpy array indicing here, see
        # <https://docs.scipy.org/doc/numpy/user/basics.indexing.html?highlight=slice#assigning-values-to-indexed-arrays>
        # for why it would be slower. If you want to get C loops here, cythonize it instead (nut
        # it's probably not worth it)
        for e in c:
            presence[e] = True
            for f in c:
                if f != e:
                    adjacency[e, f] = True
    return AdjacencyReturn(adjacency, presence)


def fast_detailed_blanc(
    key: ty.Sequence[ty.Set], response: ty.Sequence[ty.Set]
) -> ty.Tuple[
    ty.Union[ty.Tuple[float, float, float], None],
    ty.Union[ty.Tuple[float, float, float], None],
]:
    """Return BLANC `$(R, P, F)$` scores for coreference and non-coreference respectively."""

    # Edge case : a single mention in both `key` and `response` clusters
    # in that case, `C_k`, `C_r`, `N_k` and `N_r` are all empty, so we need a separate examination
    # of the mentions to know if we are very good or very bad.
    if len(key) == len(response) == 1 and len(key[0]) == len(response[0]) == 1:
        if key[0] == response[0]:
            return ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
        else:
            return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    (key, response), mentions_map = remap_clusterings([key, response])
    num_mentions = len(mentions_map)

    key_coref_links, key_presence = adjacency(key, num_mentions)
    response_coref_links, response_presence = adjacency(response, num_mentions)

    tp_c = np.logical_and(key_coref_links, response_coref_links).sum() // 2
    c_k = key_coref_links.sum() // 2
    c_r = response_coref_links.sum() // 2

    # Headache ahead
    common_links = np.logical_and(
        np.outer(key_presence, key_presence),
        np.outer(response_presence, response_presence),
    )
    # There is no link between a mention and itself
    np.fill_diagonal(common_links, False)
    tp_n = (
        np.logical_and(
            common_links,
            np.logical_not(np.logical_or(key_coref_links, response_coref_links)),
        ).sum()
        / 2
    )
    num_key_mentions = key_presence.sum()
    n_k = (num_key_mentions * (num_key_mentions - 1)) // 2 - c_k
    num_response_mentions = response_presence.sum()
    n_r = (num_response_mentions * (num_response_mentions - 1)) // 2 - c_r

    if not c_k and not c_r:
        R_c, P_c, F_c = (1.0, 1.0, 1.0)
    elif not c_k or not c_r:
        R_c, P_c, F_c = (0.0, 0.0, 0.0)
    else:
        R_c, P_c = tp_c / c_k, tp_c / c_r
        F_c = 2 * tp_c / (c_k + c_r)

    if not n_k and not n_r:
        R_n, P_n, F_n = (1.0, 1.0, 1.0)
    elif not n_k or not n_r:
        R_n, P_n, F_n = (0.0, 0.0, 0.0)
    else:
        R_n, P_n = tp_n / n_k, tp_n / n_r
        F_n = 2 * tp_n / (n_k + n_r)

    # Edge cases
    if not c_k:
        return (None, (R_n, P_n, F_n))
    if not n_k:
        return ((R_c, P_c, F_c), None)

    return ((R_c, P_c, F_c), (R_n, P_n, F_n))


def conll2012(key: ty.Sequence[ty.Set], response: ty.Sequence[ty.Set]) -> float:
    r"""
    Return the CoNLL-2012 scores for a `#response` clustering given a `#key` clustering, that is,
    the average of the MUC, B³ and CEAFₑ scores.
    """
    return mean((metric(key, response)[2] for metric in (muc, b_cubed, ceaf_e)))
