import numpy as np


def _compute_distance(features: np.ndarray) -> np.ndarray:
    squared = np.sum(np.square(features), axis=1, keepdims=True)
    dist = squared + squared.T - 2.0 * np.dot(features, features.T)
    return np.maximum(dist, 0.0)


def re_ranking(qf, gf, k1: int = 20, k2: int = 6, lambda_value: float = 0.3):
    all_features = np.vstack([qf, gf]).astype(np.float32)
    num_q = qf.shape[0]
    all_num = all_features.shape[0]

    original_dist = _compute_distance(all_features)
    original_dist = original_dist / np.max(original_dist, axis=0, keepdims=True)
    initial_rank = np.argsort(original_dist, axis=1)

    V = np.zeros_like(original_dist, dtype=np.float32)

    for i in range(all_num):
        forward = initial_rank[i, : k1 + 1]
        backward = initial_rank[forward, : k1 + 1]
        fi = np.where(backward == i)[0]
        k_reciprocal = forward[fi]

        k_reciprocal_expansion = k_reciprocal
        for candidate in k_reciprocal:
            candidate_forward = initial_rank[candidate, : int(np.around(k1 / 2)) + 1]
            candidate_backward = initial_rank[candidate_forward, : int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward == candidate)[0]
            candidate_recip = candidate_forward[fi_candidate]

            if len(np.intersect1d(candidate_recip, k_reciprocal)) > (2.0 / 3) * len(candidate_recip):
                k_reciprocal_expansion = np.append(k_reciprocal_expansion, candidate_recip)

        k_reciprocal_expansion = np.unique(k_reciprocal_expansion)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion])
        V[i, k_reciprocal_expansion] = weight / np.sum(weight)

    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            V_qe[i] = np.mean(V[initial_rank[i, :k2]], axis=0)
        V = V_qe

    inv_index = [np.where(V[:, i] != 0)[0] for i in range(all_num)]

    jaccard_dist = np.zeros((num_q, all_num), dtype=np.float32)
    for i in range(num_q):
        non_zero = np.where(V[i, :] != 0)[0]
        temp_min = np.zeros((1, all_num), dtype=np.float32)
        for j in non_zero:
            temp_min[0, inv_index[j]] += np.minimum(V[i, j], V[inv_index[j], j])
        jaccard_dist[i] = 1.0 - temp_min / (2.0 - temp_min)

    final_dist = jaccard_dist[:, num_q:] * (1.0 - lambda_value) + original_dist[:num_q, num_q:] * lambda_value
    return final_dist
