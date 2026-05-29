import numpy as np

from engine.evaluator import compute_distance_matrix, evaluate_rank
from engine.re_ranking import re_ranking


def test_evaluation() -> None:
    num_q, num_g, dim = 10, 50, 512
    qf = np.random.randn(num_q, dim).astype(np.float32)
    gf = np.random.randn(num_g, dim).astype(np.float32)

    q_pids = np.random.randint(0, 5, size=num_q)
    g_pids = np.random.randint(0, 5, size=num_g)
    q_camids = np.random.randint(1, 4, size=num_q)
    g_camids = np.random.randint(1, 4, size=num_g)

    distmat = compute_distance_matrix(qf, gf, metric="cosine")
    cmc, mAP = evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=10)

    assert 0.0 <= mAP <= 1.0, "mAP out of range"
    assert 0.0 <= cmc[0] <= 1.0, "Rank-1 out of range"

    distmat_rr = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
    assert distmat_rr.shape == distmat.shape, "Re-ranking shape mismatch"
    assert not np.isnan(distmat_rr).any(), "Re-ranking produced NaN"
    print(f"OK: evaluation | mAP={mAP:.4f} | R1={cmc[0]:.4f}")


if __name__ == "__main__":
    test_evaluation()
