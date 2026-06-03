import numpy as np
import torch
import torch.nn.functional as F

from engine.re_ranking import re_ranking


def _to_numpy(value):
    """Convert a torch Tensor or array-like to a NumPy array."""
    if torch.is_tensor(value):
        return value.cpu().numpy()
    return np.asarray(value)


def extract_features(model, dataloader, device):
    """Extract L2-normalized feature vectors from all images in a dataloader.

    Runs the model in eval mode with no gradients and BF16 autocast
    (on CUDA). Uses ``fusion_feat`` from the model output as the
    final embedding for retrieval.

    Args:
        model: A ``ClipSENet`` model instance.
        dataloader: DataLoader yielding ``(images, pids, camids, frames, paths)``.
        device: Torch device to run inference on.

    Returns:
        Tuple of ``(features, pids, camids)`` where:
          - ``features``: ``(N, 512)`` float32 NumPy array of L2-normalized embeddings.
          - ``pids``: ``(N,)`` int array of vehicle identity labels.
          - ``camids``: ``(N,)`` int array of camera IDs.
    """
    model.eval()
    autocast_enabled = device.type == "cuda"

    feats = []
    pids = []
    camids = []

    with torch.no_grad():
        for batch in dataloader:
            images, pid, camid, _, _ = batch
            images = images.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                outputs = model(images, return_logits=False)
                feat = outputs["fusion_feat"]

            feat = F.normalize(feat.float(), dim=1).cpu().numpy()
            feats.append(feat)
            pids.extend(_to_numpy(pid).tolist())
            camids.extend(_to_numpy(camid).tolist())

    if feats:
        feats = np.vstack(feats)
    else:
        feats = np.zeros((0, 512), dtype=np.float32)

    return feats, np.asarray(pids), np.asarray(camids)


def compute_distance_matrix(qf, gf, metric: str = "cosine"):
    """Compute pairwise distance matrix between query and gallery features.

    Args:
        qf: Query features of shape ``(num_query, D)``.
        gf: Gallery features of shape ``(num_gallery, D)``.
        metric: Distance metric — ``'cosine'`` (1 - cosine similarity)
            or ``'euclidean'`` (squared L2 distance).

    Returns:
        Distance matrix of shape ``(num_query, num_gallery)``.
    """
    if metric == "cosine":
        qf = qf / np.linalg.norm(qf, axis=1, keepdims=True)
        gf = gf / np.linalg.norm(gf, axis=1, keepdims=True)
        distmat = 1.0 - np.dot(qf, gf.T)
        return distmat

    qf_sq = np.sum(qf ** 2, axis=1, keepdims=True)
    gf_sq = np.sum(gf ** 2, axis=1, keepdims=True).T
    distmat = qf_sq + gf_sq - 2.0 * np.dot(qf, gf.T)
    return np.maximum(distmat, 0.0)


def evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids, max_rank: int = 50):
    """Compute mAP and CMC curve from a distance matrix.

    For each query, gallery images with the **same pid AND same camid**
    are excluded (standard ReID protocol — a query should not match
    itself from the same camera viewpoint).

    Args:
        distmat: Distance matrix of shape ``(num_query, num_gallery)``.
        q_pids: Query identity labels ``(num_query,)``.
        g_pids: Gallery identity labels ``(num_gallery,)``.
        q_camids: Query camera IDs ``(num_query,)``.
        g_camids: Gallery camera IDs ``(num_gallery,)``.
        max_rank: Maximum rank for CMC computation.

    Returns:
        Tuple of ``(cmc, mAP)`` where:
          - ``cmc``: CMC curve array of length ``max_rank``.
            ``cmc[0]`` = Rank-1 accuracy, ``cmc[4]`` = Rank-5, etc.
          - ``mAP``: Mean Average Precision (float).
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, None])

    all_cmc = []
    all_ap = []
    valid_q = 0

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        raw_matches = matches[q_idx][keep]
        if not np.any(raw_matches):
            continue

        cmc = raw_matches.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])

        num_rel = raw_matches.sum()
        tmp_cmc = raw_matches.cumsum()
        precision = tmp_cmc / (np.arange(len(tmp_cmc)) + 1.0)
        ap = (precision * raw_matches).sum() / num_rel
        all_ap.append(ap)
        valid_q += 1

    if valid_q == 0:
        return np.zeros(max_rank), 0.0

    cmc = np.mean(np.stack(all_cmc, axis=0), axis=0)
    mAP = float(np.mean(all_ap))
    return cmc, mAP


def evaluate(
    model,
    query_loader,
    gallery_loader,
    device,
    rerank: bool = True,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3,
    max_rank: int = 50,
):
    """Full evaluation pipeline: extract features, compute distances, and rank.

    Orchestrates the complete evaluation workflow:
      1. Extract features from query and gallery sets.
      2. Compute distance matrix (optionally with k-reciprocal re-ranking).
      3. Compute mAP and CMC metrics.

    Args:
        model: Trained ``ClipSENet`` model.
        query_loader: DataLoader for the query set.
        gallery_loader: DataLoader for the gallery set.
        device: Torch device.
        rerank: If True, apply k-reciprocal re-ranking to refine distances.
        k1: k-reciprocal parameter — size of the initial neighbor set.
        k2: k-reciprocal parameter — size for query expansion averaging.
        lambda_value: Interpolation weight between Jaccard and original distance.
        max_rank: Maximum rank for CMC curve.

    Returns:
        Dict with ``'cmc'`` (CMC curve array) and ``'mAP'`` (float).
    """
    qf, q_pids, q_camids = extract_features(model, query_loader, device)
    gf, g_pids, g_camids = extract_features(model, gallery_loader, device)

    if rerank:
        distmat = re_ranking(qf, gf, k1=k1, k2=k2, lambda_value=lambda_value)
    else:
        distmat = compute_distance_matrix(qf, gf, metric="cosine")

    cmc, mAP = evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=max_rank)
    return {"cmc": cmc, "mAP": mAP}
