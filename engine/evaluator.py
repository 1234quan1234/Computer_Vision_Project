import numpy as np
import torch
import torch.nn.functional as F

from engine.re_ranking import re_ranking


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.cpu().numpy()
    return np.asarray(value)


def extract_features(model, dataloader, device):
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
    qf, q_pids, q_camids = extract_features(model, query_loader, device)
    gf, g_pids, g_camids = extract_features(model, gallery_loader, device)

    if rerank:
        distmat = re_ranking(qf, gf, k1=k1, k2=k2, lambda_value=lambda_value)
    else:
        distmat = compute_distance_matrix(qf, gf, metric="cosine")

    cmc, mAP = evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=max_rank)
    return {"cmc": cmc, "mAP": mAP}
