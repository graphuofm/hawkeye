#!/usr/bin/env python
"""DyGFormer structure-channel SWAP-IN experiment.

The 4-way ablation that lets us answer: *is DyGFormer's built-in neighbour-
cooccurrence channel underfitted, and does swapping it for a cohesion-aware
(GEV) channel help?*

  --struct_channel none    : pure DyGFormer (no structure channel)
  --struct_channel cooccur : DyGFormer (original Yu et al. 2023)
  --struct_channel gev     : DyGFormer's cooccur is REPLACED by GEV cohesion ch.
  --struct_channel both    : DyGFormer keeps cooccur AND gets GEV cohesion ch.

The GEV channel is computed from a `CohesionCache` advancing in lockstep with
the event stream — the same cache is reusable for TNCN / TPNet experiments.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from gev.cache import CohesionCache  # noqa: E402
from gev.data import load_tgb_linkproppred  # noqa: E402
from gev.utils import count_parameters, get_device, set_seed  # noqa: E402
from integration.dygformer_lite import DyGFormerLite, DyGFormerLiteLinkPredictor  # noqa: E402


def chrono_batches(n: int, bs: int):
    b = list(range(0, n, bs)) + [n]
    return [(b[i], b[i + 1]) for i in range(len(b) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tgbl-wiki")
    # struct channel
    ap.add_argument("--struct_channel", default="cooccur",
                    choices=["none", "cooccur", "gev", "both"])
    ap.add_argument("--struct_indicators", default="degree,core")
    ap.add_argument("--struct_pairwise_mode", default="cohesion",
                    choices=["all", "cohesion", "generic"])
    ap.add_argument("--struct_trend_decays", default="0.99,0.999,0.9999")
    ap.add_argument("--slot_backend", default="fast", choices=["fast", "full"],
                    help="fast = 6-dim per-pair set intersection (default); "
                         "full = 23-dim CSR-based features (slow on dense graphs)")
    # DyGFormer-lite
    ap.add_argument("--embedding_dim", type=int, default=100)
    ap.add_argument("--time_dim", type=int, default=100)
    ap.add_argument("--history_size", type=int, default=32)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--num_heads", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    # training
    ap.add_argument("--batch_size", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--val_subsample", type=int, default=2000)
    ap.add_argument("--val_every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no_download", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    use_cooccur = args.struct_channel in ("cooccur", "both")
    use_gev = args.struct_channel in ("gev", "both")
    tag = f"DyGF-swap[{args.struct_channel}]"
    if use_gev:
        tag += f"({args.struct_indicators},pw={args.struct_pairwise_mode})"
    print(f"[setup] dataset={args.dataset} tag={tag} device={device}", flush=True)

    data = load_tgb_linkproppred(args.dataset, download=not args.no_download)
    src, dst, t = data.src.astype(np.int64), data.dst.astype(np.int64), data.t
    E, N = data.num_edges, data.num_nodes
    train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
    val_idx_all = np.where(val_mask)[0]
    test_idx_all = np.where(test_mask)[0]
    dst_pool = np.unique(dst)
    edge_feat = data.edge_feat if data.edge_feat is not None else np.zeros((E, 1), dtype=np.float32)
    raw_msg_dim = edge_feat.shape[1]
    avg_deg = 2.0 * E / max(N, 1)
    print(f"[data] E={E} N={N} train/val/test={int(train_mask.sum())}/{len(val_idx_all)}/{len(test_idx_all)} "
          f"raw_msg_dim={raw_msg_dim} avg_deg≈{avg_deg:.1f}", flush=True)

    # --- cohesion cache (only built if needed) ---
    cache: Optional[CohesionCache] = None
    struct_dim = 0
    if use_gev:
        cache = CohesionCache(
            indicators=[s.strip() for s in args.struct_indicators.split(",") if s.strip()],
            trend_decays=[float(s) for s in args.struct_trend_decays.split(",") if s.strip()],
            stat_groups=("current",),
            pairwise_mode=args.struct_pairwise_mode,
            use_csr=(avg_deg >= 50),
            device=device,
        )
        struct_dim = (cache.FAST_SLOT_DIM if args.slot_backend == "fast"
                      else cache.pair_feat_dim)
        print(f"[cache] indicators={cache.cfg.indicators} slot_backend={args.slot_backend} "
              f"struct_dim={struct_dim}", flush=True)

    # --- models ---
    base = DyGFormerLite(
        N, raw_msg_dim,
        embedding_dim=args.embedding_dim, time_dim=args.time_dim,
        history_size=args.history_size, num_layers=args.num_layers,
        num_heads=args.num_heads, dropout=args.dropout,
        use_cooccur=use_cooccur, struct_channel_dim=struct_dim,
    ).to(device)
    head = DyGFormerLiteLinkPredictor(args.embedding_dim).to(device)
    edge_msg_t = torch.from_numpy(np.asarray(edge_feat, dtype=np.float32)).to(device)

    params = list(base.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    rng = np.random.default_rng(args.seed)
    batches = chrono_batches(E, args.batch_size)
    n_params = sum(p.numel() for p in params if p.requires_grad)
    print(f"[model] params={n_params} struct_channel_dim={struct_dim}", flush=True)

    from tgb.linkproppred.evaluate import Evaluator
    evaluator = Evaluator(name=data._dataset.name)

    val_eval_set = val_idx_all
    if args.val_subsample and args.val_subsample > 0 and len(val_idx_all) > args.val_subsample:
        stride = len(val_idx_all) / args.val_subsample
        val_eval_set = val_idx_all[(np.arange(args.val_subsample) * stride).astype(np.int64)]
    val_eval_set = set(int(i) for i in val_eval_set)

    def make_struct_provider():
        """Return a callable usable by base.embed_pairs / base.embed."""
        if cache is None:
            return None
        c = cache
        if args.slot_backend == "fast":
            def _provider(hist_nids: torch.Tensor, peer_nids: torch.Tensor) -> torch.Tensor:
                return c.slot_features_fast(hist_nids, peer_nids, device=device)
        else:
            def _provider(hist_nids: torch.Tensor, peer_nids: torch.Tensor) -> torch.Tensor:
                return c.slot_features(hist_nids, peer_nids, device=device)
        return _provider

    def score_pairs(q_src_np: np.ndarray, q_dst_np: np.ndarray, t_scalar: int):
        # Embed src and dst separately so the cohesion channel is computed against
        # the correct peer in each case.
        q_src_t = torch.from_numpy(q_src_np).to(device).long()
        q_dst_t = torch.from_numpy(q_dst_np).to(device).long()
        t_q = torch.full((len(q_src_np),), int(t_scalar), dtype=torch.long, device=device)
        sp = make_struct_provider()
        z_s = base.embed(q_src_t, t_q, peer_ids=q_dst_t, struct_provider=sp)
        z_d = base.embed(q_dst_t, t_q, peer_ids=q_src_t, struct_provider=sp)
        return head(z_s, z_d)

    def stream_epoch(train: bool, do_val: bool, do_test: bool) -> Dict[str, float]:
        base.reset()
        if cache is not None:
            cache.reset()
        (base.train if train else base.eval)()
        head.train(mode=train)
        if do_val:
            data.load_val_ns()
        if do_test:
            data.load_test_ns()
        ns = data.negative_sampler if (do_val or do_test) else None
        total_loss, nb = 0.0, 0
        mrr_val: List[float] = []
        mrr_test: List[float] = []

        for (lo, hi) in batches:
            bsl, bdl = src[lo:hi], dst[lo:hi]
            btl = t[lo:hi]
            msg_t = edge_msg_t[lo:hi]
            local_train = np.where(train_mask[lo:hi])[0]
            t_q_scalar = int(btl.max()) if len(btl) else 0

            # --- training step on this batch ---
            if train and len(local_train) > 0:
                ps, pd = bsl[local_train], bdl[local_train]
                k = len(ps)
                neg = dst_pool[rng.integers(0, len(dst_pool), size=k)]
                sp = score_pairs(ps, pd, t_q_scalar)
                sn = score_pairs(ps, neg, t_q_scalar)
                logits = torch.cat([sp, sn])
                labels = torch.cat([torch.ones_like(sp), torch.zeros_like(sn)])
                loss = bce(logits, labels)
                opt.zero_grad(); loss.backward()
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                opt.step()
                total_loss += float(loss); nb += 1

            # --- eval step on this batch ---
            eval_jobs = []
            if do_val:
                eval_jobs.append((np.where(val_mask[lo:hi])[0], "val", mrr_val))
            if do_test:
                eval_jobs.append((np.where(test_mask[lo:hi])[0], "test", mrr_test))
            for (local_idx, split, store) in eval_jobs:
                use = [int(j) for j in local_idx if split != "val" or (lo + int(j)) in val_eval_set]
                if not use:
                    continue
                us = bsl[use].astype(np.int64); vs = bdl[use].astype(np.int64); ts = btl[use].astype(np.float64)
                neg_lists = ns.query_batch(torch.as_tensor(us), torch.as_tensor(vs),
                                           torch.as_tensor(ts), split_mode=split)
                # Batch all (query, cand) pairs into one embed call. Each query has
                # 1 pos + N neg candidates; we concatenate, score once, and split.
                q_src_parts, q_dst_parts, lens = [], [], []
                for kk, negs in enumerate(neg_lists):
                    negs = np.asarray(negs, dtype=np.int64)
                    cand = np.concatenate([np.array([vs[kk]], dtype=np.int64), negs])
                    q_dst_parts.append(cand)
                    q_src_parts.append(np.full(len(cand), us[kk], dtype=np.int64))
                    lens.append(len(cand))
                if not lens:
                    continue
                q_src_all = np.concatenate(q_src_parts)
                q_dst_all = np.concatenate(q_dst_parts)
                # chunk to keep GPU memory bounded (B*K*D = 4000*32*100 ≈ 50MB)
                CHUNK = 4096
                scores_chunks = []
                with torch.no_grad():
                    for i0 in range(0, len(q_src_all), CHUNK):
                        i1 = min(i0 + CHUNK, len(q_src_all))
                        s = score_pairs(q_src_all[i0:i1], q_dst_all[i0:i1], t_q_scalar)
                        scores_chunks.append(s.cpu().numpy())
                scores = np.concatenate(scores_chunks) if scores_chunks else np.zeros(0, dtype=np.float32)
                off = 0
                for L in lens:
                    seg = scores[off:off + L]; off += L
                    perf = evaluator.eval({"y_pred_pos": seg[0:1],
                                           "y_pred_neg": seg[1:],
                                           "eval_metric": ["mrr"]})
                    store.append(float(perf["mrr"]))

            # --- advance state ---
            t_batch = torch.from_numpy(np.asarray(btl)).to(device)
            base.update_state(torch.from_numpy(bsl), torch.from_numpy(bdl), t_batch, msg_t)
            if cache is not None:
                cache.advance(bsl, bdl, btl)

        out = {"train_loss": total_loss / max(nb, 1)}
        if do_val:
            out["val_mrr"] = float(np.mean(mrr_val)) if mrr_val else 0.0
        if do_test:
            out["test_mrr"] = float(np.mean(mrr_test)) if mrr_test else 0.0
        return out

    # --- training loop ---
    def _snapshot():
        return {
            "base": {k: v.detach().cpu().clone() for k, v in base.state_dict().items()},
            "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
        }
    def _restore(sd):
        base.load_state_dict(sd["base"]); head.load_state_dict(sd["head"])

    best_val, best_state, best_epoch, bad = -1.0, None, -1, 0
    history = []
    t_start = time.time()
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        do_val = (ep % args.val_every == 0) or (ep == args.epochs)
        r = stream_epoch(train=True, do_val=do_val, do_test=False)
        dt = time.time() - t0
        vmrr = r.get("val_mrr", float("nan"))
        history.append({"epoch": ep, "train_loss": r["train_loss"], "val_mrr": vmrr, "sec": dt})
        flag = ""
        if do_val and vmrr > best_val:
            best_val, best_epoch, bad = vmrr, ep, 0
            best_state = _snapshot()
            flag = " *"
        elif do_val:
            bad += 1
        print(f"[ep {ep:3d}] loss={r['train_loss']:.4f} val_mrr={vmrr:.4f}{flag}  ({dt:.1f}s)", flush=True)
        if do_val and bad >= args.patience:
            print(f"[early stop]"); break

    if best_state is not None:
        _restore(best_state)
    print("[final] full val+test eval ...", flush=True)
    rfin = stream_epoch(train=False, do_val=True, do_test=True)
    val_full, test_full = rfin["val_mrr"], rfin["test_mrr"]
    total_min = (time.time() - t_start) / 60
    print(f"\n[RESULT] {args.dataset} | {tag} | val_mrr={val_full:.4f} | test_mrr={test_full:.4f} "
          f"| best_epoch={best_epoch} | {total_min:.1f} min", flush=True)

    result = {
        "dataset": args.dataset, "tag": tag, "struct_channel": args.struct_channel,
        "struct_indicators": args.struct_indicators if use_gev else "",
        "struct_pairwise_mode": args.struct_pairwise_mode if use_gev else "",
        "seed": args.seed, "val_mrr_subsample": best_val, "best_epoch": best_epoch,
        "val_mrr": val_full, "test_mrr": test_full, "params": n_params,
        "total_min": total_min, "args": vars(args), "history": history,
    }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(result) + "\n")
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
