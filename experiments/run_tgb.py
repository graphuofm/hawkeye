#!/usr/bin/env python
"""GraphEagleVision on a TGB link-property-prediction dataset.

Streaming protocol (re-streamed each epoch — the structural state evolution is
deterministic, so this is just re-running the cheap CPU maintenance):

  for epoch:
    model.reset_structure()
    for each chronological batch:
      - using the *current* structural state, score the train edges in this
        batch (+ random negatives), BCE loss, backprop
      - (val-eval epochs) using the current state, evaluate val edges in this
        batch against the TGB negatives -> running MRR
      - then add the whole batch's edges (update graph + indicators + rolling stats)
  early-stop on val MRR; final pass also evaluates test.

By default `--fusion struct_only` so this is a *standalone* structural model
(no GNN). Pairwise structural features are on by default.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from gev import GEVConfig, GraphEagleVision  # noqa: E402
from gev.data import load_tgb_linkproppred  # noqa: E402
from gev.utils import count_parameters, get_device, set_seed  # noqa: E402


def chrono_batches(n: int, bs: int):
    b = list(range(0, n, bs)) + [n]
    return [(b[i], b[i + 1]) for i in range(len(b) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tgbl-wiki")
    ap.add_argument("--indicators", default="degree,core")
    ap.add_argument("--fusion", default="struct_only", choices=["struct_only", "concat", "additive", "gated"])
    ap.add_argument("--encoder", default="mlp", choices=["mlp", "identity", "gru"])
    ap.add_argument("--no_pairwise", action="store_true", help="alias for --pairwise_mode none")
    ap.add_argument("--pairwise_mode", default="all", choices=["all", "cohesion", "generic", "none"],
                    help="all = generic CN/AA + k-family feats; cohesion = only k-family-derived; "
                         "generic = only classic CN/AA heuristics; none = off")
    ap.add_argument("--struct_dim", type=int, default=64)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--output_dim", type=int, default=128)
    ap.add_argument("--encoder_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--stats_decay", type=float, default=0.95)
    ap.add_argument("--trend_decays", default="", help="comma list of extra slower decays for "
                    "multi-timescale trend features, e.g. '0.99,0.999,0.9999'")
    ap.add_argument("--stat_groups", default="all",
                    help="comma list of rolling-stat groups, or 'all' / 'static' / 'dynamic'. "
                         "Groups: current/ema/std/delta/max_change/trend_<d>/recency")
    ap.add_argument("--feature_clip", type=float, default=10.0)
    ap.add_argument("--pairwise_backend", default="auto", choices=["auto", "loop", "sparse"],
                    help="auto: sparse-matrix path if avg-degree is high (dense graphs), else Python-loop")
    ap.add_argument("--pairwise_max_2hop", type=int, default=20000)
    ap.add_argument("--window_fraction", type=float, default=0.0,
                    help="sliding-window graph: drop edges older than this fraction of the dataset's time span. "
                         "0 (default) = cumulative graph (the standard).")
    ap.add_argument("--truss_recompute_every", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--num_train_neg", type=int, default=5)
    ap.add_argument("--val_subsample", type=int, default=3000,
                    help="per-epoch val edges to evaluate (<=0: all)")
    ap.add_argument("--test_subsample", type=int, default=0,
                    help="test edges to evaluate in the final pass (<=0: all). Use a finite value for huge datasets.")
    ap.add_argument("--val_every", type=int, default=1)
    ap.add_argument("--single_pass", action="store_true",
                    help="one chronological pass with online training + eval (for huge datasets where "
                         "re-streaming per epoch is infeasible). Ignores --epochs/--patience.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no_download", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    pw_mode = "none" if args.no_pairwise else args.pairwise_mode
    print(f"[setup] dataset={args.dataset} indicators={args.indicators} fusion={args.fusion} "
          f"pairwise={pw_mode} device={device}", flush=True)

    data = load_tgb_linkproppred(args.dataset, download=not args.no_download)
    src, dst, t = data.src.astype(np.int64), data.dst.astype(np.int64), data.t
    E, N = data.num_edges, data.num_nodes
    train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
    val_idx_all = np.where(val_mask)[0]
    test_idx_all = np.where(test_mask)[0]
    dst_pool = np.unique(dst)
    avg_deg = 2.0 * E / max(N, 1)
    use_csr = (args.pairwise_backend == "sparse") or (args.pairwise_backend == "auto" and avg_deg >= 50 and not args.no_pairwise)
    print(f"[data] E={E} N={N} train/val/test={int(train_mask.sum())}/{len(val_idx_all)}/{len(test_idx_all)} "
          f"avg_deg≈{avg_deg:.1f} pairwise_backend={'sparse' if use_csr else 'loop'}", flush=True)

    cfg = GEVConfig(
        indicators=[s.strip() for s in args.indicators.split(",") if s.strip()],
        stat_groups=[s.strip() for s in args.stat_groups.split(",") if s.strip()],
        trend_decays=[float(s) for s in args.trend_decays.split(",") if s.strip()],
        stats_decay=args.stats_decay, encoder_type=args.encoder, hidden_dim=args.hidden_dim,
        struct_dim=args.struct_dim, encoder_layers=args.encoder_layers, dropout=args.dropout,
        fusion_mode=args.fusion, output_dim=args.output_dim, feature_clip=args.feature_clip,
        truss_recompute_every=args.truss_recompute_every,
        pairwise_mode=pw_mode, use_pairwise=not args.no_pairwise, pairwise_max_2hop=args.pairwise_max_2hop,
    )
    if args.fusion != "struct_only":
        raise SystemExit("run_tgb.py currently supports --fusion struct_only; use run_tgb_with_base.py for fusion")
    model = GraphEagleVision(cfg).to(device)
    if args.window_fraction > 0:
        window_abs = float(args.window_fraction) * float(np.asarray(t).max() - np.asarray(t).min())
        model.set_window(window_abs)
        print(f"[window] sliding-window mode: drop edges older than {args.window_fraction:.3f} "
              f"× time-span = {window_abs:.0f} time units", flush=True)
    print(f"[model] {model} | params={count_parameters(model)}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    rng = np.random.default_rng(args.seed)
    batches = chrono_batches(E, args.batch_size)

    from tgb.linkproppred.evaluate import Evaluator
    evaluator = Evaluator(name=data._dataset.name)

    # which edge indices to evaluate per split (may be uniform-stride sub-sampled)
    def _subsample(idx_all, k):
        if k and k > 0 and len(idx_all) > k:
            stride = len(idx_all) / k
            return set(int(i) for i in idx_all[(np.arange(k) * stride).astype(np.int64)])
        return set(int(i) for i in idx_all)
    val_eval_set = _subsample(val_idx_all, args.val_subsample)
    test_eval_set = _subsample(test_idx_all, args.test_subsample)

    def stream_epoch(train: bool, do_val: bool, do_test: bool) -> Dict[str, float]:
        model.reset_structure()
        if train:
            model.train()
        else:
            model.eval()
        if do_val:
            data.load_val_ns()
        if do_test:
            data.load_test_ns()
        ns = data.negative_sampler if (do_val or do_test) else None
        total_loss, nb = 0.0, 0
        mrr_val: List[float] = []
        mrr_test: List[float] = []
        for (lo, hi) in batches:
            bsl, bdl, btl = src[lo:hi], dst[lo:hi], t[lo:hi]
            local_train = np.where(train_mask[lo:hi])[0]
            local_val = np.where(val_mask[lo:hi])[0]
            local_test = np.where(test_mask[lo:hi])[0]
            need_preds = (train and len(local_train) > 0) or do_val or do_test
            csr = model.build_pairwise_csr() if (use_csr and model.pairwise_dim and need_preds) else None

            # --- training on this batch's train edges (state = before batch) ---
            if train and len(local_train) > 0:
                ps, pd = bsl[local_train], bdl[local_train]
                nn_ = len(ps) * args.num_train_neg
                neg = dst_pool[rng.integers(0, len(dst_pool), size=nn_)]
                neg_src = np.repeat(ps, args.num_train_neg)
                sp = model.predict_scores(ps, pd, pairwise_csr=csr)
                sn = model.predict_scores(neg_src, neg, pairwise_csr=csr)
                logits = torch.cat([sp, sn]); labels = torch.cat([torch.ones_like(sp), torch.zeros_like(sn)])
                loss = bce(logits, labels)
                opt.zero_grad(); loss.backward(); opt.step()
                total_loss += float(loss); nb += 1

            # --- eval on this batch's val/test edges (state = before batch) ---
            eval_jobs = []
            if do_val:
                eval_jobs.append((local_val, "val", mrr_val, val_eval_set))
            if do_test:
                eval_jobs.append((local_test, "test", mrr_test, test_eval_set))
            for (local_idx, split, store, keep_set) in eval_jobs:
                use = [int(j) for j in local_idx if (lo + int(j)) in keep_set]
                if not use:
                    continue
                us = bsl[use].astype(np.int64); vs = bdl[use].astype(np.int64)
                ts = btl[use].astype(np.float64)
                neg_lists = ns.query_batch(torch.as_tensor(us), torch.as_tensor(vs),
                                           torch.as_tensor(ts), split_mode=split)
                q_src, q_dst, lens = [], [], []
                for k, negs in enumerate(neg_lists):
                    negs = np.asarray(negs, dtype=np.int64)
                    cand = np.concatenate([np.array([vs[k]], dtype=np.int64), negs])
                    q_dst.append(cand)
                    q_src.append(np.full(len(cand), us[k], dtype=np.int64))
                    lens.append(len(cand))
                q_src = np.concatenate(q_src); q_dst = np.concatenate(q_dst)
                with torch.no_grad():
                    scores = model.predict_scores(q_src, q_dst, pairwise_csr=csr).cpu().numpy()
                off = 0
                for L in lens:
                    seg = scores[off:off + L]; off += L
                    perf = evaluator.eval({"y_pred_pos": seg[0:1], "y_pred_neg": seg[1:],
                                           "eval_metric": ["mrr"]})
                    store.append(float(perf["mrr"]))

            # --- advance structure ---
            model.update_structure_batch(bsl, bdl, btl)
        out = {"train_loss": total_loss / max(nb, 1)}
        if do_val:
            out["val_mrr"] = float(np.mean(mrr_val)) if mrr_val else 0.0
        if do_test:
            out["test_mrr"] = float(np.mean(mrr_test)) if mrr_test else 0.0
        return out

    # --- training loop ---
    best_val, best_state, best_epoch, bad = -1.0, None, -1, 0
    history = []
    t_start = time.time()
    if args.single_pass:
        # one chronological pass: train online on train edges, eval val+test edges as we reach them
        rfin = stream_epoch(train=True, do_val=True, do_test=True)
        val_full, test_full = rfin.get("val_mrr", 0.0), rfin.get("test_mrr", 0.0)
        best_val, best_epoch = val_full, 1
        history.append({"epoch": 1, "train_loss": rfin["train_loss"], "val_mrr": val_full, "single_pass": True})
        print(f"[single-pass] loss={rfin['train_loss']:.4f}  val={val_full:.4f} test={test_full:.4f}", flush=True)
    else:
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
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                flag = " *"
            elif do_val:
                bad += 1
            print(f"[ep {ep:3d}] loss={r['train_loss']:.4f} val_mrr={vmrr:.4f}{flag}  ({dt:.1f}s)", flush=True)
            if do_val and bad >= args.patience:
                print(f"[early stop] no val improvement for {args.patience} evals"); break
        if best_state is not None:
            model.load_state_dict(best_state)
        print("[final] full val+test eval ...", flush=True)
        rfin = stream_epoch(train=False, do_val=True, do_test=True)
        val_full, test_full = rfin["val_mrr"], rfin["test_mrr"]
    total_min = (time.time() - t_start) / 60
    print(f"\n[RESULT] {args.dataset} | {args.fusion}({args.indicators}, pairwise={pw_mode}) | "
          f"val_mrr={val_full:.4f} | test_mrr={test_full:.4f} | best_epoch={best_epoch} | {total_min:.1f} min", flush=True)

    result = {
        "dataset": args.dataset, "fusion": args.fusion, "indicators": args.indicators,
        "stat_groups": args.stat_groups, "pairwise_mode": ("none" if args.no_pairwise else args.pairwise_mode),
        "encoder": args.encoder, "seed": args.seed,
        "val_mrr_subsample": best_val, "best_epoch": best_epoch,
        "val_mrr": val_full, "test_mrr": test_full, "params": count_parameters(model),
        "node_feat_dim": model.feature_dim, "pairwise_dim": model.pairwise_dim,
        "total_min": total_min, "args": vars(args), "history": history,
    }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(result) + "\n")
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
