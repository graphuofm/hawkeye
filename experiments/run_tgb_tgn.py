#!/usr/bin/env python
"""TGN baseline, and TGN + GraphEagleVision (the "enhance" experiment).

  * pure TGN:                 --gev_indicators ""        (uses TGN's own link head)
  * TGN + GraphEagleVision:   --gev_indicators degree,core --fusion gated
        the structural embedding + pairwise structural features are fused with
        TGN's interaction embedding before the link predictor.

Streaming protocol mirrors run_tgb.py: re-stream each epoch (TGN memory + GEV
structure both reset & rebuilt deterministically), predict-then-update batches,
TGB Evaluator for MRR, early-stop on val.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from gev import GEVConfig, GraphEagleVision  # noqa: E402
from gev.data import load_tgb_linkproppred  # noqa: E402
from gev.utils import count_parameters, get_device, set_seed  # noqa: E402
from integration.tgn import TGNLinkPredictor, TGNModel  # noqa: E402


def chrono_batches(n: int, bs: int):
    b = list(range(0, n, bs)) + [n]
    return [(b[i], b[i + 1]) for i in range(len(b) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tgbl-wiki")
    # TGN
    ap.add_argument("--memory_dim", type=int, default=100)
    ap.add_argument("--time_dim", type=int, default=100)
    ap.add_argument("--embedding_dim", type=int, default=100)
    ap.add_argument("--neighbor_size", type=int, default=10)
    ap.add_argument("--dropout", type=float, default=0.1)
    # GEV (empty indicators -> pure TGN baseline)
    ap.add_argument("--gev_indicators", default="degree,core",
                    help='comma list; empty string "" disables GEV (pure TGN)')
    ap.add_argument("--coupling", default="fusion", choices=["fusion", "score_ensemble", "aux"],
                    help="fusion = late fuse z & h_struct via --fusion; score_ensemble = "
                         "alpha*score_TGN + (1-alpha)*score_GEV (learnable alpha); "
                         "aux = late fusion + auxiliary head making TGN predict the indicator values")
    ap.add_argument("--fusion", default="gated", choices=["gated", "concat", "additive", "attn", "film"])
    ap.add_argument("--aux_weight", type=float, default=0.1)
    ap.add_argument("--stat_groups", default="all", help="'all'/'static'/'dynamic' or comma list of group names")
    ap.add_argument("--trend_decays", default="", help="comma list of extra slower decays, e.g. '0.99,0.999,0.9999'")
    ap.add_argument("--no_pairwise", action="store_true", help="alias for --pairwise_mode none")
    ap.add_argument("--pairwise_mode", default="all", choices=["all", "cohesion", "generic", "none"])
    ap.add_argument("--struct_dim", type=int, default=64)
    ap.add_argument("--gev_hidden", type=int, default=128)
    ap.add_argument("--feature_clip", type=float, default=10.0)
    ap.add_argument("--pairwise_max_2hop", type=int, default=20000)
    ap.add_argument("--pairwise_backend", default="auto", choices=["auto", "loop", "sparse"])
    ap.add_argument("--truss_recompute_every", type=int, default=64)
    # training
    ap.add_argument("--batch_size", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0, help="max grad norm (0 disables)")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--val_subsample", type=int, default=3000)
    ap.add_argument("--val_every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no_download", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    use_gev = bool(args.gev_indicators.strip())
    print(f"[setup] dataset={args.dataset} model={'TGN+GEV('+args.gev_indicators+')' if use_gev else 'TGN'} "
          f"fusion={args.fusion if use_gev else '-'} device={device}", flush=True)

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
    use_csr = (args.pairwise_backend == "sparse") or (args.pairwise_backend == "auto" and avg_deg >= 50 and bool(args.gev_indicators.strip()) and not args.no_pairwise)
    print(f"[data] E={E} N={N} train/val/test={int(train_mask.sum())}/{len(val_idx_all)}/{len(test_idx_all)} "
          f"raw_msg_dim={raw_msg_dim} avg_deg≈{avg_deg:.1f} pairwise_backend={'sparse' if use_csr else 'loop'}", flush=True)

    # --- models ---
    tgn = TGNModel(N, raw_msg_dim, memory_dim=args.memory_dim, time_dim=args.time_dim,
                   embedding_dim=args.embedding_dim, neighbor_size=args.neighbor_size,
                   dropout=args.dropout).to(device)
    edge_t_t = torch.from_numpy(np.asarray(t)).to(device)
    edge_msg_t = torch.from_numpy(np.asarray(edge_feat, dtype=np.float32)).to(device)
    tgn.set_edges(edge_t_t, edge_msg_t)

    gev: Optional[GraphEagleVision] = None
    tgn_head = None
    aux_head = None
    alpha = None  # learnable mixing weight for score_ensemble
    if use_gev:
        # for score_ensemble GEV runs in struct_only mode (its own head produces s_GEV);
        # otherwise GEV fuses z with h_struct internally.
        gev_fusion = "struct_only" if args.coupling == "score_ensemble" else args.fusion
        cfg = GEVConfig(
            indicators=[s.strip() for s in args.gev_indicators.split(",") if s.strip()],
            stat_groups=[s.strip() for s in args.stat_groups.split(",") if s.strip()],
            trend_decays=[float(s) for s in args.trend_decays.split(",") if s.strip()],
            encoder_type="mlp", hidden_dim=args.gev_hidden, struct_dim=args.struct_dim,
            fusion_mode=gev_fusion, output_dim=args.embedding_dim, inter_dim=args.embedding_dim,
            feature_clip=args.feature_clip,
            pairwise_mode=("none" if args.no_pairwise else args.pairwise_mode),
            use_pairwise=not args.no_pairwise,
            pairwise_max_2hop=args.pairwise_max_2hop, truss_recompute_every=args.truss_recompute_every,
        )
        gev = GraphEagleVision(cfg).to(device)
        if args.coupling == "score_ensemble":
            tgn_head = TGNLinkPredictor(args.embedding_dim).to(device)
            alpha = torch.nn.Parameter(torch.zeros(1, device=device))  # sigmoid(0)=0.5 init
        if args.coupling == "aux":
            aux_head = torch.nn.Linear(args.embedding_dim, gev.K).to(device)
    else:
        tgn_head = TGNLinkPredictor(args.embedding_dim).to(device)

    params = list(tgn.parameters())
    if gev is not None:
        params += list(gev.parameters())
    if tgn_head is not None:
        params += list(tgn_head.parameters())
    if aux_head is not None:
        params += list(aux_head.parameters())
    if alpha is not None:
        params += [alpha]
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    rng = np.random.default_rng(args.seed)
    batches = chrono_batches(E, args.batch_size)
    n_params = sum(p.numel() for p in params if p.requires_grad)
    print(f"[model] coupling={args.coupling if use_gev else '-'} params={n_params}", flush=True)

    from tgb.linkproppred.evaluate import Evaluator
    evaluator = Evaluator(name=data._dataset.name)

    val_eval_set = val_idx_all
    if args.val_subsample and args.val_subsample > 0 and len(val_idx_all) > args.val_subsample:
        stride = len(val_idx_all) / args.val_subsample
        val_eval_set = val_idx_all[(np.arange(args.val_subsample) * stride).astype(np.int64)]
    val_eval_set = set(int(i) for i in val_eval_set)

    def assoc_lookup(z: torch.Tensor, n_id_np: np.ndarray):
        """Build a map node_id -> row in z (for the n_id we embedded)."""
        m = {int(nid): i for i, nid in enumerate(n_id_np)}
        return m

    def score_pairs(z, idx_map, q_src_np, q_dst_np, csr=None):
        rs = torch.as_tensor([idx_map[int(x)] for x in q_src_np], dtype=torch.long, device=device)
        rd = torch.as_tensor([idx_map[int(x)] for x in q_dst_np], dtype=torch.long, device=device)
        z_s, z_d = z[rs], z[rd]
        if gev is None:
            return tgn_head(z_s, z_d)
        if args.coupling == "score_ensemble":
            s_tgn = tgn_head(z_s, z_d)
            s_gev = gev.predict_scores(q_src_np, q_dst_np, pairwise_csr=csr)  # struct_only
            w = torch.sigmoid(alpha)
            return w * s_tgn + (1.0 - w) * s_gev
        # "fusion" / "aux": GEV fuses z with h_struct internally
        return gev.predict_scores(q_src_np, q_dst_np, h_inter_src=z_s, h_inter_dst=z_d, pairwise_csr=csr)

    def stream_epoch(train: bool, do_val: bool, do_test: bool) -> Dict[str, float]:
        tgn.reset()
        if gev is not None:
            gev.reset_structure()
        mode_train = train
        (tgn.train if mode_train else tgn.eval)()
        if gev is not None:
            (gev.train if mode_train else gev.eval)()
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
            need_preds = (mode_train and len(local_train) > 0) or do_val or do_test
            csr = gev.build_pairwise_csr() if (gev is not None and use_csr and gev.pairwise_dim and need_preds) else None

            # --- training on this batch's train edges ---
            if mode_train and len(local_train) > 0:
                ps, pd = bsl[local_train], bdl[local_train]
                k = len(ps) * 1  # 1 negative per positive (standard TGN)
                neg = dst_pool[rng.integers(0, len(dst_pool), size=k)]
                allnodes = np.unique(np.concatenate([ps, pd, neg]))
                z = tgn.embed(torch.from_numpy(allnodes))
                idx_map = assoc_lookup(z, allnodes)
                sp = score_pairs(z, idx_map, ps, pd, csr=csr)
                sn = score_pairs(z, idx_map, ps, neg, csr=csr)
                logits = torch.cat([sp, sn]); labels = torch.cat([torch.ones_like(sp), torch.zeros_like(sn)])
                loss = bce(logits, labels)
                if aux_head is not None and gev is not None:
                    # auxiliary task: make TGN's embeddings predict the current indicator values
                    tgt = np.stack([gev.stats.current.get(int(n), np.zeros(gev.K, dtype=np.float32)) for n in allnodes])
                    tgt = torch.as_tensor(np.log1p(np.maximum(tgt, 0.0)), dtype=torch.float32, device=device)
                    loss = loss + args.aux_weight * torch.nn.functional.mse_loss(aux_head(z), tgt)
                opt.zero_grad(); loss.backward()
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                opt.step()
                total_loss += float(loss); nb += 1

            # --- eval on this batch's val/test edges ---
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
                q_src, q_dst, lens = [], [], []
                for kk, negs in enumerate(neg_lists):
                    negs = np.asarray(negs, dtype=np.int64)
                    cand = np.concatenate([np.array([vs[kk]], dtype=np.int64), negs])
                    q_dst.append(cand); q_src.append(np.full(len(cand), us[kk], dtype=np.int64)); lens.append(len(cand))
                q_src = np.concatenate(q_src); q_dst = np.concatenate(q_dst)
                allnodes = np.unique(np.concatenate([q_src, q_dst]))
                with torch.no_grad():
                    z = tgn.embed(torch.from_numpy(allnodes))
                    idx_map = assoc_lookup(z, allnodes)
                    scores = score_pairs(z, idx_map, q_src, q_dst, csr=csr).cpu().numpy()
                off = 0
                for L in lens:
                    seg = scores[off:off + L]; off += L
                    perf = evaluator.eval({"y_pred_pos": seg[0:1], "y_pred_neg": seg[1:], "eval_metric": ["mrr"]})
                    store.append(float(perf["mrr"]))

            # --- advance state ---
            t_batch = torch.from_numpy(np.asarray(btl)).to(device)
            tgn.update_state(torch.from_numpy(bsl), torch.from_numpy(bdl), t_batch, msg_t)
            if gev is not None:
                gev.update_structure_batch(bsl, bdl, btl)
            tgn.detach()
        out = {"train_loss": total_loss / max(nb, 1)}
        if do_val:
            out["val_mrr"] = float(np.mean(mrr_val)) if mrr_val else 0.0
        if do_test:
            out["test_mrr"] = float(np.mean(mrr_test)) if mrr_test else 0.0
        return out

    # --- training loop ---
    # modules to checkpoint
    _mods = {"tgn": tgn}
    if gev is not None: _mods["gev"] = gev
    if tgn_head is not None: _mods["tgn_head"] = tgn_head
    if aux_head is not None: _mods["aux_head"] = aux_head

    def _snapshot():
        sd = {n: {k: v.detach().cpu().clone() for k, v in m.state_dict().items()} for n, m in _mods.items()}
        if alpha is not None:
            sd["_alpha"] = alpha.detach().cpu().clone()
        return sd

    def _restore(sd):
        for n, m in _mods.items():
            m.load_state_dict(sd[n])
        if alpha is not None and "_alpha" in sd:
            with torch.no_grad():
                alpha.copy_(sd["_alpha"].to(alpha.device))

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
    pw_mode = ("none" if args.no_pairwise else args.pairwise_mode) if use_gev else "-"
    coup = args.coupling if (use_gev and args.coupling != "fusion") else args.fusion
    model_name = f"TGN+GEV({args.gev_indicators},{coup},pw={pw_mode})" if use_gev else "TGN"
    print(f"\n[RESULT] {args.dataset} | {model_name} | val_mrr={val_full:.4f} | test_mrr={test_full:.4f} "
          f"| best_epoch={best_epoch} | {total_min:.1f} min", flush=True)

    result = {
        "dataset": args.dataset, "model": model_name, "gev": use_gev,
        "gev_indicators": args.gev_indicators if use_gev else "",
        "coupling": (args.coupling if use_gev else ""), "fusion": args.fusion if use_gev else "",
        "stat_groups": args.stat_groups, "pairwise_mode": pw_mode,
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
