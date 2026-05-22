#!/usr/bin/env python
"""Download TGB link-property-prediction datasets into ./data/tgb/.

Usage: python experiments/download_datasets.py [name ...]
       (default: the small/medium ones)
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from gev.data import load_tgb_linkproppred

SMALL = ["tgbl-wiki", "tgbl-enron", "tgbl-uci", "tgbl-subreddit", "tgbl-lastfm"]
MEDIUM = ["tgbl-review"]
LARGE = ["tgbl-coin", "tgbl-comment", "tgbl-flight"]

if __name__ == "__main__":
    names = sys.argv[1:] or (SMALL + MEDIUM)
    for nm in names:
        try:
            d = load_tgb_linkproppred(nm, download=True)
            print(f"OK {d.name}: E={d.num_edges} N={d.num_nodes} metric={d.metric} "
                  f"train/val/test={d.train_mask.sum()}/{d.val_mask.sum()}/{d.test_mask.sum()} "
                  f"edge_feat={None if d.edge_feat is None else d.edge_feat.shape}", flush=True)
        except Exception as e:
            print(f"FAIL {nm}: {type(e).__name__}: {e}", flush=True)
