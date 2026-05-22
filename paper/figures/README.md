# paper/figures — reproducible figure pipeline

Every figure in the paper is **fully reproducible**: its generating code and the
exact data it consumes both live here.

```
figures/
  data/            data snapshots each figure reads (CSV/JSON, committed)
  src/             generator scripts — one per figure
  <fig>.pdf        vector output  -> copied to paper_ol/figures/ for Overleaf
  <fig>.png        raster preview -> for the paper/ markdown discussion draft
  _archive_old/    figures from the earlier GraphEagleVision draft (unused)
```

## Convention

- Each figure has one script `src/make_<fig>.py`.
- The script reads ONLY from `figures/data/` (never from `../../results/`
  directly) — so the figure is reproducible even if `results/` changes.
- A separate step extracts the needed numbers from `results/...` into
  `figures/data/` (the extractor is the top of each `make_<fig>.py`, guarded so
  it is re-run only when explicitly asked).
- Each script writes both `<fig>.pdf` (vector, for Overleaf) and `<fig>.png`
  (raster, for the markdown draft).

## Regenerate everything

```
cd paper/figures
for f in src/make_*.py; do python "$f"; done
# then sync PDFs to the Overleaf bundle:
cp *.pdf ../../paper_ol/figures/
```

## Figure list

| figure | script | data | status |
|---|---|---|---|
| Fig.2 discAUC bar chart | src/make_fig_discauc.py | data/discauc.csv | ready |
| Fig.1 running example | src/make_fig_running.py | (schematic, no data) | TODO |
| Fig.3 architecture | src/make_fig_arch.py | (schematic, no data) | TODO |
| Fig.4 swap-in results | src/make_fig_swapin.py | data/swapin.csv | pending sweep |
| Fig.5 window ablation | src/make_fig_window.py | data/window.csv | pending sweep |
