# Chapter 6 — Related Work

> Project: **Hawkeye** — CIKM 2026 Full Research Paper.
> Target length: ~0.6 page, 3 subsections. Placed after Experiments, kept
> concise (no extended discussion). Citations `[cite:key]`.

## 6.1 Temporal Link Prediction

Temporal-link-prediction methods fall into several families. *Memory- and
message-passing-based* models maintain an evolving node state: TGN keeps a
per-node memory updated by every event [cite:rossi2020tgn], DyRep performs
event-driven representation updates [cite:trivedi2019dyrep], and JODIE learns
the temporal trajectory of node embeddings [cite:kumar2019jodie].
*Attention- and Transformer-based* models encode the temporal neighbourhood:
TGAT introduces temporal graph attention [cite:xu2020tgat], CAWN uses causal
anonymous walks for inductive learning [cite:wang2021cawn], and DyGFormer feeds
the historical-neighbour sequence into a Transformer with a cooccurrence
encoder as its structure channel [cite:yu2023dygformer]. *Lightweight and
matrix-projection* models include GraphMixer, which uses only MLPs and mean
pooling [cite:cong2023graphmixer], and TPNet, which projects a temporal-walk
matrix and reaches state-of-the-art accuracy on TGB [cite:lu2024tpnet].
*Signal-specific* methods include NAT, which accelerates common-neighbour
computation with dictionary-style neighbourhood representations
[cite:luo2022nat]; TNCN, which dynamically maintains a temporal-common-neighbour
dictionary [cite:zhang2024tncn]; and recent work encoding interaction-behaviour
patterns [cite:zhang2025ipnet] or removing degree-distribution bias via causal
inference [cite:zhang2025tide].

*Distinction.* In all of these the structure channel is either absent
(TGN/DyRep/TGAT/GraphMixer) or restricted to the 1-hop common neighbour
(DyGFormer's cooccurrence, TNCN, NAT). Hawkeye instead targets the **2-hop
cohesive bridge** — a structural signal far more informative than the 1-hop
common neighbour on sparse temporal graphs — realised through incremental
$k$-family maintenance.

## 6.2 Structural Features in Graph Learning

On static graphs, structure-based link prediction has a long history: common
neighbours (CN), Adamic–Adar (AA), and the Jaccard coefficient
[cite:liben2007linkpred]; SEAL extracts enclosing subgraphs and applies a GNN
[cite:zhang2018seal]. On dynamic graphs, CTGCN performs a $k$-core
decomposition on each temporal snapshot and runs a separate GCN at each core
level [cite:chen2020ctgcn] — the most direct precursor of our work — and TTGCN
extends this idea to $k$-truss [cite:ttgcn2024].

*Distinction.* CTGCN/TTGCN are discrete-snapshot models that use $k$-core /
$k$-truss to *guide GNN aggregation*; they do not model the temporal evolution
of the indicators and are validated only on small datasets. Hawkeye operates on
a continuous-time backbone, uses the $k$-family to *build 2-hop cohesive-bridge
pairwise features* rather than to steer aggregation, validates on large-scale
TGB/DGB benchmarks, and gives an explicit characterisation of its applicability
boundary.

## 6.3 Dense-Subgraph Decomposition and Incremental Maintenance

The $k$-core was introduced by Seidman [cite:seidman1983kcore]; Batagelj and
Zaversnik gave an $O(m)$ linear-time algorithm [cite:batagelj2003cores].
Incremental $k$-core maintenance on dynamic graphs performs a local
recomputation after each edge update at millisecond-scale cost in practice
[cite:sariyuce2013kcorestream]. The $k$-truss was defined by Cohen
[cite:cohen2008trusses]; Wang and Cheng studied truss community detection
[cite:wang2012truss], and Huang et al. proposed online maintenance of truss
communities [cite:huang2014trusscommunity].

*Distinction.* This line of work optimises *computation* — how to maintain
cores/trusses efficiently — and is not concerned with a downstream prediction
task. Hawkeye uses these mature incremental algorithms as a low-level compute
module and builds link-prediction-oriented structural features on top.

---

## References

All keys already exist in `references.bib`, including `zhang2025ipnet` and
`zhang2025tide` (added with the Introduction; ⚠ exact bibliographic details
still to be VERIFIED before camera-ready).
