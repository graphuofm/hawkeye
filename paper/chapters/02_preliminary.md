# Chapter 2 — Preliminary (discussion draft)

> Project: **Hawkeye** — CIKM 2026 Full Research Paper.
> Target length: 0.6–0.8 page. Citations in `[cite:key]` → `\cite{key}`.
> All references for this chapter already exist in `references.bib`.

This section formalises (i) the temporal-link-prediction task, (ii) the
$k$-family of structural-cohesiveness indicators that Hawkeye maintains, and
(iii) the *2-hop cohesive bridge*, the core structural signal of our method.

---

## 2.1 Problem Formulation

**Definition 1 (Continuous-time dynamic graph).**
A continuous-time dynamic graph is a time-ordered stream of edges
$$
\mathcal{G} = \{(u_1,v_1,t_1),(u_2,v_2,t_2),\dots,(u_m,v_m,t_m)\},
\tag{1}
$$
where $u_i,v_i\in\mathcal{V}$ are nodes, $t_i\in\mathbb{R}^{+}$ are timestamps,
and $t_1\le t_2\le\dots\le t_m$. The cumulative snapshot at time $t$, denoted
$\mathcal{G}(t)$, is the (simple, undirected) graph induced by all edges with
timestamp $\le t$; we write $\mathcal{G}(<t)$ for the strictly-earlier history.

**Definition 2 (Temporal link prediction).**
Given the history $\mathcal{G}(<t)$, a query node $u$, and a set of candidate
destinations $\mathcal{C}$, the task is to assign a score $s(u,v,t)$ to every
$v\in\mathcal{C}$ such that the edge $(u,v^{\*},t)$ that truly occurs at time
$t$ is ranked highest:
$$
v^{\*} \;=\; \arg\max_{v\in\mathcal{C}}\; s(u,v,t).
\tag{2}
$$
Following the Temporal Graph Benchmark [cite:huang2023tgb], each positive edge
is evaluated against a fixed set of negative destinations, and accuracy is
reported as the **mean reciprocal rank (MRR)**; on the DGB datasets
[cite:poursafaei2022edgebank] we additionally report average precision (AP).

---

## 2.2 The $k$-family of Structural-Cohesiveness Indicators

Hawkeye measures how *densely embedded* a node is using the classical
$k$-family of graph-cohesiveness decompositions. These indicators form a
hierarchy of **progressively tighter constraints**.

**Definition 3 ($k$-core).**
A subgraph $H\subseteq\mathcal{G}(t)$ is a $k$-core iff every node in $H$ has
degree $\ge k$ within $H$, and $H$ is the maximal such subgraph. The
**core number** of node $v$ is the largest $k$ whose $k$-core contains $v$:
$$
c(v) \;=\; \max\{\,k : v\in k\text{-core}(\mathcal{G}(t))\,\}.
\tag{3}
$$
Intuitively, a large $c(v)$ means $v$ sits in a tight region: not only does $v$
have many neighbours, its neighbours must themselves be well connected
[cite:seidman1983kcore,cite:batagelj2003cores].

**Definition 4 ($k$-truss).**
A subgraph $H\subseteq\mathcal{G}(t)$ is a $k$-truss iff every edge of $H$ is
supported by at least $k-2$ triangles within $H$, and $H$ is the maximal such
subgraph. The **trussness** of node $v$ is the largest truss number among its
incident edges:
$$
\tau(v) \;=\; \max\{\,k : \exists\,(v,w)\in k\text{-truss}(\mathcal{G}(t))\,\}.
\tag{4}
$$
A large $\tau(v)$ is a stricter requirement than a large core number: $v$'s
neighbours must additionally be connected *to each other*, closing triangles
[cite:cohen2008trusses,cite:wang2012truss].

**Property 1 (Nesting).**
The $k$-family is nested:
$$
k\text{-truss} \;\subseteq\; (k-1)\text{-core},
\tag{5}
$$
i.e. any $k$-truss is contained in the $(k-1)$-core. The indicators thus form a
loose-to-tight spectrum
$$
\text{degree} \;\longrightarrow\; k\text{-core} \;\longrightarrow\; k\text{-truss}.
$$
A tighter constraint describes finer structure but costs more to maintain: a
degree update is $O(1)$ per edge; an incremental $k$-core update is
$O(\text{local})$ per edge and millisecond-scale in practice
[cite:sariyuce2013kcorestream]; an incremental $k$-truss update is up to
$O(m^{1.5})$ in the worst case and can be slow on dense graphs
[cite:huang2014trusscommunity].

In plain terms, the three indicators answer increasingly strict questions:

| indicator | what it asks |
|---|---|
| degree | "do you have neighbours?" |
| $k$-core | "do your neighbours also have enough neighbours?" |
| $k$-truss | "do your neighbours also know each other?" |

---

## 2.3 The 2-hop Cohesive Bridge

The core structural signal Hawkeye exploits is a length-2 notion of structural
proximity between a candidate pair.

**Definition 5 (2-hop neighbourhood).**
The 2-hop neighbourhood of node $u$ at time $t$ is the set of nodes reachable
from $u$ in exactly two steps:
$$
\mathcal{N}_2(u,t) = \{\, w : \exists\, z,\;(u,z)\in\mathcal{G}(t)\,\wedge\,
(z,w)\in\mathcal{G}(t),\; w\neq u \,\}.
$$

**Definition 6 (2-hop cohesive bridge).**
For a candidate pair $(u,v)$, the 2-hop cohesive-bridge strength is the number
of $v$'s direct neighbours that lie in $u$'s 2-hop neighbourhood:
$$
\mathrm{bridge}(u,v,t) \;=\;
\bigl|\{\, w\in\mathcal{N}(v,t) : w\in\mathcal{N}_2(u,t) \,\}\bigr|.
\tag{6}
$$
Intuitively, even when $u$ and $v$ share *no* direct common neighbour
($\mathrm{CN}=0$), a large $\mathrm{bridge}(u,v,t)$ means many of $v$'s
neighbours are two steps from $u$ — the pair is embedded in the same structural
community but has not yet connected directly.

**Cohesion-weighted variant.**
We also weight each bridging node by its cohesiveness:
$$
\mathrm{bridge}_c(u,v,t) \;=\!\!
\sum_{w\,\in\,\mathcal{N}(v,t)\,\cap\,\mathcal{N}_2(u,t)} \!\! f\bigl(c(w)\bigr),
$$
where $c(w)$ is the core number of the bridging node and $f$ is the identity
map in our implementation. This up-weights bridges that pass through highly
cohesive intermediate nodes. As reported in our empirical study (Section 3 /
Introduction ¶4), on current benchmarks the weighted variant
$\mathrm{bridge}_c$ and the unweighted $\mathrm{bridge}$ have similar
single-feature discriminability ($\mathrm{cn2}\approx\mathrm{cn2\_x\_sum}$); we
therefore expose both as feature dimensions and let the model learn their
relative importance.

---

## 2.4 Relation to the 1-hop Common Neighbour

The classical 1-hop common-neighbour count,
$$
\mathrm{CN}(u,v,t) \;=\; \bigl|\mathcal{N}(u,t)\cap\mathcal{N}(v,t)\bigr|,
$$
is among the strongest heuristics for *static* link prediction
[cite:liben2007linkpred]. As shown in the Introduction, however, on sparse
temporal graphs $\mathrm{CN}$ is near-random (discAUC $\approx 0.50$): the
graph is too sparse for two arbitrary nodes to share a direct neighbour. The
2-hop cohesive bridge of Definition 6 is precisely the signal designed to
recover discriminative structure where the 1-hop count fails.

<!-- ============================================================= -->
<!-- Definition 7 (windowed snapshot) — to be added here IF the      -->
<!-- sliding-window ablation shows it belongs in the method.         -->
<!-- ============================================================= -->

---

## Equation index

| Eq | content |
|---|---|
| (1) | continuous-time dynamic graph $\mathcal{G}$ |
| (2) | temporal-link-prediction task |
| (3) | $k$-core / core number $c(v)$ |
| (4) | $k$-truss / trussness $\tau(v)$ |
| (5) | nesting $k$-truss $\subseteq(k-1)$-core |
| (6) | 2-hop cohesive bridge $\mathrm{bridge}(u,v,t)$ |

## Open items

- [ ] Figure 2 (10-node schematic: core-shading + a $k$-truss box + a 2-hop
      bridge between two starred nodes) — include only if the page budget allows.
- [ ] Keep §2.4 or fold it into the Introduction — decide at layout time.
- [ ] $f$ in $\mathrm{bridge}_c$ = identity (matches the code); stated as such.
- [ ] Definition 7 (windowed snapshot) — add only if sliding window enters the
      method, pending the window ablation.
