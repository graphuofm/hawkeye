# Chapter 7 — Conclusion

> Project: **Hawkeye** — CIKM 2026 Full Research Paper.
> Target length: ~0.3 page, one paragraph. `$N$` filled after the sweep.

---

This paper studied the design of the **structure channel** in temporal link
prediction. We found that, on sparse temporal graphs, the 1-hop common-neighbour
signal that existing methods rely on is near-random as a discriminator
(discAUC $\approx 0.50$), whereas the **2-hop cohesive bridge** — a pair of
nodes joined through high-cohesion intermediaries two hops away — carries the
genuinely useful structural signal (discAUC $0.74$–$0.96$). Building on this
finding, we proposed **Hawkeye**, an incrementally maintained, cohesion-aware
structure channel built on the $k$-family of structural-cohesiveness indicators,
which plug-and-play replaces the native structure channel of any temporal-graph
model. Experiments on $N$ TGB and DGB datasets show that Hawkeye consistently
improves the state-of-the-art DyGFormer by $+2.8$–$4$ points on structurally
non-degenerate graphs. We further characterised Hawkeye's applicability
boundary: on near-complete graphs the cohesiveness indicators saturate and
Hawkeye no longer adds discriminative power — a predictable boundary that gives
practitioners an a-priori basis for deciding whether to use it. We release
Hawkeye as an open-source toolkit. Future directions include extending Hawkeye
to heterogeneous temporal graphs, exploring more efficient $k$-truss
approximations to cover larger graphs, and studying adaptive fusion mechanisms
between the structural and interaction signals.

---

## Open items

- [ ] Sentence 4: dataset count $N$ — fill after all experiments complete.
- [ ] Sentence 4: improvement range ($+2.8$–$4$) — update once the full sweep
      lands (may widen or narrow).
- [ ] "We release Hawkeye as an open-source toolkit" — kept (a GitHub
      description has been prepared); remove if the repo is not made public.
- [ ] Future-work directions — confirm/adjust.
