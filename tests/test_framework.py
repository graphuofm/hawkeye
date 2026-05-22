import random

import torch

from gev import GEVConfig, GraphEagleVision


def _stream(n_nodes=50, n_edges=400, seed=0):
    rng = random.Random(seed)
    return [(rng.randrange(n_nodes), rng.randrange(n_nodes), float(i)) for i in range(n_edges)]


def test_struct_only_forward_and_backward():
    torch.manual_seed(0)
    model = GraphEagleVision(GEVConfig(indicators=["degree", "core"], fusion_mode="struct_only",
                                       struct_dim=16, hidden_dim=32, output_dim=16, predictor_hidden=16,
                                       use_pairwise=True)).to("cpu")
    edges = _stream()
    for u, v, t in edges:
        s = model.predict_scores([u], [v])     # predict-then-update protocol
        assert s.shape == (1,)
        model.update_structure(u, v, t)
    # one training step over a batch of pos + neg
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    src = [e[0] for e in edges[:32]]
    dst = [e[1] for e in edges[:32]]
    neg = [random.randrange(50) for _ in range(32)]
    pos = model.predict_scores(src, dst)
    negs = model.predict_scores(src, neg)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.cat([pos, negs]), torch.cat([torch.ones_like(pos), torch.zeros_like(negs)])
    )
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
    assert model.feature_dim == 12               # 2 indicators * 6 stat groups (5 base + recency)
    assert model.pairwise_dim > 0


def test_stat_groups_and_trend_decays():
    m = GraphEagleVision(GEVConfig(indicators=["degree", "core"], stat_groups=["static"]))
    assert m.feature_dim == 2                     # just "current" * 2 indicators
    m = GraphEagleVision(GEVConfig(indicators=["degree", "core"], stat_groups=["dynamic"]))
    assert m.feature_dim == 10                    # 5 dynamic groups (ema/std/delta/max_change/recency) * 2
    m = GraphEagleVision(GEVConfig(indicators=["core"], trend_decays=[0.99, 0.999]))
    # 5 base + 2 trend + recency = 8 groups, 1 indicator
    assert m.feature_dim == 8
    for u, v, t in _stream(n_edges=60):
        m.update_structure(u, v, t)
    assert m.predict_scores([1, 2], [3, 4]).shape == (2,)


def test_no_pairwise():
    model = GraphEagleVision(GEVConfig(indicators=["degree", "core"], use_pairwise=False,
                                       struct_dim=8, hidden_dim=16, output_dim=8, predictor_hidden=8))
    for u, v, t in _stream(n_edges=80):
        model.update_structure(u, v, t)
    assert model.pairwise_dim == 0
    s = model.predict_scores([1, 2, 3], [4, 5, 6])
    assert s.shape == (3,)


def test_gated_fusion_with_fake_interaction():
    torch.manual_seed(0)
    inter_dim = 8
    model = GraphEagleVision(GEVConfig(indicators=["degree", "core", "triangle"], fusion_mode="gated",
                                       inter_dim=inter_dim, struct_dim=16, hidden_dim=32,
                                       output_dim=16, predictor_hidden=16))
    edges = _stream(n_edges=120)
    for u, v, t in edges:
        model.update_structure(u, v, t)
    src = [e[0] for e in edges[:16]]
    dst = [e[1] for e in edges[:16]]
    h_inter_src = torch.randn(16, inter_dim)
    h_inter_dst = torch.randn(16, inter_dim)
    s = model.predict_scores(src, dst, h_inter_src=h_inter_src, h_inter_dst=h_inter_dst)
    assert s.shape == (16,)
    assert model.fusion.last_gate is not None and model.fusion.last_gate.shape == (16, 16)


def test_pairwise_features_shape_and_signal():
    model = GraphEagleVision(GEVConfig(indicators=["degree", "core"], use_pairwise=True))
    # build a triangle 0-1-2 and a separate edge 3-4
    for (u, v) in [(0, 1), (1, 2), (2, 0), (3, 4)]:
        model.update_structure(u, v, 0.0)
    pf = model.pairwise_features([0, 0], [2, 4])
    assert pf.shape == (2, model.pairwise_dim)
    # (0,2) share a common neighbour (1); (0,4) do not
    cn_idx = 0  # FEATURE_NAMES[0] == "cn"
    assert pf[0, cn_idx] > pf[1, cn_idx]


def test_reset_structure():
    model = GraphEagleVision(GEVConfig(indicators=["degree", "core", "triangle"]))
    for u, v, t in _stream(n_edges=50):
        model.update_structure(u, v, t)
    assert model.graph.num_edges > 0
    model.reset_structure()
    assert model.graph.num_edges == 0
    assert model.stats.memory_usage_bytes == 0
