import numpy as np

from gev.stats import RollingStatistics


def test_rolling_basic_dims_and_values():
    rs = RollingStatistics(num_indicators=2, decay=0.5)   # 5 base + recency = 6 groups
    assert rs.feature_dim == 12
    assert rs.group_names == ["current", "ema", "std", "delta", "max_change", "recency"]
    # unseen node -> zeros
    assert np.allclose(rs.get_features(99), np.zeros(12))

    rs.update(1, np.array([2.0, 0.0]))
    cur, ema, std, delta, mx, rec = np.split(rs.get_features(1), 6)
    assert np.allclose(cur, [2.0, 0.0])
    assert np.allclose(ema, [2.0, 0.0])      # first obs initialises ema
    assert np.allclose(std, [0.0, 0.0])
    assert np.allclose(delta, [0.0, 0.0])
    assert np.allclose(mx, [0.0, 0.0])
    assert np.allclose(rec, [0.0, 0.0])      # log1p(0)

    rs.update(1, np.array([4.0, 1.0]))
    cur, ema, std, delta, mx, rec = np.split(rs.get_features(1), 6)
    assert np.allclose(cur, [4.0, 1.0])
    assert np.allclose(ema, [3.0, 0.5])      # 0.5*2+0.5*4 ; 0.5*0+0.5*1
    assert np.allclose(std, [1.0, 0.5])      # var = 10-9 ; 0.5-0.25
    assert np.allclose(delta, [2.0, 1.0])
    assert np.allclose(mx, [2.0, 1.0])
    assert np.allclose(rec, [0.0, 0.0])      # both changed -> steps reset to 0

    rs.update(1, np.array([4.0, 1.0]))
    cur, ema, std, delta, mx, rec = np.split(rs.get_features(1), 6)
    assert np.allclose(delta, [0.0, 0.0])
    assert np.allclose(mx, [1.0, 0.5])       # decayed
    assert np.allclose(rec, [np.log1p(1.0), np.log1p(1.0)])   # no change -> steps += 1


def test_trend_decays():
    rs = RollingStatistics(num_indicators=1, decay=0.5, trend_decays=[0.9])  # 6 base + 1 trend = 7 groups
    assert rs.feature_dim == 7
    assert rs.group_names == ["current", "ema", "std", "delta", "max_change", "trend_0.9", "recency"]
    rs.update(0, np.array([1.0]))
    rs.update(0, np.array([5.0]))
    f = rs.get_features(0)
    # ema = 0.5*1+0.5*5 = 3 ; ema_0.9 = 0.9*1+0.1*5 = 1.4 ; trend = 3 - 1.4 = 1.6
    assert np.isclose(f[5], 3.0 - 1.4)


def test_batch_features():
    rs = RollingStatistics(num_indicators=1)
    rs.update(0, np.array([1.0]))
    rs.update(2, np.array([3.0]))
    b = rs.get_batch_features([0, 1, 2])
    assert b.shape == (3, 6)
    assert b[1].sum() == 0.0      # node 1 unseen
    assert b[0][0] == 1.0 and b[2][0] == 3.0


def test_reset():
    rs = RollingStatistics(2)
    rs.update(5, np.array([1.0, 2.0]))
    rs.reset()
    assert np.allclose(rs.get_features(5), np.zeros(rs.feature_dim))
    assert rs.memory_usage_bytes == 0
