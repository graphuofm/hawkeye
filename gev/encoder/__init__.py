from gev.encoder.gru import GRUEncoder
from gev.encoder.mlp import IdentityEncoder, MLPEncoder


def build_encoder(encoder_type: str, input_dim: int, num_indicators: int, **kw):
    et = encoder_type.lower()
    if et in ("mlp", "default"):
        return MLPEncoder(input_dim, **kw)
    if et in ("identity", "none", "raw"):
        return IdentityEncoder(input_dim)
    if et == "gru":
        # GRU consumes the raw K-dim sequence, not the 5K rolling features
        return GRUEncoder(num_indicators, **kw)
    raise KeyError(f"unknown encoder_type {encoder_type!r}")


__all__ = ["MLPEncoder", "GRUEncoder", "IdentityEncoder", "build_encoder"]
