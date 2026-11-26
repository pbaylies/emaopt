"""Optional integrations with external tooling (guarded, off by default)."""

from .latentscope_bridge import (
    install_latentscope,
    is_available,
    probe_emaopt,
    run_emeans,
    run_umap,
)

__all__ = [
    "install_latentscope",
    "is_available",
    "probe_emaopt",
    "run_emeans",
    "run_umap",
]
