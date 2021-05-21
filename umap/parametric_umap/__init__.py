from logging import warn
from .stats_utils import correlation, covariance

try:
    from .parametric_umap import (
        ParametricUMAP,
        GradientClippedModel,
        load_ParametricUMAP,
    )

    # TODO: See how to work around this
    # __all__ = ["ParametricUMAP", "GradientClippedModel", "load_ParametricUMAP"]
except ImportError as e:
    warn(str(e))
