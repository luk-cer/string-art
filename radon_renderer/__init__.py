"""
Advanced Radon-Based Line Art Renderer
Pluggable metrics, multiple rendering modes, hierarchical optimization
"""

from .config import PAPER_SIZES
from .renderer import AdvancedRadonRenderer
from .metrics import (
    AngleSelectionMetric,
    RadonVarianceMetric,
    MSEReductionMetric,
    LineSpacingMetric,
    EvenSpacingMetric,
    AdaptiveSpacingMetric,
    MSEOptimizedSpacingMetric,
    QualityMetric,
    MSEQualityMetric,
    MAEQualityMetric,
)
from .cli import main
