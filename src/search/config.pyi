"""Type stubs for SearchConfig system - generated first for type safety."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

@dataclass(frozen=True)
class RRFConfig:
    """Configuration for Reciprocal Rank Fusion algorithm."""
    k: int
    vector_weight: float
    bm25_weight: float

    def validate(self) -> None: ...

@dataclass(frozen=True)
class BoostConfig:
    """Configuration for all boost factors."""
    vendor: float
    doc_type: float
    recency: float
    entity: float
    topic: float

    def validate(self) -> None: ...

@dataclass(frozen=True)
class RecencyConfig:
    """Configuration for recency-based boosting thresholds."""
    very_recent_days: int
    recent_days: int

    def validate(self) -> None: ...

@dataclass(frozen=True)
class SearchConfig:
    """Master configuration for hybrid search system."""
    rrf: RRFConfig
    boosts: BoostConfig
    recency: RecencyConfig
    top_k_default: int
    min_score_default: float

    _instance: ClassVar[SearchConfig | None]

    def validate(self) -> None: ...

    @classmethod
    def from_env(cls) -> SearchConfig: ...

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> SearchConfig: ...

    @classmethod
    def get_instance(cls) -> SearchConfig: ...

    @classmethod
    def reset_instance(cls) -> None: ...

    def to_dict(self) -> dict[str, Any]: ...

def get_search_config() -> SearchConfig: ...
