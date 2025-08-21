"""
Relation schema for argument mining.
"""

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class RelationType(str, Enum):
    """Types of relations between ADUs."""

    SUPPORT = "support"
    ATTACK = "attack"
    UNKNOWN = "unknown"


class Relation(BaseModel):
    """
    Relation between two ADUs in an argument map.
    """

    id: str = Field(..., description="Unique identifier for the relation")
    src: str = Field(..., description="ID of the source ADU")
    tgt: str = Field(..., description="ID of the target ADU")
    type: RelationType = Field(
        default=RelationType.UNKNOWN, description="Type of relationship between the source (src) and target (tgt) ADUs (support or attack)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional metadata for the relation"
    )
