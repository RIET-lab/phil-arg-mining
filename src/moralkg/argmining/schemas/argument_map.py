"""
Argument Map schema for storing complete argument structures.

TODO: revisit to confirm this matches the updated schema and is valid.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .adu import ADU
from .relation import Relation


class ArgumentMap(BaseModel):
    """
    Complete argument map structure containing ADUs and their relations.
    """

    id: str = Field(
        ..., description="Unique identifier for the argument map (paper identifier)"
    )
    adus: List[ADU] = Field(default_factory=list, description="List of ADUs in the map")
    relations: List[Relation] = Field(
        default_factory=list, description="List of relations between ADUs"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Metadata associated with the source text"
    )

    def get_adu_by_id(self, adu_id: str) -> Optional[ADU]:
        """Get an ADU by its ID."""
        for adu in self.adus:
            if adu.id == adu_id:
                return adu
        return None

    def get_relations_for_adu(
        self, adu_id: str, as_source: bool = True
    ) -> List[Relation]:
        """
        Get all relations where the specified ADU is either source or target.

        Args:
            adu_id: ID of the ADU
            as_source: If True, find relations where ADU is source; if False, where ADU is target

        Returns:
            List of relations
        """
        if as_source:
            return [r for r in self.relations if r.source_id == adu_id]
        else:
            return [r for r in self.relations if r.target_id == adu_id]

    def to_dict(self) -> Dict[str, Any]:
        """Convert the argument map to a dictionary representation."""
        return {
            "id": self.id,
            "adus": [adu.model_dump() for adu in self.adus],
            "relations": [relation.model_dump() for relation in self.relations],
            "metadata": self.metadata,
        }

    def map_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics about the argument map."""
        claim_count = sum(1 for adu in self.adus if adu.type == "claim")
        premise_count = sum(1 for adu in self.adus if adu.type == "premise")
        support_count = sum(1 for r in self.relations if r.type == "support")
        attack_count = sum(1 for r in self.relations if r.type == "attack")

        return {
            "total_adus": len(self.adus),
            "claims": claim_count,
            "premises": premise_count,
            "total_relations": len(self.relations),
            "support_relations": support_count,
            "attack_relations": attack_count,
        }
