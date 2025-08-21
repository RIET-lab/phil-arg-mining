"""
Argument Map schema for storing complete argument structures.

Updated to match the new argmining.json schema with equivalence classes
and simplified ADU types (Major Claim and Claim only).
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .adu import ADU
from .relation import Relation


class ArgumentMap(BaseModel):
    """
    Complete argument map structure containing ADUs and their relations.
    ADU IDs represent equivalence class titles in the new schema.
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
        """
        Get an ADU by its ID (equivalence class title).
        
        Args:
            adu_id: ID of the ADU (equivalence class title)
            
        Returns:
            ADU object if found, None otherwise
        """
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
            adu_id: ID of the ADU (equivalence class title)
            as_source: If True, find relations where ADU is source; if False, where ADU is target

        Returns:
            List of relations
        """
        if as_source:
            return [r for r in self.relations if r.src == adu_id]
        else:
            return [r for r in self.relations if r.tgt == adu_id]

    def to_dict(self) -> Dict[str, Any]:
        """Convert the argument map to a dictionary representation matching the schema."""
        # Convert to schema format
        adus_dict = {}
        for adu in self.adus:
            adu_data = {
                "type": adu.type,  # Should be "Major Claim" or "Claim"
                "text": adu.text
            }
            
            # Add optional fields if present
            if hasattr(adu, 'quote') and adu.quote:
                adu_data["quote"] = adu.quote
            if hasattr(adu, 'isImplicit'):
                adu_data["isImplicit"] = adu.isImplicit
                
            adus_dict[adu.id] = adu_data
        
        # Convert relations to schema format
        relations_list = []
        for relation in self.relations:
            relations_list.append({
                "src": relation.src,
                "tgt": relation.tgt,
                "type": relation.type  # Should be "support" or "attack"
            })
        
        return {
            "ADUs": adus_dict,
            "relations": relations_list
        }

    def map_statistics(self) -> Dict[str, Any]:
        """
        Generate summary statistics about the argument map.
        Updated for new schema with only Major Claim and Claim types.
        """
        major_claim_count = sum(1 for adu in self.adus if adu.type == "Major Claim")
        claim_count = sum(1 for adu in self.adus if adu.type == "Claim")
        support_count = sum(1 for r in self.relations if r.type == "support")
        attack_count = sum(1 for r in self.relations if r.type == "attack")

        return {
            "total_adus": len(self.adus),
            "major_claims": major_claim_count,
            "claims": claim_count,
            "total_relations": len(self.relations),
            "support_relations": support_count,
            "attack_relations": attack_count,
        }
    
    def get_root_claims(self) -> List[ADU]:
        """
        Get all Major Claims (root nodes/top-level claims).
        
        Returns:
            List of ADUs with type "Major Claim"
        """
        return [adu for adu in self.adus if adu.type == "Major Claim"]
    
    def get_supporting_claims(self, adu_id: str) -> List[ADU]:
        """
        Get all ADUs that support the given ADU.
        
        Args:
            adu_id: ID of the target ADU
            
        Returns:
            List of ADUs that have support relations targeting the given ADU
        """
        supporting_relations = [
            r for r in self.relations 
            if r.tgt == adu_id and r.type == "support"
        ]
        
        supporting_adus = []
        for rel in supporting_relations:
            adu = self.get_adu_by_id(rel.src)
            if adu:
                supporting_adus.append(adu)
                
        return supporting_adus
    
    def get_attacking_claims(self, adu_id: str) -> List[ADU]:
        """
        Get all ADUs that attack the given ADU.
        
        Args:
            adu_id: ID of the target ADU
            
        Returns:
            List of ADUs that have attack relations targeting the given ADU
        """
        attacking_relations = [
            r for r in self.relations 
            if r.tgt == adu_id and r.type == "attack"
        ]
        
        attacking_adus = []
        for rel in attacking_relations:
            adu = self.get_adu_by_id(rel.src)
            if adu:
                attacking_adus.append(adu)
                
        return attacking_adus