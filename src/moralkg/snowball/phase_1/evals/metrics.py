"""
Evaluation metrics for argument mining.

Implements the evaluation metrics specified in the Phase 1 spec:
- Fuzzy-match F1 for ADU spans
- Relation-type F1 scores (macro-F1)
- Count RMSE for ADUs and relations
- Combined score
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Any
from sklearn.metrics import f1_score, precision_score, recall_score
import networkx as nx
from moralkg.argmining.schemas import ArgumentMap, ADU, Relation


def find_best_match(gold_adu: ADU, predicted_adus: List[ADU], 
                    threshold: float = 0.7) -> Tuple[ADU, float]:
    """
    Find the best matching predicted ADU for a gold standard ADU.
    
    Args:
        gold_adu: Gold standard ADU
        predicted_adus: List of predicted ADUs
        threshold: Minimum token overlap threshold for a match
        
    Returns:
        Tuple of (best_match, overlap_score), or (None, 0.0) if no match found
    """
    best_match = None
    best_score = 0.0
    
    for pred_adu in predicted_adus:
        score = ADU.fuzzy_similarity(gold_adu.text, pred_adu.text)
        if score > best_score and score >= threshold:
            best_match = pred_adu
            best_score = score
            
    return best_match, best_score


def fuzzy_match_f1(gold_map: ArgumentMap, pred_map: ArgumentMap, 
                  threshold: float = 0.7) -> Dict[str, float]:
    """
    Calculate fuzzy-match F1 score for ADU spans.
    
    Args:
        gold_map: Gold standard argument map
        pred_map: Predicted argument map
        threshold: Minimum token overlap threshold for a match
        
    Returns:
        Dictionary with precision, recall, and F1 scores
    """
    # Keep track of matched ADUs to avoid double counting
    matched_gold_adus = set()
    matched_pred_adus = set()
    
    # For each gold ADU, find the best match in predicted ADUs
    total_matches = 0
    for gold_adu in gold_map.adus:
        best_match, score = find_best_match(
            gold_adu, 
            [p for p in pred_map.adus if p.id not in matched_pred_adus],
            threshold
        )
        
        if best_match is not None:
            matched_gold_adus.add(gold_adu.id)
            matched_pred_adus.add(best_match.id)
            total_matches += 1
    
    # Calculate precision and recall
    precision = total_matches / len(pred_map.adus) if pred_map.adus else 0.0
    recall = total_matches / len(gold_map.adus) if gold_map.adus else 0.0
    
    # Calculate F1 score
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
        
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def relation_f1_score(gold_map: ArgumentMap, pred_map: ArgumentMap,
                     threshold: float = 0.7) -> Dict[str, float]:
    """
    Calculate relation-type F1 scores for support/attack edges.
    
    Args:
        gold_map: Gold standard argument map
        pred_map: Predicted argument map
        threshold: Minimum token overlap threshold for ADU matching
        
    Returns:
        Dictionary with macro-F1 score and type-specific F1 scores
    """
    # First, match predicted ADUs to gold ADUs
    adu_matches = {}  # Maps gold ADU IDs to predicted ADU IDs
    
    for gold_adu in gold_map.adus:
        best_match, score = find_best_match(gold_adu, pred_map.adus, threshold)
        if best_match is not None:
            adu_matches[gold_adu.id] = best_match.id
    
    # Check if relations match between matched ADUs
    gold_relations = []  # (source, target, type) tuples for gold relations
    pred_relations = []  # (source, target, type) tuples for predicted relations
    
    for gold_rel in gold_map.relations:
        # Only consider relations between matched ADUs
        if (gold_rel.src_id in adu_matches and gold_rel.tgt_id in adu_matches):
            gold_relations.append((gold_rel.src_id, gold_rel.tgt_id, gold_rel.type))
    
    for pred_rel in pred_map.relations:
        # Find corresponding gold ADU IDs for the predicted relation
        source_gold_id = None
        target_gold_id = None

        for gold_id, pred_id in adu_matches.items():
            if pred_id == pred_rel.src_id:
                source_gold_id = gold_id
            if pred_id == pred_rel.tgt_id:
                target_gold_id = gold_id

        if source_gold_id and target_gold_id:
            pred_relations.append((source_gold_id, target_gold_id, pred_rel.type))
    
    # Create gold and predicted relation type arrays for scoring
    gold_types = ["support" if rel[2] == "support" else "attack" for rel in gold_relations]
    pred_types = []
    
    # For each gold relation, find corresponding predicted relation
    for gold_rel in gold_relations:
        found = False
        for pred_rel in pred_relations:
            if gold_rel[0] == pred_rel[0] and gold_rel[1] == pred_rel[1]:
                pred_types.append("support" if pred_rel[2] == "support" else "attack")
                found = True
                break
        if not found:
            pred_types.append("none")  # No matching relation found
    
    # Calculate F1 scores if there are any gold relations
    if len(gold_types) > 0:
        # Calculate support F1 (treating 'support' as positive class)
        gold_support = [1 if t == "support" else 0 for t in gold_types]
        pred_support = [1 if t == "support" else 0 for t in pred_types]
        
        try:
            support_f1 = f1_score(gold_support, pred_support, zero_division=0)
        except:
            support_f1 = 0.0
        
        # Calculate attack F1 (treating 'attack' as positive class)
        gold_attack = [1 if t == "attack" else 0 for t in gold_types]
        pred_attack = [1 if t == "attack" else 0 for t in pred_types]
        
        try:
            attack_f1 = f1_score(gold_attack, pred_attack, zero_division=0)
        except:
            attack_f1 = 0.0
        
        # Calculate macro F1 (average of support and attack F1)
        macro_f1 = (support_f1 + attack_f1) / 2
    else:
        support_f1 = 0.0
        attack_f1 = 0.0
        macro_f1 = 0.0
    
    return {
        "macro_f1": macro_f1,
        "support_f1": support_f1,
        "attack_f1": attack_f1
    }


def count_rmse(gold_map: ArgumentMap, pred_map: ArgumentMap) -> Dict[str, float]:
    """
    Calculate RMSE on the total number of spans and relations.
    
    Args:
        gold_map: Gold standard argument map
        pred_map: Predicted argument map
        
    Returns:
        Dictionary with RMSE values and scaled RMSE (0-1)
    """
    # Count gold standard and predicted ADUs and relations
    gold_adu_count = len(gold_map.adus)
    pred_adu_count = len(pred_map.adus)
    
    gold_rel_count = len(gold_map.relations)
    pred_rel_count = len(pred_map.relations)
    
    # Calculate squared errors
    adu_squared_error = (gold_adu_count - pred_adu_count) ** 2
    rel_squared_error = (gold_rel_count - pred_rel_count) ** 2
    
    # Calculate RMSE values
    adu_rmse = np.sqrt(adu_squared_error)
    rel_rmse = np.sqrt(rel_squared_error)
    
    # Combined RMSE for both ADUs and relations
    combined_rmse = np.sqrt((adu_squared_error + rel_squared_error) / 2)
    
    # Scale RMSE to [0, 1] range for optimization (0 = perfect)
    # We use a simple scaling function: 1 - exp(-rmse)
    # This gives diminishing penalty for larger errors
    scaled_rmse = 1.0 - np.exp(-combined_rmse / max(1, (gold_adu_count + gold_rel_count) / 2))
    
    return {
        "adu_rmse": float(adu_rmse),
        "relation_rmse": float(rel_rmse),
        "combined_rmse": float(combined_rmse),
        "scaled_rmse": float(scaled_rmse)
    }




def combined_score(gold_map: ArgumentMap, pred_map: ArgumentMap, 
                  threshold: float = 0.7) -> Dict[str, float]:
    """
    Calculate combined score based on all metrics.
    
    Args:
        gold_map: Gold standard argument map
        pred_map: Predicted argument map
        threshold: Token overlap threshold for fuzzy matching
        
    Returns:
        Dictionary with all metrics and a combined score
    """
    # Calculate all component metrics
    adu_metrics = fuzzy_match_f1(gold_map, pred_map, threshold)
    relation_metrics = relation_f1_score(gold_map, pred_map, threshold)
    count_metrics = count_rmse(gold_map, pred_map)
    ged_metrics = graph_edit_distance_metrics(gold_map, pred_map, threshold)
    
    # Extract key scores
    adu_f1 = adu_metrics["f1"]
    relation_f1 = relation_metrics["macro_f1"]
    scaled_rmse = count_metrics["scaled_rmse"]
    ged_sim = ged_metrics["ged_sim"]
    
    # Calculate combined score (higher is better)
    # Equal-weight average across span F1, relation F1, inverted count error, and GED similarity
    combined = (adu_f1 + relation_f1 + (1 - scaled_rmse) + ged_sim) / 4
    
    # Return all metrics
    return {
        **adu_metrics,
        **relation_metrics,
        **count_metrics,
        **ged_metrics,
        "combined_score": combined,
    }


def _build_graph_from_map(argument_map: ArgumentMap) -> nx.DiGraph:
    """
    Construct a directed graph from an ArgumentMap with node/edge attributes.
    Nodes carry 'text' attribute; edges carry 'type' attribute (support/attack/unknown).
    """
    G = nx.DiGraph()
    # Nodes
    for adu in argument_map.adus:
        # Ensure text fallback from quote if needed
        text_value = getattr(adu, "text", None) or getattr(adu, "quote", "")
        G.add_node(adu.id, text=text_value)
    # Edges
    for rel in argument_map.relations:
        rel_type_obj = getattr(rel, "type", "unknown")
        # Normalize relation type to a simple lowercase string like "support" | "attack" | "unknown"
        if hasattr(rel_type_obj, "value"):
            rel_type_str = str(getattr(rel_type_obj, "value")).lower()
        else:
            rel_type_str = str(rel_type_obj).lower()
        src = getattr(rel, "src_id", None) or getattr(rel, "source_id", None) or getattr(rel, "source", None)
        tgt = getattr(rel, "tgt_id", None) or getattr(rel, "target_id", None) or getattr(rel, "target", None)
        if src is None or tgt is None:
            continue
        if src not in G:
            G.add_node(src, text="")
        if tgt not in G:
            G.add_node(tgt, text="")
        G.add_edge(src, tgt, type=rel_type_str)
    return G


def graph_edit_distance_metrics(
    gold_map: ArgumentMap,
    pred_map: ArgumentMap,
    threshold: float = 0.7,
) -> Dict[str, float]:
    """
    Compute attributed Graph Edit Distance (GED) between gold and predicted maps.

    Node match: token-overlap of node texts >= threshold.
    Edge match: exact relation-type equality.

    Returns:
      - ged: raw edit distance (lower is better)
      - ged_norm: normalized distance in [0,1]
      - ged_sim: similarity = 1 - ged_norm (higher is better)
    """
    Gg = _build_graph_from_map(gold_map)
    Gp = _build_graph_from_map(pred_map)

    def _node_match(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        ta = (a.get("text") or "").strip()
        tb = (b.get("text") or "").strip()
        if not ta or not tb:
            return False

        return ADU.fuzzy_similarity(ta, tb) >= threshold

    def _edge_match(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        # Basic T/F: relation types must match (support vs. attack); unknown treated as its own label
        ta = (a.get("type") or "").strip().lower()
        tb = (b.get("type") or "").strip().lower()
        return ta == tb

    try:
        ged_val = nx.graph_edit_distance(Gg, Gp, node_match=_node_match, edge_match=_edge_match)
        # In some versions, this may return a generator; ensure we take the first (best) value
        if ged_val is None:
            # Fallback to default cost if algorithm couldn't compute
            ged_val = float(len(Gg.nodes) + len(Gg.edges) + len(Gp.nodes) + len(Gp.edges))
        elif not isinstance(ged_val, (int, float)):
            # Assume iterable
            try:
                ged_val = float(next(iter(ged_val)))
            except Exception:
                ged_val = float(len(Gg.nodes) + len(Gg.edges) + len(Gp.nodes) + len(Gp.edges))
    except Exception:
        ged_val = float(len(Gg.nodes) + len(Gg.edges) + len(Gp.nodes) + len(Gp.edges))

    # Normalize by an upper-bound on edit ops: delete all from gold then insert all from pred
    max_ops = float(len(Gg.nodes) + len(Gg.edges) + len(Gp.nodes) + len(Gp.edges))
    max_ops = max(max_ops, 1.0)
    ged_norm = float(min(max(ged_val / max_ops, 0.0), 1.0))
    ged_sim = float(1.0 - ged_norm)

    return {
        "ged": float(ged_val),
        "ged_norm": ged_norm,
        "ged_sim": ged_sim,
    }
