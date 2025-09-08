"""
Evaluation metrics for argument mining.

Implements the evaluation metrics specified in the Phase 1 spec:
- Fuzzy-match F1 for ADU spans
- Relation-type F1 scores (macro-F1)
- Count RMSE for ADUs and relations
- Combined score
"""

import logging
import numpy as np
from typing import List, Tuple, Dict, Set, Any
from sklearn.metrics import f1_score, precision_score, recall_score
import networkx as nx
import multiprocessing
import time
import queue
from moralkg.argmining.schemas import ArgumentMap, ADU, Relation

# Set up logger for this module
logger = logging.getLogger(__name__)


def _compute_similarity_cache(gold_map: ArgumentMap, pred_map: ArgumentMap) -> Dict[Tuple[Any, Any], float]:
    """
    Precompute similarity scores between all gold and predicted ADUs.
    Returns a dictionary mapping (gold_id, pred_id) to similarity score.
    """
    logger.debug(f"Computing similarity cache for {len(gold_map.adus)} x {len(pred_map.adus)} ADU pairs")
    
    cache = {}
    for gold_adu in gold_map.adus:
        for pred_adu in pred_map.adus:
            ta = (gold_adu.text or "").strip()
            tb = (pred_adu.text or "").strip()
            if ta and tb:
                similarity = ADU.fuzzy_similarity(ta, tb)
            else:
                similarity = 0.0
            cache[(gold_adu.id, pred_adu.id)] = similarity
    
    logger.debug(f"Computed {len(cache)} similarity scores")
    return cache


def find_best_match(gold_adu: ADU, predicted_adus: List[ADU], 
                    similarity_cache: Dict[Tuple[Any, Any], float],
                    threshold: float = 0.7) -> Tuple[ADU, float]:
    """
    Find the best matching predicted ADU for a gold standard ADU using precomputed similarities.
    """
    best_match = None
    best_score = 0.0
    
    for pred_adu in predicted_adus:
        score = similarity_cache.get((gold_adu.id, pred_adu.id), 0.0)
        if score > best_score and score >= threshold:
            best_match = pred_adu
            best_score = score
            
    if best_match is None:
        logger.debug(f"No match found for gold ADU '{gold_adu.text[:50]}...' "
                    f"above threshold {threshold}")
    else:
        logger.debug(f"Best match for gold ADU '{gold_adu.text[:50]}...' "
                    f"is '{best_match.text[:50]}...' with score {best_score:.3f}")
            
    return best_match, best_score


def fuzzy_match_f1(gold_map: ArgumentMap, pred_map: ArgumentMap,
                  similarity_cache: Dict[Tuple[Any, Any], float],
                  threshold: float = 0.7) -> Dict[str, float]:
    """
    Calculate fuzzy-match F1 score for ADU spans using precomputed similarities.
    """
    logger.debug(f"Computing fuzzy-match F1 with threshold {threshold}")
    logger.debug(f"Gold ADUs: {len(gold_map.adus)}, Predicted ADUs: {len(pred_map.adus)}")
    
    # Keep track of matched ADUs to avoid double counting
    matched_gold_adus = set()
    matched_pred_adus = set()
    
    # For each gold ADU, find the best match in predicted ADUs
    total_matches = 0
    for gold_adu in gold_map.adus:
        best_match, score = find_best_match(
            gold_adu, 
            [p for p in pred_map.adus if p.id not in matched_pred_adus],
            similarity_cache,
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
    
    logger.debug(f"ADU matching results: {total_matches} matches, "
                f"P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
        
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def relation_f1_score(gold_map: ArgumentMap, pred_map: ArgumentMap,
                     similarity_cache: Dict[Tuple[Any, Any], float],
                     threshold: float = 0.7) -> Dict[str, float]:
    """
    Calculate relation-type F1 scores for support/attack edges using precomputed similarities.
    """
    logger.debug(f"Computing relation F1 scores")
    logger.debug(f"Gold relations: {len(gold_map.relations)}, "
                f"Predicted relations: {len(pred_map.relations)}")
    
    # Match predicted ADUs to gold ADUs using similarity cache
    adu_matches = {}  # Maps gold ADU IDs to predicted ADU IDs
    
    for gold_adu in gold_map.adus:
        best_match, score = find_best_match(gold_adu, pred_map.adus, similarity_cache, threshold)
        if best_match is not None:
            adu_matches[gold_adu.id] = best_match.id
    
    logger.debug(f"ADU matches for relation evaluation: {len(adu_matches)}")
    
    # Check if relations match between matched ADUs
    gold_relations = []  # (source, target, type) tuples for gold relations
    pred_relations = []  # (source, target, type) tuples for predicted relations
    
    for gold_rel in gold_map.relations:
        # Only consider relations between matched ADUs
        if (gold_rel.src in adu_matches and gold_rel.tgt in adu_matches):
            gold_relations.append((gold_rel.src, gold_rel.tgt, gold_rel.type))

    for pred_rel in pred_map.relations:
        # Find corresponding gold ADU IDs for the predicted relation
        source_gold_id = None
        target_gold_id = None

        for gold_id, pred_id in adu_matches.items():
            if pred_id == pred_rel.src:
                source_gold_id = gold_id
            if pred_id == pred_rel.tgt:
                target_gold_id = gold_id

        if source_gold_id and target_gold_id:
            pred_relations.append((source_gold_id, target_gold_id, pred_rel.type))
    
    logger.debug(f"Relations between matched ADUs - Gold: {len(gold_relations)}, "
                f"Predicted: {len(pred_relations)}")
    
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
        except Exception as e:
            logger.warning(f"Error calculating support F1: {e}")
            support_f1 = 0.0
        
        # Calculate attack F1 (treating 'attack' as positive class)
        gold_attack = [1 if t == "attack" else 0 for t in gold_types]
        pred_attack = [1 if t == "attack" else 0 for t in pred_types]
        
        try:
            attack_f1 = f1_score(gold_attack, pred_attack, zero_division=0)
        except Exception as e:
            logger.warning(f"Error calculating attack F1: {e}")
            attack_f1 = 0.0
        
        # Calculate macro F1 (average of support and attack F1)
        macro_f1 = (support_f1 + attack_f1) / 2
        
        logger.debug(f"Relation F1 scores - Support: {support_f1:.3f}, "
                    f"Attack: {attack_f1:.3f}, Macro: {macro_f1:.3f}")
    else:
        logger.debug("No gold relations found for F1 calculation")
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
    """
    # Count gold standard and predicted ADUs and relations
    gold_adu_count = len(gold_map.adus)
    pred_adu_count = len(pred_map.adus)
    
    gold_rel_count = len(gold_map.relations)
    pred_rel_count = len(pred_map.relations)
    
    logger.debug(f"Count comparison - ADUs: gold={gold_adu_count}, pred={pred_adu_count}, "
                f"Relations: gold={gold_rel_count}, pred={pred_rel_count}")
    
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
    
    logger.debug(f"RMSE scores - ADU: {adu_rmse:.3f}, Relation: {rel_rmse:.3f}, "
                f"Combined: {combined_rmse:.3f}, Scaled: {scaled_rmse:.3f}")
    
    return {
        "adu_rmse": float(adu_rmse),
        "relation_rmse": float(rel_rmse),
        "combined_rmse": float(combined_rmse),
        "scaled_rmse": float(scaled_rmse)
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
        src = getattr(rel, "src", None) or getattr(rel, "src_id", None) or getattr(rel, "source_id", None) or getattr(rel, "source", None)
        tgt = getattr(rel, "tgt", None) or getattr(rel, "tgt_id", None) or getattr(rel, "target_id", None) or getattr(rel, "target", None)
        if src is None or tgt is None:
            logger.warning(f"Relation missing source or target: src={src}, tgt={tgt}")
            continue
        if src not in G:
            G.add_node(src, text="")
        if tgt not in G:
            G.add_node(tgt, text="")
        G.add_edge(src, tgt, type=rel_type_str)
    
    logger.debug(f"Built graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
    return G


def graph_edit_distance_metrics(
    gold_map: ArgumentMap,
    pred_map: ArgumentMap,
    similarity_cache: Dict[Tuple[Any, Any], float],
    threshold: float = 0.7,
) -> Dict[str, float]:
    """
    Compute attributed Graph Edit Distance (GED) between gold and predicted maps using precomputed similarities.
    """
    logger.debug(f"Computing Graph Edit Distance with threshold {threshold}")
    
    Gg = _build_graph_from_map(gold_map)
    Gp = _build_graph_from_map(pred_map)
    
    logger.debug(f"Graph sizes - Gold: {len(Gg.nodes)} nodes, {len(Gg.edges)} edges; "
                f"Predicted: {len(Gp.nodes)} nodes, {len(Gp.edges)} edges")
    
    def _node_match(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        # Extract node IDs from the graph node attributes
        gold_id = a.get("id")
        pred_id = b.get("id")
        if gold_id is not None and pred_id is not None:
            return similarity_cache.get((gold_id, pred_id), 0.0) >= threshold
        else:
            # Fallback to direct comparison if IDs aren't available
            ta = (a.get("text") or "").strip()
            tb = (b.get("text") or "").strip()
            if not ta or not tb:
                return False
            return ADU.fuzzy_similarity(ta, tb) >= threshold
    
    def _edge_match(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        # Basic T/F: relation types must match (support vs. attack); unknown treated as support
        ta = (a.get("type") or "").strip().lower()
        tb = (b.get("type") or "").strip().lower()
        if ta == "unknown" or ta == "":
            ta = "support"
        if tb == "unknown" or tb == "":
            tb = "support"
        return ta == tb
    
    # Estimate max possible edit operations for normalization: delete all from gold then insert all from pred
    max_ops = float(len(Gg.nodes) + len(Gg.edges) + len(Gp.nodes) + len(Gp.edges))

    start_time = time.time()
    try:
        ged_val = nx.optimize_graph_edit_distance(Gg, Gp, node_match=_node_match, edge_match=_edge_match, upper_bound=max_ops)

        # In some versions, this may return a generator; ensure we take the first (best) value
        if ged_val is None:
            logger.warning("GED algorithm returned None, defaulting to max cost")
            # Fallback to default cost if algorithm couldn't compute
            ged_val = max_ops
        elif not isinstance(ged_val, (int, float)):
            # Assume iterable
            try:
                ged_val = float(next(iter(ged_val)))
                logger.debug("Extracted GED value from generator")
            except Exception as e:
                logger.warning(f"Error extracting GED from generator: {e}, defaulting to max cost")
                ged_val = max_ops

        elapsed_time = time.time() - start_time
        logger.debug(f"GED computation completed in {elapsed_time:.2f}s, value: {ged_val}")
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.warning(f"GED computation failed after {elapsed_time:.2f}s: {e}, defaulting to max cost")
        ged_val = max_ops

    # Normalize by an upper-bound on edit ops
    max_ops = max(max_ops, 1.0) # Avoid division by zero
    ged_norm = float(min(max(ged_val / max_ops, 0.0), 1.0))
    ged_sim = float(1.0 - ged_norm)
    
    logger.debug(f"GED metrics - Raw: {ged_val}, Normalized: {ged_norm:.3f}, "
                f"Similarity: {ged_sim:.3f}")
    
    return {
        "ged": float(ged_val),
        "ged_norm": ged_norm,
        "ged_sim": ged_sim,
    }

def combined_score(gold_map: ArgumentMap, pred_map: ArgumentMap, 
                  threshold: float = 0.7) -> Dict[str, float]:
    """
    Calculate combined score based on all metrics.
    """
    logger.info(f"Computing combined score with threshold {threshold}")
    
    # Precompute similarity cache once for all metrics
    logger.debug("Precomputing similarity cache...")
    similarity_cache = _compute_similarity_cache(gold_map, pred_map)
    
    # Calculate all component metrics using the shared cache
    logger.debug("Computing ADU fuzzy-match F1...")
    adu_metrics = fuzzy_match_f1(gold_map, pred_map, similarity_cache, threshold)
    
    logger.debug("Computing relation F1 scores...")
    relation_metrics = relation_f1_score(gold_map, pred_map, similarity_cache, threshold)
    
    logger.debug("Computing count RMSE...")
    count_metrics = count_rmse(gold_map, pred_map)
    
    # Calculate GED metrics with hard timeout using multiprocessing
    logger.debug("Computing GED metrics...")
    ged_metrics = None

    try:
        ged_metrics = graph_edit_distance_metrics(gold_map, pred_map, similarity_cache, threshold)
    except Exception as e:
        logger.warning(f"GED computation failed: {e}, using fallback values")
        ged_metrics = {
            "ged": float(gold_map.adus) + float(pred_map.adus) + float(gold_map.relations) + float(pred_map.relations), # max cost
            "ged_norm": 1.0,
            "ged_sim": 0.0,
        }

    # Extract key scores
    adu_f1 = adu_metrics["f1"]
    relation_f1 = relation_metrics["macro_f1"]
    scaled_rmse = count_metrics["scaled_rmse"]
    ged_sim = ged_metrics["ged_sim"]
    
    # Calculate combined score (higher is better)
    # Equal-weight average across span F1, relation F1, inverted count error, and GED similarity
    combined = (adu_f1 + relation_f1 + (1 - scaled_rmse) + ged_sim) / 4
    
    logger.info(f"Combined score calculation complete: "
               f"ADU_F1={adu_f1:.3f}, REL_F1={relation_f1:.3f}, "
               f"COUNT_SCORE={1-scaled_rmse:.3f}, GED_SIM={ged_sim:.3f}, "
               f"COMBINED={combined:.3f}")
    
    # Return all metrics
    return {
        **adu_metrics,
        **relation_metrics,
        **count_metrics,
        **ged_metrics,
        "combined_score": combined,
    }