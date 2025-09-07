"""
Snowball Phase 1 End2End Pipeline Checkpoint Evaluator

This module handles:
1. Loading output checkpoint results from snowball_phase_1.py
2. Parsing generated text into ArgumentMap objects
3. Matching with gold standard annotations
4. Computing evaluation metrics

Usage:
    evaluator = End2EndEvaluator(dataset)
    results = evaluator.evaluate_checkpoint_dir(checkpoint_dir)
    
    # Or evaluate specific output files
    results = evaluator.evaluate_output_files([file1, file2, ...])
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from moralkg.argmining.parsers.parser import Parser as ModelOutputParser
from moralkg.argmining.schemas import ArgumentMap
from moralkg.argmining.loaders import Dataset
from moralkg.config import Config
from . import metrics as legacy_metrics
from .metrics_modular import Phase1Metrics


@dataclass
class EvaluationResult:
    """Results from evaluating a single paper's generated argument map."""
    paper_id: str
    prompt_info: Dict[str, Any]
    parse_success: bool
    parse_error: Optional[str]
    metrics: Dict[str, float]
    gold_map: Optional[ArgumentMap]
    pred_map: Optional[ArgumentMap]


@dataclass
class BatchEvaluationResult:
    """Aggregated results from evaluating multiple papers."""
    total_papers: int
    successful_parses: int
    failed_parses: int
    parse_success_rate: float
    aggregate_metrics: Dict[str, float]
    per_paper_results: List[EvaluationResult]
    strategy_breakdown: Dict[str, Dict[str, float]]  # Strategy -> metrics
    shot_type_breakdown: Dict[str, Dict[str, float]]  # Shot type -> metrics


class End2EndEvaluator:
    """
    Evaluator for snowball phase 1 checkpoint results.
    
    Handles loading generated outputs, parsing them into ArgumentMaps,
    and computing evaluation metrics against gold standard annotations.
    """
    
    def __init__(self, dataset: Dataset, fuzzy_threshold: float = None):
        self.dataset = dataset
        self.annotations = dataset._load_annotations()
        self.parser = ModelOutputParser()
        
        # Load threshold from config if not provided
        cfg = Config.load()
        self.fuzzy_threshold = (
            fuzzy_threshold if fuzzy_threshold is not None 
            else cfg.get("snowball.phase_1.eval.fuzzy_thr", 0.7)
        )
        
        self.logger = logging.getLogger(__name__)
    
    def evaluate_checkpoint_dir(self, checkpoint_dir: Path, 
                               strategy: str = "standard",
                               file_patterns: List[str] = None) -> BatchEvaluationResult:
        """
        Evaluate all outputs in a checkpoint directory.
        
        Args:
            checkpoint_dir: Directory containing output files
            strategy: Strategy name (e.g., "standard", "cot", etc.)
            file_patterns: Glob patterns to match files (default: ["*.json"])
            
        Returns:
            BatchEvaluationResult with aggregated metrics
        """
        if file_patterns is None:
            file_patterns = ["generation_outputs_*.json", "individual_*.json", "checkpoint_*.json"]
        
        output_files = []
        for pattern in file_patterns:
            output_files.extend(list(Path(checkpoint_dir).glob(pattern)))
            # Also check subdirectories
            output_files.extend(list(Path(checkpoint_dir).glob(f"**/{pattern}")))
        
        if not output_files:
            self.logger.warning(f"No output files found in {checkpoint_dir} matching patterns {file_patterns}")
            return BatchEvaluationResult(
                total_papers=0, successful_parses=0, failed_parses=0,
                parse_success_rate=0.0, aggregate_metrics={},
                per_paper_results=[], strategy_breakdown={}, shot_type_breakdown={}
            )
        
        self.logger.info(f"Found {len(output_files)} output files to evaluate")
        return self.evaluate_output_files(output_files)
    
    def evaluate_output_files(self, output_files: List[Path]) -> BatchEvaluationResult:
        """
        Evaluate a list of output files.
        
        Args:
            output_files: List of paths to JSON output files
            
        Returns:
            BatchEvaluationResult with aggregated metrics
        """
        all_outputs = []
        
        # Load all outputs from files
        for file_path in output_files:
            try:
                outputs_from_file = self._load_outputs_from_file(file_path)
                all_outputs.extend(outputs_from_file)
                self.logger.debug(f"Loaded {len(outputs_from_file)} outputs from {file_path}")
            except Exception as e:
                self.logger.error(f"Error loading {file_path}: {e}")
                continue
        
        if not all_outputs:
            self.logger.warning("No valid outputs found in any files")
            return BatchEvaluationResult(
                total_papers=0, successful_parses=0, failed_parses=0,
                parse_success_rate=0.0, aggregate_metrics={},
                per_paper_results=[], strategy_breakdown={}, shot_type_breakdown={}
            )
        
        self.logger.info(f"Loaded {len(all_outputs)} total outputs to evaluate")
        return self.evaluate_outputs(all_outputs)
    
    def evaluate_outputs(self, outputs: List[Dict[str, Any]]) -> BatchEvaluationResult:
        """
        Evaluate a list of output dictionaries.
        
        Args:
            outputs: List of output dictionaries from snowball pipeline
            
        Returns:
            BatchEvaluationResult with aggregated metrics
        """
        per_paper_results = []
        successful_parses = 0
        failed_parses = 0
        
        # Track breakdowns by strategy and shot type
        strategy_metrics = {}  # strategy -> list of metrics dicts
        shot_type_metrics = {}  # shot_type -> list of metrics dicts
        
        for output in outputs:
            result = self._evaluate_single_output(output)
            per_paper_results.append(result)
            
            if result.parse_success:
                successful_parses += 1
                
                # Group by strategy
                strategy = result.prompt_info.get("strategy", "unknown")
                if strategy not in strategy_metrics:
                    strategy_metrics[strategy] = []
                strategy_metrics[strategy].append(result.metrics)
                
                # Group by shot type
                shot_type = result.prompt_info.get("shot_type", "unknown")
                if shot_type not in shot_type_metrics:
                    shot_type_metrics[shot_type] = []
                shot_type_metrics[shot_type].append(result.metrics)
            else:
                failed_parses += 1
        
        # Calculate aggregate metrics
        successful_results = [r for r in per_paper_results if r.parse_success]
        aggregate_metrics = self._aggregate_metrics([r.metrics for r in successful_results])
        
        # Calculate breakdowns
        strategy_breakdown = {
            strategy: self._aggregate_metrics(metrics_list)
            for strategy, metrics_list in strategy_metrics.items()
        }
        
        shot_type_breakdown = {
            shot_type: self._aggregate_metrics(metrics_list)
            for shot_type, metrics_list in shot_type_metrics.items()
        }
        
        parse_success_rate = successful_parses / len(outputs) if outputs else 0.0
        
        return BatchEvaluationResult(
            total_papers=len(outputs),
            successful_parses=successful_parses,
            failed_parses=failed_parses,
            parse_success_rate=parse_success_rate,
            aggregate_metrics=aggregate_metrics,
            per_paper_results=per_paper_results,
            strategy_breakdown=strategy_breakdown,
            shot_type_breakdown=shot_type_breakdown
        )
    
    def _load_outputs_from_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load outputs from a JSON file, handling different file formats."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        outputs = []
        
        if isinstance(data, list):
            # Direct list of outputs
            outputs = data
        elif isinstance(data, dict):
            if "outputs" in data:
                # Checkpoint format
                outputs = data["outputs"]
            else:
                # Single output format
                outputs = [data]
        
        return outputs
    
    def _evaluate_single_output(self, output: Dict[str, Any]) -> EvaluationResult:
        """Evaluate a single output against gold standard."""
        paper_id = output.get("id", "unknown")
        prompt_info = output.get("prompt_info", {})
        generated_text = output.get("text", "")
        
        # Try to get gold standard annotation
        gold_map = None
        try:
            gold_map = self.dataset.annotations.by_paper.get(paper_id)
        except Exception as e:
            self.logger.warning(f"Could not load gold annotation for {paper_id}: {e}")
        
        # Parse generated text into ArgumentMap
        pred_map = None
        parse_success = False
        parse_error = None

        try:
            pred_map = self.parser.parse_string(generated_text, paper_id)
            parse_success = True
        except Exception as e:
            parse_error = str(e)
            #self.logger.warning(f"Parse failed for {paper_id}: {e}")
        
        # Calculate metrics if both maps are available
        metrics = {}
        if gold_map and pred_map and parse_success:
            try:
                metrics = legacy_metrics.combined_score(
                    gold_map, pred_map, threshold=self.fuzzy_threshold
                )
            except Exception as e:
                self.logger.error(f"Metrics calculation failed for {paper_id}: {e}")
                metrics = {}
        
        return EvaluationResult(
            paper_id=paper_id,
            prompt_info=prompt_info,
            parse_success=parse_success,
            parse_error=parse_error,
            metrics=metrics,
            gold_map=gold_map,
            pred_map=pred_map
        )
    
    def _aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate a list of metrics dictionaries."""
        if not metrics_list:
            return {}
        
        # Get all metric names
        all_keys = set()
        for metrics in metrics_list:
            all_keys.update(metrics.keys())
        
        # Calculate averages
        aggregated = {}
        for key in all_keys:
            values = [metrics.get(key, 0.0) for metrics in metrics_list]
            aggregated[f"avg_{key}"] = sum(values) / len(values)
            aggregated[f"std_{key}"] = (
                sum((v - aggregated[f"avg_{key}"]) ** 2 for v in values) / len(values)
            ) ** 0.5
        
        return aggregated
    
    def save_results(self, results: BatchEvaluationResult, output_path: Path) -> None:
        """Save evaluation results to a JSON file."""
        output_data = {
            "summary": {
                "total_papers": results.total_papers,
                "successful_parses": results.successful_parses,
                "failed_parses": results.failed_parses,
                "parse_success_rate": results.parse_success_rate,
            },
            "aggregate_metrics": results.aggregate_metrics,
            "strategy_breakdown": results.strategy_breakdown,
            "shot_type_breakdown": results.shot_type_breakdown,
            "per_paper_results": [
                {
                    "paper_id": r.paper_id,
                    "prompt_info": r.prompt_info,
                    "parse_success": r.parse_success,
                    "parse_error": r.parse_error,
                    "metrics": r.metrics,
                    # Don't save the full ArgumentMap objects to keep file size manageable
                }
                for r in results.per_paper_results
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved evaluation results to {output_path}")


def main():
    """Example usage of the End2EndEvaluator."""
    from moralkg import Config, get_logger
    
    # Setup
    Config.load()
    logger = get_logger("snowball_evaluator")
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = Dataset()
    
    # Create evaluator
    evaluator = End2EndEvaluator(dataset)

    # Example: evaluate outputs from a checkpoint directory
    cfg = Config.load()
    checkpoint_dir = Path(cfg.get("paths.snowball.phase_1.outputs.end2end.standard"))
    
    if checkpoint_dir.exists():
        logger.info(f"Evaluating checkpoint directory: {checkpoint_dir}")
        results = evaluator.evaluate_checkpoint_dir(checkpoint_dir, strategy="standard")
        
        # Print summary
        logger.info("="*50)
        logger.info("EVALUATION RESULTS SUMMARY")
        logger.info("="*50)
        logger.info(f"Total papers: {results.total_papers}")
        logger.info(f"Successful parses: {results.successful_parses}")
        logger.info(f"Failed parses: {results.failed_parses}")
        logger.info(f"Parse success rate: {results.parse_success_rate:.2%}")
        
        if results.aggregate_metrics:
            logger.info("Aggregate Metrics:")
            for metric, value in results.aggregate_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
        
        if results.strategy_breakdown:
            logger.info("Strategy Breakdown:")
            for strategy, metrics in results.strategy_breakdown.items():
                logger.info(f"  {strategy}:")
                for metric, value in metrics.items():
                    logger.info(f"    {metric}: {value:.4f}")
        
        # Save results
        output_file = checkpoint_dir / "evaluation_results.json"
        evaluator.save_results(results, output_file)
    
    else:
        logger.warning(f"Checkpoint directory not found: {checkpoint_dir}")


if __name__ == "__main__":
    main()