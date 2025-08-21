Purpose
- Compute Phase‑1 metrics and composite loss; host an Evaluator for generation‑time scoring.

Exposed API
- Evaluator(dataset, metrics, retriever?, prompt_builder?, model_cfg?, gen_cfg?, use_rag=False, use_cot=False)
  - evaluate_checkpoint(checkpoint_dir, split="validation") -> Dict[str, float]
  - evaluate_argument_maps_single(gold_map, pred_map, threshold=0.7) -> Dict[str, float]
  - evaluate_argument_maps_batch(gold_maps, pred_maps, threshold=0.7) -> {average_metrics, per_map_results}
- metrics.py: fuzzy_match_f1, relation_f1_score, count_rmse, graph_edit_distance_metrics, combined_score
- metrics_modular.py: Phase1Metrics.compute(preds, golds) -> Dict[str, float]
- cost.py: WeightedCost.compute(metrics) -> float
- datasets.py: BaseDatasetAdapter.iter(split) -> yields (id, text, gold_struct)
- prompts.py: DefaultPromptBuilder.build(input_text, contexts, use_cot) -> str

Config usage
- snowball.phase_1.eval.fuzzy_thr (Phase‑1 threshold)
- snowball.phase_1.eval.loss.w (weights for combined cost)

Notes / TODO
- Retriever and prompt builder are pluggable; provide your own FAISS retriever to enable RAG.

