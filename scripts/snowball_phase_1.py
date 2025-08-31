"""
Usage note: For End2End processing, prompts will have to be generated first using the prompt generation script: src/moralkg/snowball/phase_1/generate_llm_prompts.py

TODO:
1. load in the Dataset via the Dataset class
-> Done and tested successfully
2. load in each pipeline:
 a. End2End model class
    -> Done and tested successfully
 b. ADUR model class + ARE model class
    -> Not yet finished. Both classes have KeyErrors while loading their pipelines.
3. For each pipeline, run generate()
    -> End2End implementation completed for standard prompting, not yet implemented for CoT and/or RAG
    -> Not yet implemented for ADUR and ARE
4. Parse results into ArgumentMaps
    -> Implemented and untested for End2End
5. Load in the phase_1 evals class to compare generated maps to annotation maps
    -> Not yet implemented
"""
import rootutils
import json
import os
from pathlib import Path
from datetime import datetime
import tempfile
import shutil
import argparse
rootutils.setup_root(__file__, indicator=".git")
from moralkg import Config, get_logger
from moralkg.argmining.loaders import Dataset
from moralkg.argmining.models import End2End, ADUR, ARE

def load_prompts_from_directory(prompt_dir: Path, logger):
    """Load system and user prompts from a directory structure created by the prompt generation script."""
    all_prompts = []
    
    # Look for shot-type subdirectories (zero-shot, one-shot, few-shot)
    shot_subdirs = [d for d in prompt_dir.iterdir() if d.is_dir() and d.name.endswith('-shot')]
    
    if not shot_subdirs:
        # Fallback: look directly in the prompt_dir for prompt files
        logger.warning(f"No shot-type subdirectories found in {prompt_dir}, looking for files directly")
        shot_subdirs = [prompt_dir]
    
    for shot_dir in shot_subdirs:
        shot_type = shot_dir.name if shot_dir != prompt_dir else "direct"
        logger.info(f"Loading prompts from {shot_type} directory: {shot_dir}")
        
        # Find all system prompt files
        system_files = list(shot_dir.glob("system_prompt*.txt"))
        
        if not system_files:
            logger.warning(f"No system prompt files found in {shot_dir}")
            continue
        
        # Extract number from filename like "system_prompt_1.txt" or "system_prompt_1_zs2.txt".
        system_files.sort(key=lambda x: int(x.stem.split('_')[2]))

        # Find corresponding user prompt files
        user_files = list(shot_dir.glob("user_prompt*.txt"))
        
        if not user_files:
            logger.warning(f"No user prompt files found in {shot_dir}")
            continue
        
        # Sort user files similarly
        user_files.sort(key=lambda x: x.name)
        
        # If there's only one user prompt file, pair it with all system prompts
        if len(user_files) == 1:
            user_file = user_files[0]
            try:
                with open(user_file, 'r', encoding='utf-8') as f:
                    user_prompt = f.read().strip()
                
                # Pair this user prompt with each system prompt
                for sys_file in system_files:
                    try:
                        with open(sys_file, 'r', encoding='utf-8') as f:
                            system_prompt = f.read().strip()
                        
                        prompt_info = {
                            "system_prompt": system_prompt,
                            "user_prompt": user_prompt,
                            "shot_type": shot_type,
                            "system_file": sys_file.name,
                            "user_file": user_file.name,
                            "variation": "default" # Currently, variations are used to indicate which version of the zero-shot system prompt was inserted into the x-shot prompts
                        }

                        # Use the number of underscore-separated terms to check if this is a variation (has extra suffix after number)
                        # A typical file with a suffix will have 4 or more terms, e.g. "system_prompt_1_zs2.txt" > "system", "prompt", "1", "zs2"
                        underscore_terms = sys_file.stem.split('_')
                        if len(underscore_terms) > 3:
                            # Name the variation based on everything that follows the number
                            prompt_info["variation"] = '_'.join(underscore_terms[3:])

                        all_prompts.append(prompt_info)
                        logger.info(f"Loaded prompt pair: {sys_file.name} & {user_file.name}")
                        
                    except Exception as e:
                        logger.error(f"Error loading system prompt {sys_file}: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error loading user prompt {user_file}: {e}")
                continue
        
        else:
            # Multiple user prompts - pair them with system prompts by index
            min_count = min(len(system_files), len(user_files))
            
            for i in range(min_count):
                try:
                    with open(system_files[i], 'r', encoding='utf-8') as f:
                        system_prompt = f.read().strip()
                    with open(user_files[i], 'r', encoding='utf-8') as f:
                        user_prompt = f.read().strip()
                    
                    prompt_info = {
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "shot_type": shot_type,
                        "system_file": system_files[i].name,
                        "user_file": user_files[i].name,
                        "variation": "default"
                    }
                    
                    # Check if this is a variation
                    underscore_terms = system_files[i].stem.split('_')
                    if len(underscore_terms) > 3:
                        prompt_info["variation"] = '_'.join(underscore_terms[3:])

                    all_prompts.append(prompt_info)
                    logger.info(f"Loaded prompt pair: {system_files[i].name} & {user_files[i].name}")
                    
                except Exception as e:
                    logger.error(f"Error loading prompt pair {i}: {e}")
                    continue
    
    if not all_prompts:
        logger.warning(f"No valid prompt pairs found in {prompt_dir}")
        return []
    
    logger.info(f"Loaded {len(all_prompts)} total prompt pairs from {prompt_dir}")
    return all_prompts

def get_prompt_key(prompt_info):
    """Generate a unique key for a prompt configuration."""
    return f"{prompt_info['shot_type']}_{prompt_info['system_file']}_{prompt_info['user_file']}_{prompt_info['variation']}"

def save_outputs(outputs, output_dir: Path, prompt_strategy: str, logger):
    """Save generation outputs to the specified directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get shot types and variations used in this batch
    shot_types = set()
    variations = set()
    for output in outputs:
        if "prompt_info" in output:
            shot_types.add(output["prompt_info"]["shot_type"])
            variations.add(output["prompt_info"]["variation"])
    
    # Create descriptive filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shot_desc = "_".join(sorted(shot_types)) if shot_types else "unknown"
    var_desc = "_".join(sorted(variations)) if len(variations) > 1 else ""
    
    filename_parts = ["generation_outputs", prompt_strategy, shot_desc]
    if var_desc and var_desc != "default":
        filename_parts.append(var_desc)
    filename_parts.append(timestamp)
    
    output_file = output_dir / f"{'_'.join(filename_parts)}.json"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(outputs)} outputs to {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"Error saving outputs to {output_file}: {e}")
        return None

def save_individual_output(output_data, output_dir: Path, prompt_strategy: str, logger):
    """Save a single output immediately after generation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create individual output filename with paper ID and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
    paper_id = output_data.get("id", "unknown")
    shot_type = output_data.get("prompt_info", {}).get("shot_type", "unknown")
    variation = output_data.get("prompt_info", {}).get("variation", "default")
    
    filename_parts = ["individual", prompt_strategy, shot_type, paper_id]
    if variation != "default":
        filename_parts.append(variation)
    filename_parts.append(timestamp)
    
    output_file = output_dir / "individual_outputs" / f"{'_'.join(filename_parts)}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Saved individual output to {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"Error saving individual output to {output_file}: {e}")
        return None

def save_checkpoint(outputs, output_dir: Path, prompt_strategy: str, checkpoint_name: str, logger):
    """Save a checkpoint with current outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_file = output_dir / "checkpoints" / f"checkpoint_{prompt_strategy}_{checkpoint_name}_{timestamp}.json"
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint_data = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint_name": checkpoint_name,
        "strategy": prompt_strategy,
        "num_outputs": len(outputs),
        "outputs": outputs
    }
    
    try:
        # Write to temporary file first, then move to avoid corruption
        temp_file = checkpoint_file.with_suffix('.tmp')
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        
        # Atomic move
        shutil.move(temp_file, checkpoint_file)
        
        logger.info(f"Saved checkpoint '{checkpoint_name}' with {len(outputs)} outputs to {checkpoint_file}")
        return checkpoint_file
        
    except Exception as e:
        logger.error(f"Error saving checkpoint to {checkpoint_file}: {e}")
        # Clean up temp file if it exists
        if temp_file.exists():
            temp_file.unlink()
        return None

def load_existing_outputs(output_dir: Path, prompt_strategy: str, logger, skip_existing: bool = False):
    """Load existing outputs from previous runs to avoid reprocessing.
    
    Args:
        output_dir: Directory containing outputs
        prompt_strategy: Strategy name (e.g., "standard")
        logger: Logger instance  
        skip_existing: If True, don't load any existing outputs (forces reprocessing)
        
    Returns:
        tuple: (existing_outputs, processed_combinations)
        where processed_combinations is a set of (paper_id, prompt_key) tuples
    """
    if skip_existing:
        logger.info("Skipping existing outputs - will reprocess all prompt pairs")
        return [], set()
    
    existing_outputs = []
    processed_combinations = set()  # Set of (paper_id, prompt_key) tuples
    
    # Check for individual outputs
    individual_dir = output_dir / "individual_outputs"
    if individual_dir.exists():
        for individual_file in individual_dir.glob(f"individual_{prompt_strategy}_*.json"):
            try:
                with open(individual_file, 'r', encoding='utf-8') as f:
                    output_data = json.load(f)
                    existing_outputs.append(output_data)
                    
                    # Create key for this specific prompt configuration
                    paper_id = output_data.get("id")
                    if "prompt_info" in output_data:
                        prompt_key = get_prompt_key(output_data["prompt_info"])
                        processed_combinations.add((paper_id, prompt_key))
                        
            except Exception as e:
                logger.warning(f"Could not load existing output {individual_file}: {e}")
    
    # Check for checkpoints
    checkpoint_dir = output_dir / "checkpoints"
    if checkpoint_dir.exists():
        # Find the most recent checkpoint
        checkpoint_files = list(checkpoint_dir.glob(f"checkpoint_{prompt_strategy}_*.json"))
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
            try:
                with open(latest_checkpoint, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                    checkpoint_outputs = checkpoint_data.get("outputs", [])
                    
                    # Merge with individual outputs, avoiding duplicates
                    for output in checkpoint_outputs:
                        paper_id = output.get("id")
                        if "prompt_info" in output:
                            prompt_key = get_prompt_key(output["prompt_info"])
                            combination = (paper_id, prompt_key)
                            
                            if combination not in processed_combinations:
                                existing_outputs.append(output)
                                processed_combinations.add(combination)
                    
                    logger.info(f"Loaded {len(checkpoint_outputs)} outputs from latest checkpoint: {latest_checkpoint}")
            except Exception as e:
                logger.warning(f"Could not load checkpoint {latest_checkpoint}: {e}")
    
    if existing_outputs:
        logger.info(f"Found {len(existing_outputs)} existing outputs for {prompt_strategy}")
        logger.info(f"Already processed combinations: {len(processed_combinations)}")
        
        # Show some stats about what's been processed
        processed_papers = set(combo[0] for combo in processed_combinations)
        processed_prompts = set(combo[1] for combo in processed_combinations)
        logger.info(f"Processed papers: {len(processed_papers)}")
        logger.info(f"Processed prompt configurations: {len(processed_prompts)}")
    
    return existing_outputs, processed_combinations

def get_annotated_papers(dataset: Dataset, logger, limit: int = None) -> list:
    """Get list of paper IDs that have large gold-standard annotations.
    
    Args:
        dataset: The Dataset object containing metadata and annotations
        logger: Logger instance
        limit: Optional limit on number of papers to return (for testing)
        
    Returns:
        List of paper IDs that have annotations
    """
    annotated_paper_ids = list(dataset.annotations.by_paper.keys())
    
    if not annotated_paper_ids:
        logger.warning("No papers with annotations found in the dataset")
        return []
    
    logger.info(f"Found {len(annotated_paper_ids)} papers with annotations")
    
    # Apply limit if specified
    if limit is not None and limit > 0:
        limited_papers = annotated_paper_ids[:limit]
        logger.info(f"Limited to {len(limited_papers)} papers for processing")
        return limited_papers
    
    return annotated_paper_ids

def main(force_reprocess: bool = False, paper_limit: int = None) -> None:
    Config.load()
    logger = get_logger("snowball_phase_1")
    logger.info("Config loaded and logger initialized")

    if force_reprocess:
        logger.info("force_reprocess=True: Will reprocess all prompt pairs")
    else:
        logger.info("force_reprocess=False: Will skip already processed (paper, prompt) combinations")

    # Load the dataset
    logger.info("Loading dataset...")
    dataset = Dataset()
    logger.info(f"Dataset loaded successfully!")
    logger.info(f"Metadata contains {len(dataset.metadata.ids)} papers")
    logger.info(f"Available metadata columns: {dataset.metadata.columns}")
    logger.info(f"Annotations loaded for {len(dataset.annotations.by_paper)} papers")
    logger.info(f"Total annotation maps: {len(dataset.annotations.all)}")
   
    # Load the model pipelines
    pipelines = {
        "end2end": End2End(),
        #"adur_roberta": ADUR(), # TODO: Fix roberta loading, since the pytorch-ie AutoPipeline approach doesn't seem to be compatible with roberta's lack of a taskmodule_config.json
        #"adur_sciarg": ADUR(use_model_2=True), TODO: The taskmodule for this model, "LabeledSpanExtractionByTokenClassificationTaskModule", is not listed in the regular pytorch-ie package; to recognize it, we'll probably need to import it from pie-modules, which we don't actually seem to be importing yet.
        #"are_roberta": ARE(), # TODO: Ensure that roberta loading works, since the taskmodule_config.json might be missing
        #"are_sciarg": ARE(use_model_2=True), TODO: This module has a taskmodule_type KeyError like the Roberta ones, so it might also be missing its config. Or it might be encountering this during the ADUR step.
    }
    
    # Store all pipeline outputs
    all_pipeline_outputs = {}
    
    for pipeline_name, model in pipelines.items():
        logger.info(f"Running {pipeline_name} pipeline")
        
        # Run generate() for End2End
        if pipeline_name == "end2end":
            cfg = Config.load()
            
            # Define the prompt strategies to run
            #prompt_strategies = ["standard", "standard_rag", "cot", "rag_cot"] # TODO: Implement all prompt strategies, including by generating custom prompts
            prompt_strategies = ["standard"]

            for strategy in prompt_strategies:
                logger.info(f"Processing {strategy} prompting strategy")
                
                # Load prompts from the configured directory
                prompt_dir = Path(cfg.get(f"paths.snowball.phase_1.prompts.meta_llama_3.{strategy}"))
                
                if not prompt_dir.exists():
                    logger.warning(f"Prompt directory {prompt_dir} does not exist, skipping {strategy}")
                    continue
                
                prompt_pairs = load_prompts_from_directory(prompt_dir, logger)
                
                if not prompt_pairs:
                    logger.warning(f"No valid prompts found for {strategy}, skipping")
                    continue
                
                logger.info(f"Loaded {len(prompt_pairs)} prompt pairs for {strategy}")

                # Set up output directory and load existing outputs
                output_dir = Path(cfg.get(f"paths.snowball.phase_1.outputs.end2end.{strategy}"))
                existing_outputs, processed_combinations = load_existing_outputs(
                    output_dir, strategy, logger, force_reprocess
                )
                
                # Process each prompt pair
                strategy_outputs = existing_outputs.copy()  # Start with existing outputs
                
                # Get papers that have gold standard annotations
                annotated_papers = get_annotated_papers(dataset, logger, limit=paper_limit)
                
                if not annotated_papers:
                    logger.error("No annotated papers found to process")
                    continue
                
                logger.info(f"Processing {len(annotated_papers)} papers with existing annotations")
                
                total_combinations_possible = len(prompt_pairs) * len(annotated_papers)
                total_combinations_processed = len(processed_combinations)
                
                logger.info(f"Total possible combinations: {total_combinations_possible}")
                logger.info(f"Already processed combinations: {total_combinations_processed}")
                logger.info(f"Remaining combinations: {total_combinations_possible - total_combinations_processed}")
                
                for i, prompt_info in enumerate(prompt_pairs):
                    # For testing purposes, skip all variations that are not either "default" or "zs1". Revert this when the pipelines are complete in order to test more prompt strategies.
                    #if prompt_info["variation"] not in ["default", "zs1"]:
                    #    logger.warning(f"Skipping prompt pair {i+1}/{len(prompt_pairs)} for {strategy} due to variation")
                    #    continue

                    logger.info(f"Processing prompt pair {i+1}/{len(prompt_pairs)} for {strategy}")
                    logger.info(f"  Shot type: {prompt_info['shot_type']}")
                    logger.info(f"  Files: {prompt_info['system_file']} & {prompt_info['user_file']}")
                    logger.info(f"  Variation: {prompt_info['variation']}")
                    
                    system_prompt = prompt_info["system_prompt"]
                    user_prompt = prompt_info["user_prompt"]
                    prompt_key = get_prompt_key(prompt_info)
                    
                    prompt_outputs = []
                    
                    # Filter papers to only those not yet processed with this specific prompt
                    remaining_papers = [
                        paper_id for paper_id in annotated_papers 
                        if (paper_id, prompt_key) not in processed_combinations
                    ]
                    
                    logger.info(f"Papers remaining for this prompt: {len(remaining_papers)}/{len(annotated_papers)}")
                    
                    if not remaining_papers:
                        logger.info(f"All papers already processed for prompt pair {i+1}, skipping")
                        continue
                    
                    papers_processed_this_run = 0
                    
                    for paper_idx, paper_id in enumerate(remaining_papers):
                        try:
                            logger.info(f"Processing paper {paper_idx+1}/{len(remaining_papers)}: {paper_id}")
                            
                            paper_text = dataset.get_paper(paper_id)
                            if paper_text is None:
                                logger.warning(f"Could not load paper text for {paper_id}")
                                continue
                            
                            # Replace placeholder in the user prompt with the paper text
                            processed_user_prompt = user_prompt.replace("<paper text inserted here>", paper_text)

                            # Generate argument map using End2End model
                            result = model.generate(
                                system_prompt=system_prompt,
                                user_prompt=processed_user_prompt,
                                prompt_files=None
                            )
                            
                            # Store raw output with additional annotation metadata
                            output_data = {
                                "id": paper_id,
                                "text": result["text"],
                                "trace": result.get("trace", {}),
                                "prompt_info": {
                                    "pair_index": i + 1,
                                    "strategy": strategy,
                                    "shot_type": prompt_info["shot_type"],
                                    "system_file": prompt_info["system_file"],
                                    "user_file": prompt_info["user_file"],
                                    "variation": prompt_info["variation"]
                                },
                                "prompts_used": {
                                    "system_prompt": system_prompt,
                                    "user_prompt": processed_user_prompt
                                },
                                "timestamp": datetime.now().isoformat()
                            }
                            
                            # Save individual output immediately
                            save_individual_output(output_data, output_dir, strategy, logger)
                            
                            prompt_outputs.append(output_data)
                            strategy_outputs.append(output_data)
                            processed_combinations.add((paper_id, prompt_key))
                            papers_processed_this_run += 1
                            
                            logger.info(f"Generated output for {paper_id}: {len(result['text'])} chars")
                            
                            # Save checkpoint every 5 papers
                            if papers_processed_this_run % 5 == 0:
                                checkpoint_name = f"prompt_{i+1}_paper_{papers_processed_this_run}"
                                save_checkpoint(strategy_outputs, output_dir, strategy, checkpoint_name, logger)
                            
                        except Exception as e:
                            logger.error(f"Error processing paper {paper_id}: {e}")
                            continue
                    
                    # Save checkpoint after completing prompt pair
                    if papers_processed_this_run > 0:
                        checkpoint_name = f"prompt_{i+1}_completed"
                        save_checkpoint(strategy_outputs, output_dir, strategy, checkpoint_name, logger)
                    
                    logger.info(f"Completed prompt pair {i+1}: {len(prompt_outputs)} new outputs generated")
                
                # Save final outputs for this strategy
                output_file = save_outputs(strategy_outputs, output_dir, strategy, logger)
                
                # Store in overall results
                if output_file:
                    all_pipeline_outputs[f"end2end_{strategy}"] = {
                        "strategy": strategy,
                        "output_file": str(output_file),
                        "num_outputs": len(strategy_outputs),
                        "papers_processed": len(set(output["id"] for output in strategy_outputs)),
                        "annotated_papers_targeted": True,  # NEW: Flag to indicate we targeted annotated papers
                        "prompt_pairs": len(prompt_pairs),
                        "new_outputs_this_run": len(strategy_outputs) - len(existing_outputs),
                        "total_combinations_possible": len(prompt_pairs) * len(annotated_papers),
                        "combinations_processed": len(processed_combinations)
                    }
                
                logger.info(f"Completed {strategy} strategy: {len(strategy_outputs)} total outputs ({len(strategy_outputs) - len(existing_outputs)} new)")
       
        # TODO: Implement ADUR pipeline generation
        elif pipeline_name.startswith("adur"):
            logger.info(f"Skipping {pipeline_name} - implementation pending")
            
        # TODO: Implement ARE pipeline generation  
        elif pipeline_name.startswith("are"):
            logger.info(f"Skipping {pipeline_name} - implementation pending")
    
    # Save summary of all pipeline runs
    cfg = Config.load()
    summary_dir = Path(cfg.get("paths.snowball.phase_1.outputs.end2end.standard")).parent
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = summary_dir / f"pipeline_summary_{timestamp}.json"
    
    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "pipelines_run": list(all_pipeline_outputs.keys()),
        "total_papers_in_dataset": len(dataset.metadata.ids),
        "force_reprocess": force_reprocess,
        "paper_limit": paper_limit,
        "pipeline_results": all_pipeline_outputs
    }
    
    try:
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved pipeline summary to {summary_file}")
        
    except Exception as e:
        logger.error(f"Error saving pipeline summary: {e}")
    
    # Print final summary
    logger.info("="*50)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("="*50)
    
    for pipeline_name, results in all_pipeline_outputs.items():
        logger.info(f"{pipeline_name}:")
        logger.info(f"  Strategy: {results['strategy']}")
        logger.info(f"  Total outputs: {results['num_outputs']}")
        logger.info(f"  New outputs this run: {results.get('new_outputs_this_run', 'N/A')}")
        logger.info(f"  Papers processed: {results['papers_processed']}")
        logger.info(f"  Prompt pairs used: {results['prompt_pairs']}")
        logger.info(f"  Output file: {results['output_file']}")
        
        # Show combination statistics
        if 'combinations_processed' in results and 'total_combinations_possible' in results:
            combinations_processed = results['combinations_processed']
            total_combinations = results['total_combinations_possible']
            completion_pct = (combinations_processed / total_combinations) * 100 if total_combinations > 0 else 0
            logger.info(f"  Combinations processed: {combinations_processed}/{total_combinations} ({completion_pct:.1f}%)")
        
        # Show shot type distribution if available
        if 'num_outputs' in results and results['num_outputs'] > 0:
            # Try to read the output file to get shot type statistics
            try:
                with open(results['output_file'], 'r', encoding='utf-8') as f:
                    outputs_data = json.load(f)
                
                shot_types = {}
                variations = {}

                for output in outputs_data:
                    if "prompt_info" in output:
                        shot_type = output["prompt_info"]["shot_type"]
                        variation = output["prompt_info"]["variation"]
                        shot_types[shot_type] = shot_types.get(shot_type, 0) + 1
                        variations[variation] = variations.get(variation, 0) + 1
                
                if shot_types:
                    logger.info(f"  Shot type distribution: {dict(shot_types)}")
                if len(variations) > 1:
                    logger.info(f"  Variation distribution: {dict(variations)}")
                
            except Exception as e:
                logger.warning(f"Could not load output statistics: {e}")
        
        logger.info("")
    
    logger.info("Pipeline execution completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Snowball Phase 1 pipeline generation")
    parser.add_argument(
        "--force-reprocess", 
        action="store_true", # Defaults to False
        help="Skip loading existing outputs and reprocess all prompt pairs"
    )
    parser.add_argument(
        "--paper-limit", 
        type=int, 
        default=None,
        help="Limit the number of papers to process (for testing)"
    )
    
    args = parser.parse_args()
    main(force_reprocess=args.force_reprocess, paper_limit=args.paper_limit)