"""
TODO:
1. load in the Dataset via the Dataset class
-> Done and tested successfully
2. load in each pipeline:
 a. End2End model class
    -> Done and tested successfully
 b. ADUR model class + ARE model class
    -> Not yet finished. Both classes have KeyErrors while loading their pipelines.
3. For each pipeline, run generate()
    -> Implemented and untested for End2End
        -> Prompts need to be added to the prompt files in models/meta_llama_3.1_8B/prompts
    -> Not yet implemented for ADUR and ARE
4. Parse results into ArgumentMaps
    -> Implemented and untested for End2End
5. Load in the phase_1 evals class to compare generated maps to annotation maps
    -> Not yet implemented
"""
import rootutils
import json
from pathlib import Path
rootutils.setup_root(__file__, indicator=".git")
from moralkg import Config, get_logger
from moralkg.argmining.loaders import Dataset
from moralkg.argmining.models import End2End, ADUR, ARE
from moralkg.argmining.parsers import Parser # Parser for model outputs into argument maps.
from moralkg.argmining.schemas import ArgumentMap # Argument Map schema for storing complete argument structures.
from moralkg.snowball.phase_1.evals import Evaluator, ModelConfig, GenerationConfig # Unified evaluation functions for phase 1.

def main() -> None:
    Config.load()
    logger = get_logger("snowball")
    logger.info("Config loaded and logger initialized")
   
    # Load the dataset
    logger.info("Loading dataset...")
    dataset = Dataset()
    logger.info(f"Dataset loaded successfully!")
    logger.info(f"Metadata contains {len(dataset.metadata.ids)} papers")
    logger.info(f"Available metadata columns: {dataset.metadata.columns}")
    logger.info(f"Annotations loaded for {len(dataset.annotations.by_paper)} papers")
    logger.info(f"Total annotation maps: {len(dataset.annotations.all)}") # Loading papers now works successfully!
   
    # Load the model pipelines
    pipelines = {
        "end2end": End2End(),
        #"adur_roberta": ADUR(), # TODO: Fix roberta loading, since the pytorch-ie AutoPipeline approach doesn't seem to be compatible with roberta's lack of a taskmodule_config.json
        #"adur_sciarg": ADUR(use_model_2=True), TODO: The taskmodule for this model, "LabeledSpanExtractionByTokenClassificationTaskModule", is not listed in the regular pytorch-ie package; to recognize it, we'll probably need to import it from pie-modules, which we don't actually seem to be importing yet.
        #"are_roberta": ARE(), # TODO: Ensure that roberta loading works, since the taskmodule_config.json might be missing
        #"are_sciarg": ARE(use_model_2=True), TODO: This module has a taskmodule_type KeyError like the Roberta ones, so it might also be missing its config. Or it might be encountering this during the ADUR step.
    }
    
    # Initialize parser for converting outputs to ArgumentMaps
    parser = Parser()
    
    # Store outputs for each pipeline
    pipeline_outputs = {}
    
    for name, model in pipelines.items():
        logger.info(f"Running {name} pipeline")
        
        # Run generate() for End2End
        if name == "end2end":
            cfg = Config.load()
            
            # Load prompts from configured directory
            prompt_dir = Path(cfg.get("paths.snowball.phase_1.prompts.x_shot"))
            
            # Read system and user prompt files
            system_prompts = []
            user_prompts = []

            # Load system prompts and user prompts for different prompt strategies
            system_prompt_files = ["zero_shot_system", "one_shot_system", "few_shot_system"]
            user_prompt_files = ["zero_shot_user", "one_shot_user", "few_shot_user"]
            for sys_name, user_name in zip(system_prompt_files, user_prompt_files):
                sys_name += ".txt"
                user_name += ".txt"
                sys_path = prompt_dir / sys_name
                user_path = prompt_dir / user_name
                system_prompt = sys_path.read_text(encoding="utf-8")
                user_prompt = user_path.read_text(encoding="utf-8")
                system_prompts.append(system_prompt)
                user_prompts.append(user_prompt)

            logger.info(f"Loaded prompts from {prompt_dir}")


            # Process the data for each prompt
            # TODO: Make sure outputs are being saved separately for each set of prompts
            for system_prompt, user_prompt in zip(system_prompts, user_prompts):
                # Load the response format json schema to insert it in the system prompts
                json_schema = Path("/opt/extra/avijit/projects/moralkg/src/moralkg/argmining/schemas/argmining.json") # TODO: Get the path from config instead of hardcoding it
                if json_schema.exists():
                    json_schema = json.loads(json_schema.read_text(encoding="utf-8"))
                else:
                    logger.warning(f"Could not find response format json in {json_schema}")
                    continue

                # Replace the text "{json schema}" in the system prompt with the json schema
                system_prompt = system_prompt.replace("{json schema}", json.dumps(json_schema, indent=2))

                outputs = []
                
                # Process a sample of papers (limit to 5 for testing)
                sample_papers = dataset.metadata.ids[:5]
                logger.info(f"Processing {len(sample_papers)} papers with End2End pipeline")
                
                for paper_id in sample_papers:
                    try:
                        paper_text = dataset.get_paper(paper_id)
                        if paper_text is None:
                            logger.warning(f"Could not load paper text for {paper_id}")
                            continue
                        
                        logger.info(f"Processing paper {paper_id} (length: {len(paper_text)} chars)")
                        
                        # Get paper metadata for context
                        metadata = dataset.metadata[paper_id] or {}
                        title = metadata.get("title", "Unknown Title")
                        authors = metadata.get("authors", "Unknown Authors")
                        
                        # Prepare prompt files with paper content and metadata
                        prompt_files = {
                            "paper_text": paper_text,
                            "paper_id": paper_id,
                            "title": title,
                            "authors": authors
                        }

                        # Replace "{paper}" in the user prompt with the paper text
                        user_prompt = user_prompt.replace("{paper}", paper_text)

                        # Generate argument map using End2End model
                        result = model.generate(
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            prompt_files=prompt_files
                        )
                        
                        # Store raw output
                        output_data = {
                            "id": paper_id,
                            "text": result["text"],
                            "trace": result["trace"],
                            "metadata": metadata
                        }
                        outputs.append(output_data)
                        
                        logger.info(f"Generated output for {paper_id}: {len(result['text'])} chars")
                        
                    except Exception as e:
                        logger.error(f"Error processing paper {paper_id}: {e}")
                        continue
                
                # Parse outputs into ArgumentMaps
                parsed_maps = []
                for output in outputs:
                    try:
                        # Parse the generated text into an ArgumentMap
                        arg_map = parser.parse_text(output["text"], map_id=output["id"])
                        parsed_maps.append(arg_map)
                        logger.info(f"Parsed ArgumentMap for {output['id']}: {len(arg_map.adus) if hasattr(arg_map, 'adus') else 0} ADUs")
                    except Exception as e:
                        logger.error(f"Error parsing output for {output['id']}: {e}")
                        continue
                
                # Store results
                pipeline_outputs[name] = {
                    "raw_outputs": outputs,
                    "parsed_maps": parsed_maps
                }
                
            logger.info(f"Completed {name} pipeline: {len(outputs)} outputs, {len(parsed_maps)} parsed maps")
       
        # TODO: Run generate() for ADUR
        if name == "adur_roberta" or name == "adur_sciarg":
            outputs = []
            sample_papers = dataset.metadata.ids[:5]
            
            for paper_id in sample_papers:
                try:
                    paper_text = dataset.get_paper(paper_id)
                    if paper_text is None:
                        continue
                    
                    # Create temporary file for ADUR input (it expects file path)
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
                        tmp_file.write(paper_text)
                        tmp_path = tmp_file.name
                    
                    try:
                        result = model.generate(tmp_path)
                        result["id"] = paper_id
                        outputs.append(result)
                    finally:
                        Path(tmp_path).unlink(missing_ok=True)
                        
                except Exception as e:
                    logger.error(f"Error processing paper {paper_id} with {name}: {e}")
                    continue
            
            pipeline_outputs[name] = {"raw_outputs": outputs, "parsed_maps": []}
            
        # TODO: Run generate() for ARE  
        if name == "are_roberta" or name == "are_sciarg":
            outputs = []
            sample_papers = dataset.metadata.ids[:5]
            
            for paper_id in sample_papers:
                try:
                    paper_text = dataset.get_paper(paper_id)
                    if paper_text is None:
                        continue
                    
                    # Create temporary file for ARE input
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
                        tmp_file.write(paper_text)
                        tmp_path = tmp_file.name
                    
                    try:
                        result = model.generate(tmp_path)
                        result["id"] = paper_id
                        outputs.append(result)
                    finally:
                        Path(tmp_path).unlink(missing_ok=True)
                        
                except Exception as e:
                    logger.error(f"Error processing paper {paper_id} with {name}: {e}")
                    continue
            
            pipeline_outputs[name] = {"raw_outputs": outputs, "parsed_maps": []}
    
    # Load the evaluation class
    try:
        evaluator = Evaluator(ModelConfig(), GenerationConfig())
        logger.info("Loaded evaluation class")
        
        # Compare generated maps to annotation maps
        for name, results in pipeline_outputs.items():
            if results["parsed_maps"]:
                logger.info(f"Evaluating {name} pipeline with {len(results['parsed_maps'])} maps")
                # TODO: Implement evaluation comparison
                # evaluator.evaluate(results["parsed_maps"], dataset.annotations.all)
            else:
                logger.info(f"No parsed maps available for {name} pipeline evaluation")
                
    except Exception as e:
        logger.error(f"Error loading evaluator: {e}")

if __name__ == "__main__":
    main()