"""
TODO:
1. load in the Dataset via the Dataset class
-> Done and tested successfully
2. load in each pipeline:
 a. End2End model class
    -> Done and tested successfully
 b. ADUR model class + ARE model class
    -> Not yet finished. Both classes need to load models using paths from config file.
3. For each pipeline, run generate()
4. Parse results into ArgumentMaps
5. Load in the phase_1 evals class to compare generated maps to annotation maps
"""

import rootutils

rootutils.setup_root(__file__, indicator=".git")

from moralkg import Config, get_logger
from moralkg.argmining.loaders import Dataset
from moralkg.argmining.models import End2End, ADUR, ARE, generate
from moralkg.argmining.parsers import parser # Parser for model outputs into argument maps.
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
        "adur": ADUR(), # TODO: Implement model selection to choose from ADUR's model_1 and model_2 from the config.yml 
        "are": ARE() # TODO: Implement model selection to choose from ARE's model_1 and model_2 from the config.yml
    }

    for name, model in pipelines.items():
        logger.info(f"Running {name} pipeline")
        model.generate(dataset.dataloader)
        # Parse results into ArgumentMaps
        for output in model.outputs:
            arg_map = parser.parse_dict(output, map_id=output.get("id", None))
            logger.info(f"Parsed ArgumentMap for {name} pipeline: {arg_map}")

    # Load the evaluation class
    evaluator = Evaluator(ModelConfig(), GenerationConfig())
    logger.info("Loaded evaluation class")

    # Compare generated maps to annotation maps
    for name, model in pipelines.items():
        logger.info(f"Evaluating {name} pipeline")
        evaluator.evaluate(model.outputs, dataset.annotations.all)

if __name__ == "__main__":
    main()