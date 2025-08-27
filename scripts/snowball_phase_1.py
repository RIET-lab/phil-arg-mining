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
        #"adur_roberta": ADUR(), # TODO: Fix roberta loading, since the pytorch-ie AutoPipeline approach doesn't seem to be compatible with roberta's lack of a taskmodule_config.json
        #"adur_sciarg": ADUR(use_model_2=True), TODO: The taskmodule for this model, "LabeledSpanExtractionByTokenClassificationTaskModule", is not listed in the regular pytorch-ie package; to recognize it, we'll probably need to import it from pie-modules, which we don't actually seem to be importing yet.
        #"are_roberta": ARE(), # TODO: Ensure that roberta loading works, since the taskmodule_config.json might be missing
        #"are_sciarg": ARE(use_model_2=True), TODO: This module has a taskmodule_type KeyError like the Roberta ones, so it might also be missing its config. Or it might be encountering this during the ADUR step.
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