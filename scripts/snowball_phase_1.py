"""
TODO:
1. load in the Dataset via the Dataset class
2. load in each pipeline:
 a. End2End model class
 b. ADUR model class + ARE model class
3. For each pipeline, run generate()
4. Parse results into ArgumentMaps
5. Load in the phase_1 evals class to compare generated maps to annotation maps
"""

import rootutils

rootutils.setup_root(__file__, indicator=".git")

from moralkg import Config, get_logger

from moralkg.argmining.loaders import Dataset

def main() -> None:
    Config.load()
    logger = get_logger("snowball")
    logger.info("Config loaded and logger initialized")
    
    # Load the dataset
    dataset = Dataset()
    logger.info(f"Loaded dataset with {len(dataset.metadata.ids)} papers")
    logger.info(f"Loaded {len(dataset.annotations.all)} annotation maps")
    dataset = Dataset()
    dataset.load()
    dataset.prepare()
    dataset.split()
    dataset.tokenize()
    dataset.create_dataloader()


if __name__ == "__main__":
    main()