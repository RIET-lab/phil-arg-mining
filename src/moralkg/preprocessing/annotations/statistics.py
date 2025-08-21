"""
file that loads in the workshop sample annotations - workshop.annotations.use to get which directory to use
if large - just load in json files in paths.workshop.annotations.large_maps, if both, also load in paths.workshop.annotations.small_maps.
Then, embed models with paths.models.end2end.embedder.
Collect the following statistics:
1. Average token length of workshop annotations
2. average ADU token length (within each json schema)
3. average number of ADU candidates in a response.

Parse the json file and get the right keys with the Parser from src/moralkg/argmining/parsers/parser.py (from moral)

"""

from moralkg import Config, get_logger
from moralkg.argmining.parsers import Parser
