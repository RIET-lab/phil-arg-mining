""" 
This should first load in the metadata for philpapers by getting it from config.paths.philpapers.metadata. if the path is a directory, gets most recent file. if the path is a file, loads that. Either way, there should be a metadata attribute that links paper metadata to the paper contents. There should be metadata 'sub-attributes' that correspond with the metadata columns which make it simple to access specific metadata. metadata should be structured where the key is the paper ID and values are the other columns. 

Papers should be dynamically grabbed with get_paper() and loaded in from the path declared at config.paths.philpapers.docling.cleaned. It should look for a file like <metadata.id>.md then <metadata.id>.txt. No paper should raise an error.

Annotations should be batch loaded in during the init and tied to papers. what annotations to load in should be decided by config.workshop.annotations.use. if "large", just use config.paths.workshop.annotations.large_maps, if "both" also use config.paths.workshop.annotations.small_maps.

Maps should be parsed into an ArgumentMap (see from moralkg.argmining.schemas import ArgumentMap, ADU, Relation). Parsing should be done with from moralkg.argmining.parsers import Parser.
"""

class Dataset:
    def __init__(self):
        self.metadata
        self.annotations
