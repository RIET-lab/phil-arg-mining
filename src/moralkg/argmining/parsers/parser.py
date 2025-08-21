"""
Parser for model outputs in argument mining tasks.

This module provides functionality to parse model responses into
standardized ArgumentMap objects.
"""

import re
import json
import uuid
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from ..schemas import ArgumentMap, ADU, Relation, ADUType, RelationType, SpanPosition


class Parser:
    """
    Parser for model output in JSON format.
    
    Converts model responses into ArgumentMap objects based on the
    defined model response schema.
    """
    
    def __init__(self, schema_path: Optional[str] = None, log_level: str = "INFO"):
        """
        Initialize the model output parser.
        
        Args:
            schema_path: Path to the JSON schema file (optional)
            log_level: Logging level
        """
        self._setup_logging(log_level)
        
        # Load schema if provided
        self.schema_path = schema_path
        if schema_path:
            try:
                with open(schema_path, 'r') as f:
                    self.schema = json.load(f)
                self.logger.info(f"Loaded schema from {schema_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load schema: {e}")
                self.schema = None
        else:
            self.schema = None
    
    def _setup_logging(self, log_level: str):
        """Set up logging configuration."""
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            numeric_level = logging.INFO
            
        logging.basicConfig(
            level=numeric_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("ModelOutputParser")
    
    def parse_json_file(self, file_path: Union[str, Path], source_text: Optional[str] = None) -> ArgumentMap:
        """
        Parse a JSON file containing model output into an ArgumentMap.
        
        Args:
            file_path: Path to the JSON file
            source_text: Optional source text for ADU position resolution
            
        Returns:
            ArgumentMap object
        """
        file_path = Path(file_path)
        self.logger.info(f"Parsing model output from file: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                model_output = json.load(f)
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse JSON from {file_path}: {e}")
                raise
        
        # Use the file stem as the map ID
        map_id = file_path.stem
        return self.parse_dict(model_output, map_id, source_text)
    
    def parse_string(self, json_str: str, map_id: Optional[str] = None, 
                    source_text: Optional[str] = None) -> ArgumentMap:
        """
        Parse a JSON string containing model output into an ArgumentMap.
        
        Args:
            json_str: JSON string containing model output
            map_id: Optional ID for the argument map (generated if not provided)
            source_text: Optional source text for ADU position resolution
            
        Returns:
            ArgumentMap object
        """
        try:
            model_output = json.loads(json_str)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON string: {e}")
            self.logger.debug(f"JSON string: {json_str[:100]}...")
            raise
        
        if map_id is None:
            map_id = str(uuid.uuid4())
            
        return self.parse_dict(model_output, map_id, source_text)
    
    def parse_dict(self, model_output: Dict[str, Any], map_id: str, 
                  source_text: Optional[str] = None) -> ArgumentMap:
        """
        Parse a dictionary containing model output into an ArgumentMap.
        
        Args:
            model_output: Dictionary containing model output
            map_id: ID for the argument map
            source_text: Optional source text for ADU position resolution
            
        Returns:
            ArgumentMap object
        """
        # Validate structure (basic checks)
        if "ADUs" not in model_output:
            self.logger.error("Model output missing required 'ADUs' field")
            raise ValueError("Invalid model output: missing 'ADUs' field")
        
        # Initialize ADUs and relations lists
        adus = []
        relations = []
        
        # Process ADUs
        for adu_id, adu_data in model_output["ADUs"].items():
            # Extract ADU data with defaults
            text = adu_data.get("text", "")
            
            # Handle quote/content field
            # If quote is provided, use it as content, otherwise use text as content
            content = adu_data.get("quote", text)
            
            # Map ADU type
            type_str = adu_data.get("type", "unknown").lower()
            if type_str == "argument":
                adu_type = ADUType.ARGUMENT
            elif type_str == "claim":
                adu_type = ADUType.CLAIM
            elif type_str == "premise":
                adu_type = ADUType.PREMISE
            else:
                adu_type = ADUType.UNKNOWN
            
            # Extract positions if available
            positions = []
            if "positions" in adu_data and adu_data["positions"]:
                for pos in adu_data["positions"]:
                    if "start" in pos and "end" in pos:
                        positions.append(SpanPosition(start=pos["start"], end=pos["end"]))
            
            # Create ADU object
            adu = ADU(
                id=adu_id,
                text=text,
                content=content,
                type=adu_type,
                positions=positions,
                metadata=adu_data.get("metadata", {})
            )
            
            # If source text is provided, initialize ADU with it to resolve positions
            if source_text:
                adu.init(source_text)
                
            adus.append(adu)
        
        # Process relations
        if "relations" in model_output and isinstance(model_output["relations"], list):
            for idx, rel_data in enumerate(model_output["relations"]):
                # Extract relation fields - handle both src/tgt and source/target formats
                src_id = rel_data.get("src", rel_data.get("source", ""))
                tgt_id = rel_data.get("tgt", rel_data.get("target", ""))
                
                if not src_id or not tgt_id:
                    self.logger.warning(f"Skipping relation with missing source or target: {rel_data}")
                    continue
                
                # Map relation type
                type_str = rel_data.get("type", "unknown").lower()
                if type_str == "support":
                    rel_type = RelationType.SUPPORT
                elif type_str == "attack":
                    rel_type = RelationType.ATTACK
                else:
                    rel_type = RelationType.UNKNOWN
                
                # Create relation object
                relation = Relation(
                    id=f"rel-{idx+1}",
                    src_id=src_id,
                    tgt_id=tgt_id,
                    type=rel_type,
                    metadata=rel_data.get("metadata", {})
                )
                relations.append(relation)
        
        # Extract metadata
        metadata = model_output.get("metadata", {})
        
        # Create argument map
        argument_map = ArgumentMap(
            id=map_id,
            adus=adus,
            relations=relations,
            source_text=source_text,
            metadata=metadata,
            source_metadata={} # No source metadata from model response
        )
        
        self.logger.info(
            f"Parsed argument map with {len(adus)} ADUs and {len(relations)} relations"
        )
        return argument_map
    
    def extract_from_text(self, text: str, allow_partial: bool = True) -> Dict[str, Any]:
        """
        Extract JSON from text that may contain other content.
        
        This handles cases where the model outputs additional text before/after the JSON.
        
        Args:
            text: Text that may contain JSON
            allow_partial: Whether to allow partial JSON extraction
            
        Returns:
            Extracted JSON as a dictionary
        """
        # Try to find JSON block using regex patterns
        json_patterns = [
            r'```json\s*(.*?)\s*```',  # Code block with json syntax
            r'```\s*(.*?)\s*```',       # Any code block
            r'\{.*\}'                   # Just find outer braces
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                for match in matches:
                    try:
                        return json.loads(match)
                    except json.JSONDecodeError:
                        continue
        
        if allow_partial:
            # Try to find outermost JSON object
            try:
                start = text.find('{')
                end = text.rfind('}')
                if start >= 0 and end > start:
                    json_substr = text[start:end+1]
                    return json.loads(json_substr)
            except json.JSONDecodeError:
                pass
        
        # If we got here, no valid JSON found
        raise ValueError("Could not extract valid JSON from text")
    
    def parse_model_response(self, response: str, map_id: Optional[str] = None,
                           source_text: Optional[str] = None) -> ArgumentMap:
        """
        Parse a raw model response that may contain JSON within text.
        
        Args:
            response: Raw model response text
            map_id: Optional ID for the argument map
            source_text: Optional source text for ADU position resolution
            
        Returns:
            ArgumentMap object
        """
        try:
            # Try to extract JSON from the response
            model_output = self.extract_from_text(response)
            return self.parse_dict(model_output, map_id or str(uuid.uuid4()), source_text)
        except Exception as e:
            self.logger.error(f"Failed to parse model response: {e}")
            raise
    
    def save_to_json(self, argument_map: ArgumentMap, output_path: Union[str, Path]):
        """
        Save an ArgumentMap to a JSON file.
        
        Args:
            argument_map: ArgumentMap to save
            output_path: Path to save the JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to JSON string using json.dumps
        json_str = json.dumps(argument_map.to_dict(), indent=2)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
            
        self.logger.info(f"Saved argument map to {output_path}")
