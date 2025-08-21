"""
Parser for model outputs in argument mining tasks.

This module provides functionality to parse model responses into
standardized ArgumentMap objects based on the new schema with
equivalence classes and simplified ADU types.
"""

import re
import json
import uuid
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from ..schemas import ArgumentMap, ADU, Relation


class Parser:
    """
    Parser for model output in JSON format.
    
    Converts model responses into ArgumentMap objects based on the
    updated model response schema with equivalence classes.
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
    
    def parse_json_file(self, file_path: Union[str, Path]) -> ArgumentMap:
        """
        Parse a JSON file containing model output into an ArgumentMap.
        
        Args:
            file_path: Path to the JSON file
            
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
        return self.parse_dict(model_output, map_id)
    
    def parse_string(self, json_str: str, map_id: Optional[str] = None) -> ArgumentMap:
        """
        Parse a JSON string containing model output into an ArgumentMap.
        
        Args:
            json_str: JSON string containing model output
            map_id: Optional ID for the argument map (generated if not provided)
            
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
            
        return self.parse_dict(model_output, map_id)
    
    def parse_dict(self, model_output: Dict[str, Any], map_id: str) -> ArgumentMap:
        """
        Parse a dictionary containing model output into an ArgumentMap.
        
        Args:
            model_output: Dictionary containing model output
            map_id: ID for the argument map
            
        Returns:
            ArgumentMap object
        """
        # Validate structure (basic checks)
        if "ADUs" not in model_output:
            self.logger.error("Model output missing required 'ADUs' field")
            raise ValueError("Invalid model output: missing 'ADUs' field")
        
        if "relations" not in model_output:
            self.logger.warning("Model output missing 'relations' field, using empty list")
            model_output["relations"] = []
        
        # Initialize ADUs and relations lists
        adus = []
        relations = []
        
        # Process ADUs
        for adu_id, adu_data in model_output["ADUs"].items():
            # Validate required fields
            if "type" not in adu_data:
                self.logger.error(f"ADU {adu_id} missing required 'type' field")
                raise ValueError(f"ADU {adu_id} missing required 'type' field")
            
            if "text" not in adu_data:
                self.logger.error(f"ADU {adu_id} missing required 'text' field")
                raise ValueError(f"ADU {adu_id} missing required 'text' field")
            
            # Extract ADU data
            adu_type = adu_data["type"]
            text = adu_data["text"]
            
            # Validate ADU type (only Major Claim and Claim allowed in new schema)
            if adu_type not in ["Major Claim", "Claim"]:
                self.logger.warning(
                    f"ADU {adu_id} has invalid type '{adu_type}'. "
                    f"Expected 'Major Claim' or 'Claim'. Defaulting to 'Claim'."
                )
                adu_type = "Claim"
            
            # Create ADU object with fields matching the new schema
            adu = ADU(
                id=adu_id,  # ID is the equivalence class title
                type=adu_type,
                text=text,
                quote=adu_data.get("quote"),  # Optional field
                isImplicit=adu_data.get("isImplicit", False)  # Default to False
            )
            
            adus.append(adu)
        
        # Process relations
        if isinstance(model_output["relations"], list):
            for idx, rel_data in enumerate(model_output["relations"]):
                # Validate required fields
                if "src" not in rel_data:
                    self.logger.error(f"Relation {idx} missing required 'src' field")
                    continue
                
                if "tgt" not in rel_data:
                    self.logger.error(f"Relation {idx} missing required 'tgt' field")
                    continue
                
                if "type" not in rel_data:
                    self.logger.error(f"Relation {idx} missing required 'type' field")
                    continue
                
                src_id = rel_data["src"]
                tgt_id = rel_data["tgt"]
                rel_type = rel_data["type"]
                
                # Validate relation type
                if rel_type not in ["support", "attack"]:
                    self.logger.warning(
                        f"Relation {idx} has invalid type '{rel_type}'. "
                        f"Expected 'support' or 'attack'. Skipping."
                    )
                    continue
                
                # Validate that src and tgt ADUs exist
                src_exists = any(adu.id == src_id for adu in adus)
                tgt_exists = any(adu.id == tgt_id for adu in adus)
                
                if not src_exists:
                    self.logger.warning(f"Relation {idx} references non-existent source ADU: {src_id}")
                    continue
                
                if not tgt_exists:
                    self.logger.warning(f"Relation {idx} references non-existent target ADU: {tgt_id}")
                    continue
                
                # Create relation object
                relation = Relation(
                    id=f"rel-{idx+1}",
                    src=src_id,  # Using 'src' to match schema
                    tgt=tgt_id,  # Using 'tgt' to match schema
                    type=rel_type
                )
                relations.append(relation)
        
        # Extract metadata if present
        metadata = model_output.get("metadata", {})
        
        # Create argument map
        argument_map = ArgumentMap(
            id=map_id,
            adus=adus,
            relations=relations,
            metadata=metadata
        )
        
        self.logger.info(
            f"Parsed argument map with {len(adus)} ADUs and {len(relations)} relations"
        )
        
        # Log statistics
        stats = argument_map.map_statistics()
        self.logger.debug(f"Map statistics: {stats}")
        
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
    
    def parse_model_response(self, response: str, map_id: Optional[str] = None) -> ArgumentMap:
        """
        Parse a raw model response that may contain JSON within text.
        
        Args:
            response: Raw model response text
            map_id: Optional ID for the argument map
            
        Returns:
            ArgumentMap object
        """
        try:
            # Try to extract JSON from the response
            model_output = self.extract_from_text(response)
            return self.parse_dict(model_output, map_id or str(uuid.uuid4()))
        except Exception as e:
            self.logger.error(f"Failed to parse model response: {e}")
            raise
    
    def save_to_json(self, argument_map: ArgumentMap, output_path: Union[str, Path]):
        """
        Save an ArgumentMap to a JSON file in the schema format.
        
        Args:
            argument_map: ArgumentMap to save
            output_path: Path to save the JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to schema-compliant dictionary
        output_dict = argument_map.to_dict()
        
        # Ensure the output matches the schema structure
        if "ADUs" not in output_dict or "relations" not in output_dict:
            self.logger.error("ArgumentMap.to_dict() did not return schema-compliant structure")
            raise ValueError("Invalid ArgumentMap structure for output")
        
        # Write to file with proper formatting
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_dict, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"Saved argument map to {output_path}")
    
    def validate_against_schema(self, argument_map: ArgumentMap) -> bool:
        """
        Validate an ArgumentMap against the loaded schema.
        
        Args:
            argument_map: ArgumentMap to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not self.schema:
            self.logger.warning("No schema loaded, skipping validation")
            return True
        
        try:
            import jsonschema
            
            # Convert to dictionary format
            data = argument_map.to_dict()
            
            # Validate against schema
            jsonschema.validate(instance=data, schema=self.schema)
            self.logger.info("Argument map validated successfully against schema")
            return True
            
        except ImportError:
            self.logger.warning("jsonschema library not installed, skipping validation")
            return True
            
        except jsonschema.ValidationError as e:
            self.logger.error(f"Schema validation failed: {e.message}")
            return False
        
        except Exception as e:
            self.logger.error(f"Unexpected error during validation: {e}")
            return False