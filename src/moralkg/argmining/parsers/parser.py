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
            format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
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
            content = f.read()
        
        # Use the file stem as the map ID
        map_id = file_path.stem
        return self.parse_string(content, map_id)
    
    def parse_string(self, json_str: str, map_id: Optional[str] = None) -> ArgumentMap:
        """
        Parse a JSON string containing model output into an ArgumentMap.
        
        Args:
            json_str: JSON string containing model output
            map_id: Optional ID for the argument map (generated if not provided)
            
        Returns:
            ArgumentMap object
        """
        original_str = json_str
        attempts = 0
        max_attempts = 8
        
        while attempts < max_attempts:
            try:
                model_output = json.loads(json_str)
                break
            except json.JSONDecodeError as e:
                attempts += 1
                self.logger.error(f"Attempt {attempts}: Failed to parse JSON string: {e}")
                
                # Enhanced error context logging
                context_start = max(0, e.pos - 100)
                context_end = min(len(json_str), e.pos + 100)
                context = json_str[context_start:context_end]
                self.logger.debug(f"Context of error: {context}")
                self.logger.debug(f"Error location:   {' ' * min(100, e.pos - context_start)}^")

                if attempts >= max_attempts:
                    self.logger.error(f"Exhausted all {max_attempts} parsing attempts")
                    raise e
                
                try:
                    self.logger.info(f"Attempt {attempts}: Applying JSON fixes...")
                    json_str = self.fix_json_string_structure_issues(json_str, e)
                    self.logger.debug(f"Applied fixes, new length: {len(json_str)}")
                except Exception as fix_error:
                    self.logger.error(f"Error during JSON fixing: {fix_error}")
                    if attempts >= max_attempts:
                        raise e
                    # Reset to original and try different approach
                    json_str = original_str
                    continue
        # Log success
        self.logger.info(f"Successfully parsed JSON string after {attempts} attempts")
        
        if map_id is None:
            map_id = str(uuid.uuid4())
            
        return self.parse_dict(model_output, map_id)
    
    def fix_json_string_structure_issues(self, json_string: str, error: json.JSONDecodeError = None) -> str:
        """
        Fix common structural issues and JSON syntax problems.

        Args:
            json_string: The malformed JSON string
            error: The specific JSONDecodeError to help guide fixes
            
        Returns:
            Fixed JSON string
        """
        self.logger.debug("Fixing JSON string structure issues...")
        original_length = len(json_string)
        
        # First, try balancing brackets
        json_string = self._balance_brackets(json_string)

        try:
            # See if this is enough to fix loading
            json.loads(json_string)
            self.logger.debug("JSON loaded successfully after balancing brackets")
        except Exception as e:
            self.logger.error(f"Error loading JSON after balancing brackets: {e}")
            error = e if isinstance(e, json.JSONDecodeError) else error

        # Apply fixes based on error type
        if error:
            if "Unterminated string" in str(error):
                json_string = self._fix_unterminated_strings_advanced(json_string, error.pos)
            elif "Expecting ',' delimiter" in str(error):
                json_string = self._fix_missing_commas_advanced(json_string, error.pos)
            elif "Expecting property name" in str(error):
                json_string = self._fix_property_names_advanced(json_string, error.pos) # Expecting property name enclosed in double quotes
            else:
                # Not sure what kind of error this is. For debugging purpose, print much wider context
                context_start = max(0, error.pos - 100)
                context_end = min(len(json_string), error.pos + 100)
                context = json_string[context_start:context_end]
                self.logger.debug(f"Context of error: {context}")
                self.logger.debug(f"Error location:   {' ' * min(100, error.pos - context_start)}^")
        
        fixed_length = len(json_string)
        self.logger.debug(f"JSON fixing complete. Length changed from {original_length} to {fixed_length}")
        
        return json_string
    
    def _truncate_and_repair(self, json_string: str, error_pos: int) -> str:
        """Truncate JSON at a safe point and repair the structure."""
        self.logger.debug(f"Truncating and repairing JSON at error position {error_pos}")
        
        # Find a safe truncation point by going backwards from error
        safe_pos = error_pos
        bracket_depth = 0
        brace_depth = 0
        in_string = False
        
        # Scan backwards to find a complete structure
        for i in range(error_pos - 1, max(0, error_pos - 1000), -1):
            char = json_string[i]
            
            if char == '"' and (i == 0 or json_string[i-1] != '\\'):
                in_string = not in_string
            elif not in_string:
                if char == '}':
                    brace_depth += 1
                elif char == '{':
                    brace_depth -= 1
                elif char == ']':
                    bracket_depth += 1
                elif char == '[':
                    bracket_depth -= 1
                
                # Found a safe truncation point
                if (brace_depth == 0 and bracket_depth == 0 and 
                    char in ['}', ']'] and i < error_pos - 50):
                    safe_pos = i + 1
                    break
        
        # Truncate at safe position
        truncated = json_string[:safe_pos]
        
        # Add necessary closing characters
        truncated += '}' * brace_depth
        truncated += ']' * bracket_depth
        
        # If it doesn't look like valid JSON structure, wrap it
        truncated = truncated.strip()
        if not (truncated.startswith(('{', '[')) and truncated.endswith(('}', ']'))):
            truncated = '{' + truncated + '}'
        
        self.logger.debug(f"Truncated from {len(json_string)} to {len(truncated)} characters")
        return truncated
    
    def _extract_partial_data(self, json_string: str) -> Optional[Dict[str, Any]]:
        """Extract whatever valid data we can from malformed JSON."""
        self.logger.debug("Attempting partial data extraction...")
        
        try:
            # Strategy 1: Find individual ADUs using regex
            adu_pattern = r'"([^"]+)"\s*:\s*\{\s*"type"\s*:\s*"([^"]+)"\s*,\s*"text"\s*:\s*"([^"]*(?:\\.[^"]*)*)"[^}]*\}'
            adu_matches = re.findall(adu_pattern, json_string)
            
            adus = {}
            for match in adu_matches:
                adu_id, adu_type, text = match
                adus[adu_id] = {
                    "type": adu_type,
                    "text": text
                }
            
            if adus:
                self.logger.info(f"Extracted {len(adus)} ADUs using pattern matching")
                
                # Try to find relations
                relations = []
                rel_pattern = r'\{\s*"src"\s*:\s*"([^"]+)"\s*,\s*"tgt"\s*:\s*"([^"]+)"\s*,\s*"type"\s*:\s*"([^"]+)"\s*\}'
                rel_matches = re.findall(rel_pattern, json_string)
                
                for match in rel_matches:
                    src, tgt, rel_type = match
                    if rel_type in ["support", "attack"]:
                        relations.append({
                            "src": src,
                            "tgt": tgt,
                            "type": rel_type
                        })
                
                self.logger.info(f"Extracted {len(relations)} relations using pattern matching")
                
                return {
                    "ADUs": adus,
                    "relations": relations
                }
        
        except Exception as e:
            self.logger.error(f"Partial extraction failed: {e}")
        
        return None
    
    def _fix_unterminated_strings_advanced(self, json_string: str, error_pos: int) -> str:
        """Fix unterminated strings with more sophisticated logic."""
        self.logger.debug(f"Fixing unterminated string at position {error_pos}")
        
        # Find the problematic quote
        context_start = max(0, error_pos - 100)
        context = json_string[context_start:error_pos + 100]
        
        # Look backwards from error position to find the opening quote
        pos = error_pos
        quote_pos = -1
        while pos >= 0:
            if json_string[pos] == '"':
                # Check if this quote is escaped
                escape_count = 0
                check_pos = pos - 1
                while check_pos >= 0 and json_string[check_pos] == '\\':
                    escape_count += 1
                    check_pos -= 1
                
                # If even number of escapes (or none), this is a real quote
                if escape_count % 2 == 0:
                    quote_pos = pos
                    break
            pos -= 1
        
        if quote_pos >= 0:
            # Find where to place the closing quote
            # Look for structural characters that indicate end of string value
            search_start = error_pos
            search_end = min(len(json_string), search_start + 200)
            # For debugging, log the search context
            search_string = json_string[search_start:search_end]
            self.logger.debug(f"Searching for a location to place a closing quote in: {search_string}")

            for i in range(search_start, search_end):
                char = json_string[i]
                if char in [',', '}', ']', '\n']:
                    # Insert closing quote before this character
                    json_string = json_string[:i] + '"' + json_string[i:]
                    self.logger.debug(f"Added closing quote at position {i}")
                    break
            else:
                # If no structural character found, add at end
                json_string = json_string[:search_end] + '"' + json_string[search_end:]

        return json_string
    
    def _balance_brackets(self, json_string: str) -> str:
        """Balance brackets and braces by adding missing ones."""
        # More sophisticated bracket balancing
        open_braces = []
        open_brackets = []
        
        i = 0
        while i < len(json_string):
            char = json_string[i]
            
            if char == '{':
                open_braces.append(i)
            elif char == '}':
                if open_braces:
                    open_braces.pop()
                else:
                    # Extra closing brace - remove it
                    json_string = json_string[:i] + json_string[i+1:]
                    i -= 1
                    
            elif char == '[':
                open_brackets.append(i)
            elif char == ']':
                if open_brackets:
                    open_brackets.pop()
                else:
                    # Extra closing bracket - remove it
                    json_string = json_string[:i] + json_string[i+1:]
                    i -= 1
            
            # Skip quoted strings to avoid counting brackets inside quotes
            elif char == '"':
                i += 1
                while i < len(json_string) and json_string[i] != '"':
                    if json_string[i] == '\\':
                        i += 1  # Skip escaped character
                    i += 1
            
            i += 1
        
        # Add missing closing braces
        json_string += '}' * len(open_braces)
        
        # Add missing closing brackets  
        json_string += ']' * len(open_brackets)
        
        return json_string
    
    def _truncate_at_error(self, json_string: str, error_pos: int) -> str:
        """Truncate JSON string at error position and try to close it properly."""
        self.logger.debug(f"Truncating JSON at position {error_pos}")
        
        # Find a safe truncation point before the error
        safe_pos = error_pos
        
        # Look backwards for a safe place to truncate (after complete object/value)
        bracket_depth = 0
        brace_depth = 0
        in_string = False
        
        for i in range(error_pos - 1, -1, -1):
            char = json_string[i]
            
            if char == '"' and (i == 0 or json_string[i-1] != '\\'):
                in_string = not in_string
            elif not in_string:
                if char == '}':
                    brace_depth += 1
                elif char == '{':
                    brace_depth -= 1
                elif char == ']':
                    bracket_depth += 1
                elif char == '[':
                    bracket_depth -= 1
                
                # If we're at depth 0 and just closed an object/array
                if (brace_depth == 0 and bracket_depth == 0 and 
                    char in ['}', ']'] and i < error_pos - 10):
                    safe_pos = i + 1
                    break
        
        truncated = json_string[:safe_pos]
        
        # Try to close the structure properly
        truncated += '}' * brace_depth
        truncated += ']' * bracket_depth
        
        return truncated
    
    def _fix_missing_commas_advanced(self, json_string: str, error_pos: int) -> str:
        """Fix missing commas with better context awareness."""
        self.logger.debug(f"Fixing missing comma at position {error_pos}")
        
        # Check the immediate context
        if error_pos > 0:
            before_char = json_string[error_pos - 1]
            after_char = json_string[error_pos] if error_pos < len(json_string) else ''
            
            # First, just try adding a comma at the error position and checking if that would load correctly. If not, remove the comma and proceed.
            json_string = json_string[:error_pos] + ',' + json_string[error_pos:]
            try:
                json.loads(json_string)
            except json.JSONDecodeError as e:
                # Check if the error is actually still a missing comma, or if it's a different error now
                if "Expecting ',' delimiter" in str(e):
                    # Log the context of the error
                    self.logger.debug(f"Tried simply adding a comma. JSONDecodeError at position {error_pos}: {json_string[error_pos-30:error_pos+30]}")
                    self.logger.debug(f"Error location:                                                       {' ' * 30}^")
                    json_string = json_string[:error_pos] + json_string[error_pos+1:]
                    self.logger.debug(f"Removed the comma, reverting to original string.")
                else:
                    self.logger.debug(f"Tried simply adding a comma. Now different error at position {e.pos}: {e}")
                    self.logger.debug(f"New context of error: {json_string[e.pos-30:e.pos+30]}")
                    self.logger.debug(f"Error location:       {' ' * 30}^")
                    #json_string = json_string[:e.pos] + json_string[e.pos+1:]
                    self.logger.debug(f"Keeping the comma as it resolved the original error.")
                    return json_string

            # Common patterns that need commas:
            if before_char == '}' and after_char in ['"', '{']:
                json_string = json_string[:error_pos] + ',' + json_string[error_pos:]

            elif before_char == '}' and after_char == ']':
                json_string = json_string[:error_pos] + ',' + json_string[error_pos:]

            elif before_char == ']' and after_char == '{':
                json_string = json_string[:error_pos] + ',' + json_string[error_pos:]
                
            elif before_char == '"' and after_char == '"':
                # Might need comma between string values in array
                json_string = json_string[:error_pos] + ',' + json_string[error_pos:]
                
            # Handle the case where we're at the end and missing comma before closing
            elif before_char == '}' and after_char == '}':
                # This suggests we're in an array of objects missing comma
                json_string = json_string[:error_pos] + ',' + json_string[error_pos:]
        
        return json_string
    
    def _fix_property_names_advanced(self, json_string: str, error_pos: int) -> str:
        """Fix property name quoting with better pattern recognition."""
        self.logger.debug(f"Fixing property name at position {error_pos}")
        
        # Look for unquoted property names around error position
        search_start = max(0, error_pos - 20)
        search_end = min(len(json_string), error_pos + 50)
        
        # Find the problematic property name
        # Look for pattern: {unquoted_name: or ,unquoted_name:
        for i in range(search_start, search_end):
            if json_string[i] in ['{', ',']:
                # Look for the next colon
                colon_pos = json_string.find(':', i, search_end)
                if colon_pos > 0:
                    # Extract property name candidate
                    prop_start = i + 1
                    while prop_start < colon_pos and json_string[prop_start].isspace():
                        prop_start += 1
                    
                    prop_end = colon_pos
                    while prop_end > prop_start and json_string[prop_end-1].isspace():
                        prop_end -= 1
                    
                    if prop_start < prop_end:
                        prop_candidate = json_string[prop_start:prop_end]
                        
                        # Check if it's an unquoted property name
                        if not (prop_candidate.startswith('"') and prop_candidate.endswith('"')):
                            # Quote it properly
                            if prop_candidate.startswith('"') or prop_candidate.endswith('"'):
                                # Partially quoted, fix it
                                prop_candidate = prop_candidate.strip('"')
                            
                            quoted_prop = f'"{prop_candidate}"'
                            json_string = json_string[:prop_start] + quoted_prop + json_string[prop_end:]
                            self.logger.debug(f"Fixed property name: {prop_candidate} -> {quoted_prop}")
                            return json_string
                        else:
                            # The property name is probably quoted correctly; try other approaches
                            json_string = self._fix_property_formatting(json_string, prop_start, prop_end)
                            self.logger.debug(f"Fixed property formatting: {json_string}")
                            return json_string

        return json_string

    def _fix_property_formatting(self, json_string: str, prop_start: int, prop_end: int) -> str:
        """Fix formatting issues around a property name."""
        self.logger.debug(f"Fixing property formatting around {prop_start} to {prop_end}")

        # Try closing any open brackets or braces before this property name
        open_brackets = json_string[:prop_start].count('{') - json_string[:prop_start].count('}')
        open_braces = json_string[:prop_start].count('[') - json_string[:prop_start].count(']')
        if open_brackets > 0:
            # Make sure that the space between the last bracket and the start of the property name is the same
            json_string = json_string[:prop_start-1] + '}' * open_brackets + json_string[prop_start-1:]
        if open_braces > 0:
            json_string = json_string[:prop_start-1] + ']' * open_braces + json_string[prop_start-1:]

        return json_string

    def parse_dict(self, model_output: Dict[str, Any], map_id: str) -> ArgumentMap:
        """
        Parse a dictionary containing model output into an ArgumentMap.
        
        Args:
            model_output: Dictionary containing model output
            map_id: ID for the argument map
            
        Returns:
            ArgumentMap object
        """
        # Handle case where we have an array of objects (from fixing multiple objects)
        if isinstance(model_output, list):
            self.logger.info(f"Found array of {len(model_output)} objects, merging into single map")
            merged_output = {"ADUs": {}, "relations": []}
            
            for obj in model_output:
                if isinstance(obj, dict):
                    if "ADUs" in obj:
                        merged_output["ADUs"].update(obj["ADUs"])
                    if "relations" in obj:
                        if isinstance(obj["relations"], list):
                            merged_output["relations"].extend(obj["relations"])
            
            model_output = merged_output
        
        # Enhanced structure validation
        if not isinstance(model_output, dict):
            self.logger.error(f"Model output is not a dictionary: {type(model_output)}")
            raise ValueError(f"Invalid model output: expected dictionary, got {type(model_output)}")
        
        # Initialize ADUs and relations lists
        adus = []
        relations = []
        
        if "ADUs" not in model_output:
            self.logger.error("Model output missing required 'ADUs' field")
            self.logger.debug(f"Available keys: {list(model_output.keys())}")
            
            # If the "properties" field exists, the ADUs and relations are probably in there.
            if "properties" in model_output:
                self.logger.info("Found 'properties' field, extracting ADUs and relations")
                model_output = {
                    "ADUs": model_output["properties"].get("ADUs", {}),
                    "relations": model_output["properties"].get("relations", [])
                }
            else:
                self.logger.error("Model output missing required 'ADUs' field")
                raise ValueError("Invalid model output: missing 'ADUs' field and cannot recover structure")

        # Process ADUs with better error handling
        adu_dict = model_output["ADUs"]
        if not isinstance(adu_dict, dict):
            self.logger.error(f"ADUs field is not a dictionary: {type(adu_dict)}")
            raise ValueError(f"Invalid ADUs field: expected dictionary, got {type(adu_dict)}")
        
        for adu_id, adu_data in adu_dict.items():
            try:
                # Validate required fields
                if not isinstance(adu_data, dict):
                    self.logger.error(f"ADU {adu_id} is not a dictionary: {type(adu_data)}")
                    continue
                
                if "type" not in adu_data:
                    self.logger.error(f"ADU {adu_id} missing required 'type' field")
                    continue
                
                if "text" not in adu_data:
                    self.logger.error(f"ADU {adu_id} missing required 'text' field")
                    continue
                
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
                
                # Handle isImplicit field - convert string to boolean if needed
                is_implicit = adu_data.get("isImplicit", False)
                if isinstance(is_implicit, str):
                    is_implicit = is_implicit.lower() == "true"
                
                # Create ADU object
                adu = ADU(
                    id=adu_id,
                    type=adu_type,
                    text=text,
                    quote=adu_data.get("quote"),
                    isImplicit=is_implicit
                )
                
                adus.append(adu)
                
            except Exception as e:
                self.logger.error(f"Error processing ADU {adu_id}: {e}")
                continue
        
        # Process relations with better error handling
        relations_data = model_output["relations"]
        if isinstance(relations_data, list):
            for idx, rel_data in enumerate(relations_data):
                try:
                    if not isinstance(rel_data, dict):
                        self.logger.error(f"Relation {idx} is not a dictionary: {type(rel_data)}")
                        continue
                    
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
                    
                    src_id = str(rel_data["src"])  # Ensure string
                    tgt_id = str(rel_data["tgt"])  # Ensure string
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
                        src=src_id,
                        tgt=tgt_id,
                        type=rel_type
                    )
                    relations.append(relation)
                    
                except Exception as e:
                    self.logger.error(f"Error processing relation {idx}: {e}")
                    continue
        
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
        try:
            stats = argument_map.map_statistics()
            self.logger.debug(f"Map statistics: {stats}")
        except Exception as e:
            self.logger.warning(f"Could not generate map statistics: {e}")
        
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
        # Enhanced JSON extraction patterns
        json_patterns = [
            r'```json\s*(.*?)\s*```',  # Code block with json syntax
            r'```\s*(.*?)\s*```',       # Any code block
            r'(?s)\{.*?\}(?=\s*$)',     # JSON object at end of text
            r'(?s)\{.*?\}(?=\s*\n\s*[A-Z])',  # JSON object followed by text
            r'(?s)\{.*\}'               # Any JSON-like object
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
            if matches:
                for match in matches:
                    try:
                        return json.loads(match)
                    except json.JSONDecodeError as e:
                        # Try fixing the extracted JSON
                        try:
                            fixed_match = self.fix_json_string_structure_issues(match, e)
                            return json.loads(fixed_match)
                        except json.JSONDecodeError:
                            continue
        
        if allow_partial:
            # Enhanced partial extraction
            # Try to find the largest valid JSON structure
            brace_positions = []
            for i, char in enumerate(text):
                if char == '{':
                    brace_positions.append(i)
            
            # Try each opening brace as a potential start
            for start_pos in reversed(brace_positions):  # Start with later positions
                # Find matching closing brace
                depth = 0
                in_string = False
                for i in range(start_pos, len(text)):
                    char = text[i]
                    
                    if char == '"' and (i == 0 or text[i-1] != '\\'):
                        in_string = not in_string
                    elif not in_string:
                        if char == '{':
                            depth += 1
                        elif char == '}':
                            depth -= 1
                            
                        if depth == 0 and i > start_pos:
                            # Found complete object
                            json_substr = text[start_pos:i+1]
                            try:
                                return json.loads(json_substr)
                            except json.JSONDecodeError as e:
                                try:
                                    fixed_substr = self.fix_json_string_structure_issues(json_substr, e)
                                    return json.loads(fixed_substr)
                                except json.JSONDecodeError:
                                    break  # Try next position
        
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
            # Log some context for debugging
            self.logger.debug(f"Response length: {len(response)}")
            self.logger.debug(f"Response preview: {response[:500]}...")
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