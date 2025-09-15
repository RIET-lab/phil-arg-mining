"""
Streamlined parser for model outputs in argument mining tasks.

This module provides functionality to parse model responses into
standardized ArgumentMap objects using a regex-based approach for
robustness against malformed JSON.
"""

import re
import json
import uuid
import logging # TODO: Replace with "from moralkg import get_logger" and update logger loading accordingly
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from ..schemas import ArgumentMap, ADU, Relation


class Parser:
    """
    Parser for model output in JSON format.
    
    Converts model responses into ArgumentMap objects, gracefully handling
    malformed JSON by extracting ADUs and relations using regex patterns.
    """
    
    def __init__(self, schema_path: Optional[str] = None, log_level: str = "INFO"):
        """
        Initialize the parser.
        
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
        self.logger = logging.getLogger("StreamlinedParser")
    
    def parse_json_file(self, file_path: Union[str, Path], map_id: Optional[str] = None) -> ArgumentMap:
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

        if not map_id:
                map_id = file_path.stem
        return self.parse_string(content, map_id)
    
    def parse_string(self, json_str: str, map_id: Optional[str] = None) -> ArgumentMap:
        """
        Parse a JSON string containing model output into an ArgumentMap.
        
        Args:
            json_str: JSON string containing model output
            map_id: Optional ID for the argument map
            
        Returns:
            ArgumentMap object
        """
        if map_id is None:
            map_id = str(uuid.uuid4())
        
        # Step 1: Try normal JSON parsing
        try:
            model_output = json.loads(json_str)
            self.logger.info("Successfully parsed JSON using standard parser")
            return self.parse_dict(model_output, map_id)
        except json.JSONDecodeError as e:
            self.logger.warning(f"Standard JSON parsing failed: {e}")
            self.logger.info("Falling back to regex-based parsing")
            
            # Step 2: Use regex-based parsing
            return self._parse_with_regex(json_str, map_id)
    
    def _parse_with_regex(self, text: str, map_id: str) -> ArgumentMap:
        """
        Parse text using regex patterns to extract ADUs and relations.
        
        Args:
            text: Text containing ADU and relation data
            map_id: ID for the argument map
            
        Returns:
            ArgumentMap object
        """
        self.logger.info("Starting regex-based parsing")
        
        # Step 1: Split into ADUs and relations sections
        adus_section, relations_section = self._split_sections(text)
        
        # Step 2: Extract ADUs and relations
        adus = self._extract_adus(adus_section)
        if not adus:
            self.logger.warning("No ADUs extracted, resulting ArgumentMap will be empty")
            relations = []
        else:
            relations = self._extract_relations(relations_section)
        
        # Create argument map
        argument_map = ArgumentMap(
            id=map_id,
            adus=adus,
            relations=relations,
            metadata={}
        )
        
        self.logger.info(
            f"Regex parsing complete: {len(adus)} ADUs, {len(relations)} relations"
        )
        
        return argument_map
    
    def _split_sections(self, text: str) -> tuple[str, str]:
        """
        Split text into ADUs and relations sections.
        
        Args:
            text: Full text to split
            
        Returns:
            Tuple of (adus_section, relations_section)
        """
        # Find last instance of "ADUs". There may be many if the model also generated a schema description.
        adu_matches = list(re.finditer(r'\"ADUs\"', text, re.IGNORECASE))
        if not adu_matches:
            self.logger.warning("No 'ADUs' section found in text")
            adus_section = ""
            relations_search_start = 0
        else:
            last_adu_match = adu_matches[-1]
            adus_start = last_adu_match.end()
            
            # The relations section starts after the ADUs section, so the ADUs section is everything between the last "ADUs" and the first "relations"
            relations_match = re.search(r'\"relations\"', text[adus_start:], re.IGNORECASE)
            if relations_match:
                adus_end = adus_start + relations_match.start()
                adus_section = text[adus_start:adus_end]
                relations_search_start = adus_end
            else:
                # No relations section found after ADUs, take everything after ADUs as ADUs section
                adus_section = text[adus_start:]
                relations_search_start = len(text)
        
        # Find relations section
        relations_match = re.search(r'\"relations\"', text[relations_search_start:], re.IGNORECASE)
        if relations_match:
            relations_start = relations_search_start + relations_match.end()
            relations_section = text[relations_start:]
        else:
            self.logger.warning("No 'relations' section found in text")
            relations_section = ""
        
        self.logger.debug(f"ADUs section length: {len(adus_section)}")
        self.logger.debug(f"Relations section length: {len(relations_section)}")
        
        return adus_section, relations_section

    
    def _extract_adus(self, adus_section: str) -> List[ADU]:
        """
        Extract ADU objects from the ADUs section.
        
        Args:
            adus_section: Text containing ADU definitions
            
        Returns:
            List of ADU objects
        """
        adus = []
        
        # Pattern to match individual ADU definitions
        # Handles various quote styles and missing fields gracefully
        adu_pattern = r'''
            ["']?([^"':]+?)["']?\s*:\s*\{    # ADU ID (capture group 1)
            (.*?)                           # ADU content (capture group 2)
            \}(?=\s*[,}]|\s*["']?\w+["']?\s*:|$)  # End of ADU object
        '''
        
        adu_matches = re.finditer(adu_pattern, adus_section, re.DOTALL | re.VERBOSE)
        
        for match in adu_matches:
            adu_id = match.group(1).strip()
            adu_content = match.group(2)
            
            try:
                adu = self._parse_single_adu(adu_id, adu_content)
                if adu:
                    adus.append(adu)
            except Exception as e:
                self.logger.error(f"Error parsing ADU {adu_id}: {e}")
                continue

        if len(adus) < 1:
            self.logger.warning("No ADUs extracted, possible parsing issue")
        
        self.logger.info(f"Extracted {len(adus)} ADUs")
        return adus
    
    def _parse_single_adu(self, adu_id: str, content: str) -> Optional[ADU]:
        """
        Parse a single ADU from its content string.
        
        Args:
            adu_id: ADU identifier
            content: ADU content string
            
        Returns:
            ADU object or None if parsing fails
        """
        # Extract type
        type_match = re.search(r'"?type"?\s*:\s*["\']([^"\']+)["\']', content, re.IGNORECASE)
        adu_type = type_match.group(1) if type_match else "Claim"
        
        # Validate and normalize type
        if adu_type.lower() not in ["major claim", "claim"]:
            if adu_type.lower() == "majorclaim":
                adu_type = "Major Claim"
            elif adu_type.lower() == "premise":
                adu_type = "Claim"
            elif adu_type.lower() in ["string", "boolean"]:
                # This is from the schema description. Skip this ADU and log that the schema description was improperly included in the ADUs section.
                self.logger.warning(f"ADU {adu_id}: Detected schema description, skipping")
                return None
            elif adu_type.lower() in ["support", "attack"]:
                self.logger.warning(f"ADU {adu_id}: Detected relation type '{adu_type}', defaulting to 'Claim'")
                adu_type = "Claim"
            else:
                self.logger.warning(f"ADU {adu_id}: Invalid type '{adu_type}', defaulting to 'Claim'")
                adu_type = "Claim"
        
        # Extract text (required field)
        text_match = re.search(r'"?text"?\s*:\s*["\'](.*?)["\']\s*[,}]', content, re.DOTALL | re.IGNORECASE)
        if not text_match:
            # Try without quotes for edge cases
            text_match = re.search(r'"?text"?\s*:\s*([^,}]+)', content, re.IGNORECASE)
        
        if not text_match:
            self.logger.error(f"ADU {adu_id}: Missing required 'text' field")
            return None
        
        text = text_match.group(1).strip()
        # Clean up escaped quotes
        text = re.sub(r'\\["\'"]', '"', text)
        
        # Extract optional quote field
        quote_match = re.search(r'"?quote"?\s*:\s*["\'](.*?)["\']\s*[,}]', content, re.DOTALL | re.IGNORECASE)
        quote = quote_match.group(1).strip() if quote_match else None
        if quote:
            quote = re.sub(r'\\["\'"]', '"', quote)
        
        # Extract optional isImplicit field
        implicit_match = re.search(r'"?isImplicit"?\s*:\s*(true|false|True|False)', content, re.IGNORECASE)
        is_implicit = False
        if implicit_match:
            is_implicit = implicit_match.group(1).lower() == "true"
        
        return ADU(
            id=adu_id,
            type=adu_type,
            text=text,
            quote=quote,
            isImplicit=is_implicit
        )
    
    def _extract_relations(self, relations_section: str) -> List[Relation]:
        """
        Extract relation objects from the relations section.
        
        Args:
            relations_section: Text containing relation definitions
            
        Returns:
            List of Relation objects
        """
        relations = []
        
        # Pattern to match individual relation objects
        relation_pattern = r'''
            \{                              # Start of relation object
            (.*?)                           # Relation content (capture group 1)
            \}                              # End of relation object
        '''
        
        relation_matches = re.finditer(relation_pattern, relations_section, re.DOTALL | re.VERBOSE)
        
        for idx, match in enumerate(relation_matches):
            relation_content = match.group(1)
            
            try:
                relation = self._parse_single_relation(idx, relation_content)
                if relation:
                    relations.append(relation)
            except Exception as e:
                self.logger.error(f"Error parsing relation {idx}: {e}")
                continue
        
        self.logger.info(f"Extracted {len(relations)} relations")
        return relations
    
    def _parse_single_relation(self, idx: int, content: str) -> Optional[Relation]:
        """
        Parse a single relation from its content string.
        
        Args:
            idx: Relation index for ID generation
            content: Relation content string
            
        Returns:
            Relation object or None if parsing fails
        """
        # Check if this is somehow a claim
        if re.search(r'"?type"?\s*:\s*["\'](Major Claim|Claim)["\']', content, re.IGNORECASE):
            self.logger.warning(f"Relation {idx}: Detected ADU instead of relation, skipping")
            return None

        # Extract src
        src_match = re.search(r'"?src"?\s*:\s*["\'"]([^"\']+)["\'"]', content, re.IGNORECASE)
        if not src_match:
            self.logger.error(f"Relation {idx}: Missing required 'src' field")
            return None
        src = src_match.group(1).strip()
        
        # Extract tgt
        tgt_match = re.search(r'"?tgt"?\s*:\s*["\'"]([^"\']+)["\'"]', content, re.IGNORECASE)
        if not tgt_match:
            self.logger.error(f"Relation {idx}: Missing required 'tgt' field")
            return None
        tgt = tgt_match.group(1).strip()
        
        # Extract type
        type_match = re.search(r'"?type"?\s*:\s*["\'"]([^"\']+)["\'"]', content, re.IGNORECASE)
        if not type_match:
            self.logger.error(f"Relation {idx}: Missing required 'type' field")
            return None
        rel_type = type_match.group(1).strip()
        
        # Validate relation type
        if rel_type not in ["support", "attack", "unknown"]:
            self.logger.warning(f"Relation {idx}: Invalid type '{rel_type}', skipping")
            return None
        
        return Relation(
            id=f"rel-{idx+1}",
            src=src,
            tgt=tgt,
            type=rel_type
        )
    
    def parse_dict(self, model_output: Dict[str, Any], map_id: str) -> ArgumentMap:
        """
        Parse a dictionary containing model output into an ArgumentMap.
        TODO: Make sure this is not redundant with the regex-based parsing.

        Args:
            model_output: Dictionary containing model output
            map_id: ID for the argument map
            
        Returns:
            ArgumentMap object
        """
        # Validate structure
        if not isinstance(model_output, dict):
            raise ValueError(f"Invalid model output: expected dictionary, got {type(model_output)}")

        # Normalize field names to be case-insensitive
        model_output = {k.lower(): v for k, v in model_output.items()}

        # Reintroduce case sensitivity for known field names where capitalization is informative
        if "adus" in model_output:
            model_output["ADUs"] = model_output.pop("adus")
        else:
            raise ValueError("Invalid model output: missing 'ADUs' field (case-insensitive)")
        if "isimplicit" in model_output:
            model_output["isImplicit"] = model_output.pop("isimplicit")

        # Process ADUs and relations using helper methods
        adu_dict = model_output.get("ADUs", {})
        adus = self._parse_structured_adu_data(adu_dict)

        relations_data = model_output.get("relations", [])
        relations = self._parse_relations_from_list(relations_data, adus)
        
        # Create argument map
        metadata = model_output.get("metadata", {})
        argument_map = ArgumentMap(
            id=map_id,
            adus=adus,
            relations=relations,
            metadata=metadata
        )
        
        self.logger.info(f"Parsed argument map with {len(adus)} ADUs and {len(relations)} relations")
        return argument_map

    def _parse_structured_adu_data(self, adu_data: Any) -> List[ADU]:
        """
        Convert ADU entries into a list of ADU objects.

        Accepts either:
        - dict mapping adu_id -> {type, text, ...}
        - list of ADU objects where each item is a dict containing at least 'text' and optionally 'id' and 'type'

        This preserves previous logging and defensive checks.
        """
        adus: List[ADU] = []

        if isinstance(adu_data, dict):
            items = list(adu_data.items())
        elif isinstance(adu_data, list):
            # convert list to (id, data) pairs; id may be inside element or generated
            items = []
            for idx, item in enumerate(adu_data):
                if isinstance(item, dict):
                    adu_id = item.get("id") or item.get("ID") or f"adu-{idx+1}"
                    items.append((adu_id, item))
                else:
                    # Non-dict items are invalid for ADU entries
                    self.logger.error(f"ADU at index {idx} is not a dictionary, skipping")
            
        else:
            self.logger.error("ADUs is neither a dict nor a list; cannot parse")
            return adus

        for adu_id, adu_item in items:
            try:
                if not isinstance(adu_item, dict):
                    self.logger.error(f"ADU {adu_id} is not a dictionary")
                    continue

                # Validate required fields
                if "text" not in adu_item:
                    # If it was the dict form, previous behavior required both 'type' and 'text'. For list form, 'type' can be optional.
                    self.logger.error(f"ADU {adu_id} missing required 'text' field")
                    continue

                adu_type = adu_item.get("type", "Claim")
                if adu_type not in ["Major Claim", "Claim"]:
                    self.logger.warning(f"ADU {adu_id}: Invalid type '{adu_type}', defaulting to 'Claim'")
                    adu_type = "Claim"

                # Handle isImplicit field
                is_implicit = adu_item.get("isImplicit", False)
                if isinstance(is_implicit, str):
                    is_implicit = is_implicit.lower() == "true"

                adu = ADU(
                    id=str(adu_id),
                    type=adu_type,
                    text=adu_item["text"],
                    quote=adu_item.get("quote"),
                    isImplicit=is_implicit,
                )
                adus.append(adu)

            except Exception as e:
                self.logger.error(f"Error processing ADU {adu_id}: {e}")
                continue

        return adus

    def _parse_relations_from_list(self, relations_data: Any, adus: List[ADU]) -> List[Relation]:
        """
        Convert relations list entries into Relation objects, validating references against ADUs.
        """
        relations: List[Relation] = []

        if not isinstance(relations_data, list):
            return relations

        for idx, rel_data in enumerate(relations_data):
            try:
                if not isinstance(rel_data, dict):
                    continue

                # Validate required fields
                required_fields = ["src", "tgt", "type"]
                if not all(field in rel_data for field in required_fields):
                    self.logger.error(f"Relation {idx} missing required fields")
                    continue

                src_id = str(rel_data["src"])
                tgt_id = str(rel_data["tgt"])
                rel_type = rel_data["type"]

                if rel_type not in ["support", "attack", "unknown"]:
                    self.logger.warning(f"Relation {idx}: Invalid type '{rel_type}', skipping")
                    continue

                # Validate ADU references
                src_exists = any(adu.id == src_id for adu in adus)
                tgt_exists = any(adu.id == tgt_id for adu in adus)

                if not src_exists or not tgt_exists:
                    self.logger.warning(f"Relation {idx} references non-existent ADUs")
                    continue

                relation = Relation(
                    id=f"rel-{idx+1}",
                    src=src_id,
                    tgt=tgt_id,
                    type=rel_type,
                )
                relations.append(relation)

            except Exception as e:
                self.logger.error(f"Error processing relation {idx}: {e}")
                continue

        return relations
    
    def parse_model_response(self, response: str, map_id: Optional[str] = None) -> ArgumentMap:
        """
        Parse a raw model response that may contain JSON within text.
        
        Args:
            response: Raw model response text
            map_id: Optional ID for the argument map
            
        Returns:
            ArgumentMap object
        """
        # Try to extract JSON from response first
        json_patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'(?s)\{.*\}'
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                for match in matches:
                    try:
                        return self.parse_string(match, map_id)
                    except Exception:
                        continue
        
        # If no JSON found, try parsing the entire response
        return self.parse_string(response, map_id)
    
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