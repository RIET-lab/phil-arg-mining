"""
generate.py

Generates arguments for papers using OpenAI and Anthropic models.
"""

import argparse
import csv
import glob
import json
import os
import random
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

import anthropic
import openai
import rootutils

# Get root
root = rootutils.setup_root(__file__, indicator=".env")

# Load API keys
oai_key = os.getenv("OPEN_AI_API_KEY")
atrpc_key = os.getenv("ANTHROPIC_API_KEY")

# Initialize API clients
openai_client = openai.OpenAI(api_key=oai_key) if oai_key else None
anthropic_client = anthropic.Anthropic(api_key=atrpc_key) if atrpc_key else None


def load_prompt_template() -> str:
    """Load the prompt template from prompt.txt"""
    prompt_path = Path(__file__).parent / "prompt.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found at {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def load_paper_content(
    papers_dir: Path, identifier: str, prefer_md: bool = False
) -> Optional[str]:
    """
    Load paper content from the papers directory.
    Prioritizes .txt unless prefer_md is True.
    """
    if prefer_md:
        extensions = [".md", ".txt"]
    else:
        extensions = [".txt", ".md"]

    for ext in extensions:
        paper_path = papers_dir / f"{identifier}{ext}"
        if paper_path.exists():
            with open(paper_path, "r", encoding="utf-8") as f:
                return f.read()

    return None


def generate_hyperparameters(
    min_temp: float,
    max_temp: float,
    min_tokens: int,
    max_tokens: int,
    num_combinations: int = 4,
) -> List[Dict[str, Any]]:
    """Generate random hyperparameter combinations within specified ranges"""
    combinations = []

    for _ in range(num_combinations):
        temp = random.uniform(min_temp, max_temp)
        tokens = random.randint(min_tokens, max_tokens)
        combinations.append({"temperature": temp, "max_tokens": tokens})

    return combinations


def validate_adus_in_text(
    parsed_response: Dict[str, Any], source_text: str
) -> Dict[str, Any]:
    """Validate that all ADU text spans actually exist in the source text"""
    if "ADUs" not in parsed_response:
        return parsed_response

    validated_adus = {}
    invalid_adus = []
    valid_relations = []

    # First pass: validate ADUs
    for adu_id, adu_data in parsed_response["ADUs"].items():
        adu_text = adu_data.get("text", "")

        # Remove all whitespace and convert to lowercase for comparison
        normalized_adu = "".join(adu_text.split()).lower()
        normalized_source = "".join(source_text.split()).lower()
        
        # Basic whitespace-normalized approach
        basic_normalized_adu = " ".join(adu_text.split())
        basic_normalized_source = " ".join(source_text.split())
        
        is_valid = False
        
        if normalized_adu and (
            basic_normalized_adu in basic_normalized_source or
            normalized_adu in normalized_source
        ):
            is_valid = True

        if is_valid:
            validated_adus[adu_id] = adu_data
        else:
            invalid_adus.append({"id": adu_id, "text": adu_text})

    # Second pass: validate relations (only keep if both source and target ADUs are valid)
    if "relations" in parsed_response:
        for relation in parsed_response["relations"]:
            source_id = relation.get("source", "")
            target_id = relation.get("target", "")

            if source_id in validated_adus and target_id in validated_adus:
                valid_relations.append(relation)

    # Log validation results
    total_adus = len(parsed_response["ADUs"])
    valid_adus = len(validated_adus)
    filtered_adus = len(invalid_adus)

    if filtered_adus > 0:
        print(
            f"ADU Validation: {valid_adus}/{total_adus} ADUs validated, {filtered_adus} filtered out"
        )
        for invalid in invalid_adus[:3]:  # Show first 3 invalid ADUs
            print(f"  Filtered ADU: {invalid['text'][:100]}...")

    total_relations = len(parsed_response.get("relations", []))
    valid_relations_count = len(valid_relations)
    filtered_relations = total_relations - valid_relations_count

    if filtered_relations > 0:
        print(
            f"Relation Validation: {valid_relations_count}/{total_relations} relations validated, {filtered_relations} filtered out"
        )

    return {"ADUs": validated_adus, "relations": valid_relations}


def call_openai_api(
    system_prompt: str,
    user_content: str,
    temperature: float = 0.1,
    max_tokens: int = 4000,
    retries: int = 3,
) -> Optional[Dict[str, Any]]:
    """Call OpenAI API with retry logic"""
    if not openai_client:
        print("Warning: OpenAI API key not found, skipping OpenAI call")
        return None

    for attempt in range(retries):
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-2024-11-20",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {"role": "user", "content": user_content},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Handle potential None content
            message_content = response.choices[0].message.content
            if message_content is None:
                print(f"OpenAI returned empty content (attempt {attempt + 1})")
                if attempt == retries - 1:
                    return {
                        "source": "gpt-4o-2024-11-20",
                        "hyperparameters": {
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                        },
                        "raw_response": None,
                        "parsed_response": None,
                        "error": "Empty response content",
                    }
                continue

            content = message_content.strip()

            # Remove markdown code block delimiters if present
            if content.startswith("```json") and content.endswith("```"):
                content = content[7:-3].strip()
            elif content.startswith("```") and content.endswith("```"):
                content = content[3:-3].strip()

            # Try to parse as JSON to validate format
            try:
                result = json.loads(content)

                # Validate that all ADUs actually exist in the source text
                validated_result = validate_adus_in_text(result, user_content)

                return {
                    "source": "gpt-4o-2024-11-20",
                    "hyperparameters": {
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                    "raw_response": content,
                    "parsed_response_unvalidated": result,
                    "parsed_response": validated_result,
                }
            except json.JSONDecodeError as e:
                print(f"OpenAI response is not valid JSON (attempt {attempt + 1}): {e}")
                print(f"DEBUG - Raw OpenAI response: {repr(content)}")
                if attempt == retries - 1:
                    return {
                        "source": "OpenAI GPT-4o-2024-11-20",
                        "hyperparameters": {
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                        },
                        "raw_response": content,
                        "parsed_response": None,
                        "error": f"Invalid JSON: {e}",
                    }

        except Exception as e:
            print(f"OpenAI API error (attempt {attempt + 1}): {e}")
            if attempt < retries - 1:
                time.sleep(2**attempt)  # Exponential backoff
            else:
                return {
                    "source": "gpt-4o-2024-11-20",
                    "hyperparameters": {
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                    "raw_response": None,
                    "parsed_response": None,
                    "error": str(e),
                }

    return None


def call_anthropic_api(
    system_prompt: str,
    user_content: str,
    temperature: float = 0.1,
    max_tokens: int = 4000,
    retries: int = 3,
) -> Optional[Dict[str, Any]]:
    """Call Anthropic API with retry logic"""
    if not anthropic_client:
        print("Warning: Anthropic API key not found, skipping Anthropic call")
        return None

    for attempt in range(retries):
        try:
            response = anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                system=system_prompt,  # Use system parameter for Anthropic
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": user_content}],
            )

            # Extract text content from response
            content = ""
            for block in response.content:
                if (
                    hasattr(block, "type")
                    and block.type == "text"
                    and hasattr(block, "text")
                ):
                    content += block.text

            if not content:
                print(f"Anthropic returned empty content (attempt {attempt + 1})")
                if attempt == retries - 1:
                    return {
                        "source": "claude-3-5-sonnet-20241022",
                        "hyperparameters": {
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                        },
                        "raw_response": None,
                        "parsed_response": None,
                        "error": "Empty response content",
                    }
                continue

            content = content.strip()

            # Remove markdown code block delimiters if present
            if content.startswith("```json") and content.endswith("```"):
                content = content[7:-3].strip()
            elif content.startswith("```") and content.endswith("```"):
                content = content[3:-3].strip()

            # Try to parse as JSON to validate format
            try:
                result = json.loads(content)

                # Validate that all ADUs actually exist in the source text
                validated_result = validate_adus_in_text(result, user_content)

                return {
                    "source": "claude-3-5-sonnet-20241022",
                    "hyperparameters": {
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                    "raw_response": content,
                    "parsed_response_unvalidated": result,
                    "parsed_response": validated_result,
                }
            except json.JSONDecodeError as e:
                print(
                    f"Anthropic response is not valid JSON (attempt {attempt + 1}): {e}"
                )
                print(f"DEBUG - Raw Anthropic response: {repr(content)}")
                if attempt == retries - 1:
                    return {
                        "source": "claude-3-5-sonnet-20241022",
                        "hyperparameters": {
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                        },
                        "raw_response": content,
                        "parsed_response": None,
                        "error": f"Invalid JSON: {e}",
                    }

        except Exception as e:
            print(f"Anthropic API error (attempt {attempt + 1}): {e}")
            if attempt < retries - 1:
                time.sleep(2**attempt)  # Exponential backoff
            else:
                return {
                    "source": "claude-3-5-sonnet-20241022",
                    "hyperparameters": {
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                    "raw_response": None,
                    "parsed_response": None,
                    "error": str(e),
                }

    return None


def save_arguments(
    args_path: Path,
    identifier: str,
    system_prompt: str,
    user_content: str,
    responses: List[Dict[str, Any]],
    overwrite: bool,
):
    """Save the full prompt and responses to the arguments file"""
    args_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "identifier": identifier,
        "system_prompt": system_prompt,
        "user_content": user_content,
        "responses": responses,
        "timestamp": time.time(),
    }

    if overwrite or not args_path.exists():
        mode = "w"
        output_data = [data]
    else:
        mode = "r+"
        try:
            with open(args_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = [existing_data]
        except (json.JSONDecodeError, FileNotFoundError):
            existing_data = []

        output_data = existing_data + [data]
        mode = "w"

    with open(args_path, mode, encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def save_labels(
    labels_path: Path, identifier: str, responses: List[Dict[str, Any]], overwrite: bool
):
    """Save extracted claims (not premises) to the labels file"""
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    labels_data = {}

    for response in responses:
        if response.get("parsed_response") and "ADUs" in response["parsed_response"]:
            adus = response["parsed_response"]["ADUs"]
            source = response["source"]
            hyperparams = response.get("hyperparameters", {})

            # Only show hyperparameters if they're not the default values
            temp = hyperparams.get("temperature", 0.1)
            tokens = hyperparams.get("max_tokens", 4000)
            if temp != 0.1 or tokens != 4000:
                source_with_params = f"{source} (T:{temp:.2f}, MT:{tokens})"
            else:
                source_with_params = source

            for adu_id, adu_data in adus.items():
                text = adu_data.get("text", "")
                label_type = adu_data.get("label", "")

                if text and label_type == "claim":
                    # Only save claims to the labels file, not premises
                    labels_data[text] = source_with_params

    if not labels_data:
        print(f"No valid claims extracted for {identifier}")
        return

    if overwrite or not labels_path.exists():
        output_data = labels_data
    else:
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                if isinstance(existing_data, dict):
                    existing_data.update(labels_data)
                    output_data = existing_data
                else:
                    output_data = labels_data
        except (json.JSONDecodeError, FileNotFoundError):
            output_data = labels_data

    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def load_sample_identifiers(sample_path: Path) -> List[str]:
    """Load paper identifiers from the sample CSV file"""
    identifiers = []

    if not sample_path.exists():
        raise FileNotFoundError(f"Sample file not found: {sample_path}")

    with open(sample_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Validate that 'identifier' column exists
        fieldnames = reader.fieldnames or []
        if "identifier" not in fieldnames:
            raise ValueError(
                f"CSV file must contain an 'identifier' column. Found columns: {fieldnames}"
            )

        for row in reader:
            identifier = row["identifier"].strip()
            if identifier:  # Skip empty identifiers
                identifiers.append(identifier)

    return identifiers


def main():
    """Main function to process papers and generate annotations"""
    parser = argparse.ArgumentParser(
        description="Generate argumentative annotations for papers using AI models"
    )

    parser.add_argument(
        "-s",
        "--sample-size",
        type=int,
        required=True,
        help="Sample size for processing",
    )
    parser.add_argument("-f", "--sample", type=str, help="Input sample CSV file")
    parser.add_argument(
        "-n", "--number", type=int, help="Number of papers to process (for testing)"
    )
    parser.add_argument(
        "-x",
        "--overwrite",
        action="store_true",
        help="Overwrite output files (default: append)",
    )
    parser.add_argument(
        "--md", action="store_true", help="Prioritize .md files over .txt files"
    )
    parser.add_argument(
        "-p",
        "--papers",
        type=str,
        default="data/docling",
        help="Directory containing papers",
    )
    parser.add_argument("-l", "--labels", type=str, help="Output directory for labels")
    parser.add_argument(
        "-a", "--arguments", type=str, help="Output directory for arguments"
    )
    parser.add_argument(
        "-r", "--retries", type=int, default=3, help="Number of retries per API call"
    )
    parser.add_argument(
        "--skip-not-found",
        action="store_true",
        help="Skip if docling file for paper identifier is not found",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip papers that already have output files generated",
    )

    # Hyperparameter search arguments
    parser.add_argument(
        "--min-temperature",
        type=float,
        help="Minimum temperature for hyperparameter search",
    )
    parser.add_argument(
        "--max-temperature",
        type=float,
        help="Maximum temperature for hyperparameter search",
    )
    parser.add_argument(
        "--min-tokens", type=int, help="Minimum max_tokens for hyperparameter search"
    )
    parser.add_argument(
        "--max-tokens", type=int, help="Maximum max_tokens for hyperparameter search"
    )
    parser.add_argument(
        "--num-combinations",
        type=int,
        default=2,
        help="Number of hyperparameter combinations to try per API",
    )

    args = parser.parse_args()

    # Set default paths based on sample size
    if not args.sample:
        args.sample = f"data/annotations/samples/n{args.sample_size}/sample.csv"

    if not args.labels:
        args.labels = "data/annotations/labels"

    if not args.arguments:
        args.arguments = "data/annotations/arguments"

    # Convert to Path objects
    sample_path = Path(args.sample)
    papers_dir = Path(args.papers)
    labels_dir = Path(args.labels)
    arguments_dir = Path(args.arguments)

    # Load prompt template (this will be the system prompt)
    try:
        system_prompt = load_prompt_template()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Load sample identifiers
    try:
        identifiers = load_sample_identifiers(sample_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Limit number of papers if specified
    if args.number:
        identifiers = identifiers[: args.number]
        print(
            f"Limited to processing {len(identifiers)} papers (--number={args.number})"
        )

    print(f"Processing {len(identifiers)} papers from {sample_path}")
    print(f"Papers directory: {papers_dir}")
    print(f"Labels output: {labels_dir}")
    print(f"Arguments output: {arguments_dir}")
    print(f"File preference: {'Markdown (.md)' if args.md else 'Text (.txt)'}")

    # Check if hyperparameter search should be enabled
    enable_hyperparam_search = (
        args.min_temperature is not None
        and args.max_temperature is not None
        and args.min_tokens is not None
        and args.max_tokens is not None
        and (
            args.min_temperature != args.max_temperature
            or args.min_tokens != args.max_tokens
        )
    )

    if enable_hyperparam_search:
        print(f"Hyperparameter search enabled")
        print(f"Temperature range: {args.min_temperature}-{args.max_temperature}")
        print(f"Max tokens range: {args.min_tokens}-{args.max_tokens}")
        print(f"Combinations per API: {args.num_combinations}")

        # Generate hyperparameter combinations
        hyperparams = generate_hyperparameters(
            args.min_temperature,
            args.max_temperature,
            args.min_tokens,
            args.max_tokens,
            args.num_combinations,
        )
        print(f"Generated hyperparameter combinations: {hyperparams}")
    else:
        print(f"Using default hyperparameters: temperature=0.1, max_tokens=4000")
        hyperparams = [{"temperature": 0.1, "max_tokens": 4000}]

    # Process each paper
    for i, identifier in enumerate(identifiers, 1):
        print(f"\nProcessing {i}/{len(identifiers)}: {identifier}")

        # Check if output files already exist (skip-existing logic)
        args_path = arguments_dir / f"{identifier}.json"
        labels_path = labels_dir / f"{identifier}.json"
        
        if args.skip_existing and args_path.exists() and labels_path.exists():
            print(f"Skipping {identifier}: output files already exist")
            continue

        # Load paper content (this will be the user content)
        paper_content = load_paper_content(papers_dir, identifier, args.md)
        if not paper_content:
            if args.skip_not_found:
                print(f"Skipping {identifier}: paper content not found")
                continue
            else:
                print(f"Error: paper content not found for {identifier}")
                sys.exit(1)

        # Call APIs
        responses = []

        # Call OpenAI with hyperparameter combinations
        for j, params in enumerate(hyperparams, 1):
            if enable_hyperparam_search:
                print(
                    f"Calling OpenAI API for {identifier} (combination {j}/{len(hyperparams)}) - T:{params['temperature']:.2f}, MT:{params['max_tokens']}"
                )
            else:
                print(f"Calling OpenAI API for {identifier}...")
            oai_response = call_openai_api(
                system_prompt,
                paper_content,
                params["temperature"],
                params["max_tokens"],
                args.retries,
            )
            if oai_response:
                responses.append(oai_response)

        # Call Anthropic with hyperparameter combinations
        for j, params in enumerate(hyperparams, 1):
            if enable_hyperparam_search:
                print(
                    f"Calling Anthropic API for {identifier} (combination {j}/{len(hyperparams)}) - T:{params['temperature']:.2f}, MT:{params['max_tokens']}"
                )
            else:
                print(f"Calling Anthropic API for {identifier}...")
            anthropic_response = call_anthropic_api(
                system_prompt,
                paper_content,
                params["temperature"],
                params["max_tokens"],
                args.retries,
            )
            if anthropic_response:
                responses.append(anthropic_response)

        if not responses:
            print(f"Warning: No successful API responses for {identifier}")
            continue

        # Save results
        save_arguments(
            args_path,
            identifier,
            system_prompt,
            paper_content,
            responses,
            args.overwrite,
        )
        save_labels(labels_path, identifier, responses, args.overwrite)

        if enable_hyperparam_search:
            print(f"Saved results for {identifier} ({len(responses)} total responses)")
        else:
            print(f"Saved results for {identifier}")

    print(f"\nCompleted processing {len(identifiers)} papers")


if __name__ == "__main__":
    main()
