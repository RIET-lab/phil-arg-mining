# pre-trained-am-model.py

"""
This script uses pre-trained models (listed below) to extract 
and classify ADUs and their relationships from text.

Model Information:
----
The modeld can be found here: 
https://huggingface.co/ArneBinder/sam-adur-sciarg/tree/main,
https://huggingface.co/ArneBinder/sam-are-sciarg/tree/main,
The model is based on: 
https://aclanthology.org/2022.emnlp-main.713/,
And trained with: 
https://huggingface.co/datasets/DFKI-SLT/sciarg, 
https://aclanthology.org/W18-5206.pdf.

Original Labels:
----
ADU Classification: background_claim, own_claim, or data.
    background_claim: "an argumentative statement relating to the background of 
        [the] authors' work".
    own_claim: "an argumentative statement that closely relates to the authors' 
        own work".
    data: "represents a fact that serves as evidence for or against a claim. 
        Note that references or (factual) examples can also serve as data".
Relation Classification: supports, contradicts, parts_of_same, semantically_same.
    supports: "relation holds between components a and b if the assumed veracity 
        of b increases with the veracity of a".
    contradicts: "relation holds between components a and b if the assumed 
        veracity of b decreases with the veracity of a".
    semantically_same: "relation is annotated between two mentions of 
        effectively the same claim or data component. This relation can be seen 
        as argument coreference, analogous to entity and event coreference".
    parts_of_same: (Not in orignal acl anthology paper) depicts two ADUs that
        are part of the same component but were split during ADU extraction.

Label Translations:
----
We make no distinction between background and own claims:
    background_claim -> claim
    own_claim -> claim
Data is relabeled to "premise":
    data -> premise
Contradicts relations become attacks:
    contradicts -> attacks
We drop semantically_same relations:
    semantically_same -> [dropped]
We merge parts_of_same relations:
    parts_of_same -> [merged]

"""

import argparse
import json
import glob
import warnings
import os
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Suppress common pipeline warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_ie")
warnings.filterwarnings("ignore", category=UserWarning, module="pie_modules")

from pie_modules.models import * 
from pie_modules.taskmodules import *
from pytorch_ie.models import *
from pytorch_ie.taskmodules import *
from pytorch_ie import AutoPipeline
from pytorch_ie.annotations import LabeledSpan
from pytorch_ie.documents import (
    TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
)

# Example text for testing
EXAMPLE_TEXT = """
## 3. Comparing Equal Treatment and Different Senses

According to Different Senses there are two distinct, incommensurable senses of rationality that both apply to belief states. One is the same as the sense of rationality that applies to states like (A) -(E). The other is a distinctively epistemic sense of rationality that applies only to belief states. 9

Philosophers who use the phrase ' epistemic rationality ' do not all have the same idea in mind. Some take it to be a sort of rationality essentially connected with truth (perhaps via the notion of evidence, or the notion of a reliable belief-forming process); others take it to be essentially connected with knowledge; and so forth. What is important here is just that, whatever epistemic rationality is, according to Different Senses a single belief state can be rational in this epistemic sense while being irrational in the other (non-epistemic) sense, and vice versa. According to Equal Treatment, on the other hand, every belief state is univocally either rational or not. 10 I will argue that this is an important advantage that Equal Treatment has over Different Senses.

It is natural to think of rationality as constituting an ideal to which one might aspire, and by which one might be guided. But rationality is poorly suited to play this role if it consists of two different voices urging us in incompatible directions. It is metaphysically impossible to believe P while also failing to believe P. And yet according to Different Senses, there are cases in which, unless one does so, one is bound to be irrational in at least one sense. How could one coherently aspire to satisfy the demands of rationality when doing so (in both of its senses) would be metaphysically impossible?

According to Equal Treatment, on the other hand, rationality speaks in a single voice. Insofar as we look to rationality for guidance, and conceive of it as an ideal to which one might coherently aspire, this tells in favor of Equal Treatment and against Different Senses. Moreover, as we saw above, Equal Treatment is a more uni /uniFB01 ed approach to rationality: it treats all states, including belief states, alike, whereas Different Senses postulates special complexities in the case of belief that do not arise for states like (A) -(E). 11
"""


class ArgumentMiner:
    """Main class for ADU extraction and classification."""
    
    def __init__(self, device: int = -1, verbose_debug: bool = False):
        """
        Initialize ADU extractor with pre-trained model.
        
        Args:
            device: Device for inference (-1 for CPU, 0+ for GPU)
            verbose_debug: Enable verbose debugging output
        """
        self.device = device
        self.ner_pipeline = None
        self.re_pipeline = None
        self.verbose_debug = verbose_debug
        
    def load_models(self):
        """Load the pre-trained ADU detection and relation extraction models."""
        self.ner_pipeline = AutoPipeline.from_pretrained(
            "ArneBinder/sam-adur-sciarg",
            device=self.device,
            taskmodule_kwargs={"combine_token_scores_method": "product"}
        )
        
        self.re_pipeline = AutoPipeline.from_pretrained(
            "ArneBinder/sam-are-sciarg", 
            device=self.device,
            taskmodule_kwargs={"collect_statistics": False}
        )
    
    def create_document(self, text: str, doc_id: str = "document") -> TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions:
        """
        Create a document from text with a single partition covering the whole text.
        
        Args:
            text: Input text
            doc_id: Document identifier
            
        Returns:
            Document object ready for processing
        """
        document = TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions(text)
        document.id = doc_id
        document.metadata = {}
        document.labeled_partitions.append(LabeledSpan(start=0, end=len(text), label="text"))
        return document
    
    def extract_adus(self, document) -> None:
        """
        Extract ADUs and relations from document in-place.
        
        Args:
            document: Document to process
        """
        if not self.ner_pipeline or not self.re_pipeline:
            raise RuntimeError("Models not loaded. Call load_models() first.")
            
        # Clear existing annotations
        document.labeled_spans.clear()
        document.labeled_spans.predictions.clear()
        document.binary_relations.clear()
        document.binary_relations.predictions.clear()
        
        # Extract ADUs
        self.ner_pipeline(document, inplace=True)
        
        # Move predicted entities to main layer for relation extraction
        predicted_entities = list(document.labeled_spans.predictions)
        document.labeled_spans.clear()
        document.labeled_spans.predictions.clear()
        document.labeled_spans.extend(predicted_entities)
        
        # Extract relations
        self.re_pipeline(document, inplace=True)
        
        # Move all predicted entities back to predictions layer for final output
        final_entities = list(document.labeled_spans)
        document.labeled_spans.clear()
        document.labeled_spans.predictions.clear()
        
        # Create new span objects to avoid any reference issues
        for entity in final_entities:
            new_span = LabeledSpan(
                start=entity.start,
                end=entity.end,
                label=entity.label,
                score=getattr(entity, 'score', 1.0)
            )
            document.labeled_spans.predictions.append(new_span)
        
        # Move all relations to final output (no filtering)
        all_relations = list(document.binary_relations.predictions)
        document.binary_relations.clear()
        document.binary_relations.predictions.clear()
        document.binary_relations.predictions.extend(all_relations)
        
        if self.verbose_debug:
            print(f"DEBUG: Final result - {len(document.labeled_spans.predictions)} ADUs, {len(document.binary_relations.predictions)} relations")
            
            # Show relation type breakdown
            relation_counts = {}
            for rel in document.binary_relations.predictions:
                relation_counts[rel.label] = relation_counts.get(rel.label, 0) + 1
            print(f"DEBUG: Relation types: {relation_counts}")
    
    def _translate_labels(self, label: str) -> str:
        """
        Translate model labels according to documented mapping.
        Handles both ADU labels and relation labels.
        
        Args:
            label: Original model label
            
        Returns:
            Translated label
        """
        # Label translations as documented
        label_map = {
            # ADU label translations
            "background_claim": "claim",
            "own_claim": "claim", 
            "data": "premise",
            # Relation label translations
            "contradicts": "attacks"
        }
        return label_map.get(label, label)

    def process_text(self, text: str, doc_id: str = "document") -> Dict[str, Any]:
        """
        Process text and return extracted ADUs and relations.
        
        Args:
            text: Input text
            doc_id: Document identifier
            
        Returns:
            Dictionary containing ADUs, relations, and statistics
        """
        document = self.create_document(text, doc_id)
        self.extract_adus(document)
        
        adus = []
        for adu in document.labeled_spans.predictions:
            translated_label = self._translate_labels(adu.label)
            adus.append({
                "text": document.text[adu.start:adu.end],
                "label": translated_label,
                "original_label": adu.label,
                "start": adu.start,
                "end": adu.end,
                "score": getattr(adu, "score", None)
            })
        
        relations = []
        for relation in document.binary_relations.predictions:
            translated_label = self._translate_labels(relation.label)
            relations.append({
                "head_text": document.text[relation.head.start:relation.head.end],
                "tail_text": document.text[relation.tail.start:relation.tail.end],
                "label": translated_label,
                "original_label": relation.label,
                "head_start": relation.head.start,
                "head_end": relation.head.end,
                "tail_start": relation.tail.start,
                "tail_end": relation.tail.end,
                "score": getattr(relation, "score", None)
            })
        
        # Generate statistics
        adu_counts = {}
        for adu in adus:
            adu_counts[adu["label"]] = adu_counts.get(adu["label"], 0) + 1
        
        return {
            "document_id": doc_id,
            "adus": adus,
            "relations": relations,
            "statistics": {
                "total_adus": len(adus),
                "total_relations": len(relations),
                "adu_types": adu_counts
            }
        }


def find_input_files(input_path: str) -> List[Path]:
    """
    Find input text files based on path or glob pattern.
    Excludes .doctags.txt files.
    
    Args:
        input_path: File path, directory path, or glob pattern
        
    Returns:
        List of Path objects for text files
    """
    path = Path(input_path)
    
    if path.is_file() and path.suffix == '.txt' and not path.name.endswith('.doctags.txt'):
        return [path]
    elif path.is_dir():
        files = [f for f in path.glob('*.txt') if not f.name.endswith('.doctags.txt')]
        return sorted(files)
    else:
        # Treat as glob pattern
        files = [Path(p) for p in glob.glob(input_path) 
                if p.endswith('.txt') and not p.endswith('.doctags.txt')]
        return sorted(files)


def print_results(results: Dict[str, Any]):
    """Print formatted results to console."""
    print(f"\nDocument: {results['document_id']}")
    print("=" * 80)
    
    print(f"\nExtracted ADUs ({results['statistics']['total_adus']}):")
    for i, adu in enumerate(results['adus'], 1):
        print(f"{i}. [{adu['label']}] \"{adu['text']}\"")
    
    print(f"\nExtracted Relations ({results['statistics']['total_relations']}):")
    for i, rel in enumerate(results['relations'], 1):
        print(f"{i}. \"{rel['head_text']}\" --[{rel['label']}]--> \"{rel['tail_text']}\"")
    
    print(f"\nStatistics:")
    print(f"- Total ADUs: {results['statistics']['total_adus']}")
    print(f"- Total Relations: {results['statistics']['total_relations']}")
    if results['statistics']['adu_types']:
        print("- ADU types:")
        for adu_type, count in results['statistics']['adu_types'].items():
            print(f"  * {adu_type}: {count}")


def save_results(results: List[Dict[str, Any]], output_path: str):
    """Save results to JSON file. Creates file and parent directories if needed."""
    output_path_obj = Path(output_path)
    output_dir = output_path_obj.parent
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path_obj, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Argument mining with sam-adur-sciarg and sam-are-sciarg",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '-i', '--input',
        help="Input file, directory, or glob pattern for .txt files"
    )
    parser.add_argument(
        '-o', '--output',
        help="Output JSON file path (default: sciarg-am-results-YYYY-MM-DD.json)"
    )
    parser.add_argument(
        '-d', '--device',
        type=int,
        default=0,
        help="Device for inference (-1 for CPU, 0+ for GPU)"
    )
    parser.add_argument(
        '-e', '--example',
        action='store_true',
        help="Run with example text instead of file input"
    )
    parser.add_argument(
        '-l', '--limit',
        type=int,
        help="Maximum number of files to process"
    )
    parser.add_argument(
        '-v', '--verbose-debug',
        action='store_true',
        help="Enable verbose debugging output"
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Initialize extractor
    extractor = ArgumentMiner(device=args.device, verbose_debug=args.verbose_debug)
    extractor.load_models()
    
    results = []
    
    if args.example:
        # Process example text
        result = extractor.process_text(EXAMPLE_TEXT, "example")
        results.append(result)
        print_results(result)
    else:
        if not args.input:
            print("Error: --input required unless using --example")
            return
            
        # Find input files
        input_files = find_input_files(args.input)
        if not input_files:
            print(f"No .txt files found for input: {args.input}")
            return
                
        # Apply limit if specified
        if args.limit:
            input_files = input_files[:args.limit]
            print(f"Processing first {len(input_files)} files due to limit")
        
        # Process each file
        for file_path in input_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                result = extractor.process_text(text, str(file_path))
                results.append(result)
                print_results(result)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
    
    # Save results if output specified
    if args.output or not args.example:
        if not args.output:
            today = datetime.now().strftime("%Y-%m-%d-%H-%M")
            output_path = f"sciarg-am-results-{today}.json"
        else:
            output_path = args.output
            
        save_results(results, output_path)


if __name__ == "__main__":
    main()