"""
Argument Mining Models Package

This package provides model-facing components for argument mining:
- End2End: LLM-based end-to-end argument mining with optional RAG and CoT
- ADUR: Argumentative Discourse Unit Recognition via pre-trained classification
- ARE: Argumentative Relation Extraction via pre-trained classification
- Supporting utilities for generation, retrieval (RAG), and chain-of-thought reasoning
"""

from .models import End2End, ADUR, ARE
from .generator import (
    load_generator_model,
    load_generator_model_from_config,
    build_input_ids,
    generate,
    generate_chat,
    get_device_and_dtype,
)
from .rag import RAG
from .cot import CoT

__all__ = [
    # Main model classes
    "End2End",
    "ADUR", 
    "ARE",
    # Generator utilities
    "load_generator_model",
    "load_generator_model_from_config", 
    "build_input_ids",
    "generate",
    "generate_chat",
    "get_device_and_dtype",
    # Supporting components
    "RAG",
    "CoT",
]
