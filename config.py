"""
Configuration file for Persona-Driven Document Intelligence System.
Contains all model parameters, paths, and settings for the custom NLP model.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / "src"
TEST_DOCS_DIR = PROJECT_ROOT / "test_docs"
OUTPUT_DIR = PROJECT_ROOT

# Model configuration
MODEL_CONFIG = {
    "transformer_model": "sentence-transformers/all-MiniLM-L6-v2",
    "max_length": 512,
    "hidden_size": 384,
    "num_heads": 8,
    "dropout": 0.1,
    "temperature": 0.1,  # For contrastive learning
}

# Training configuration
TRAINING_CONFIG = {
    "epochs": 10,
    "batch_size": 4,
    "learning_rate": 1e-4,
    "contrastive_weight": 0.1,
    "num_support_examples": 5,
}

# NLP configuration
NLP_CONFIG = {
    "spacy_model": "en_core_web_sm",
    "num_topics": 3,
    "num_words_per_topic": 5,
    "max_summary_length": 60,
    "min_summary_length": 20,
    "top_keywords": 15,
}

# Scoring weights
SCORING_WEIGHTS = {
    "keyword_match": 1.0,
    "concept_match": 1.5,
    "structural_importance": 2.0,
    "persona_heuristics": 1.5,
    "semantic_similarity": 2.0,
    "entity_overlap": 1.5,
    "topic_alignment": 1.0,
    "few_shot_score": 1.0,
}

# Output configuration
OUTPUT_CONFIG = {
    "max_sections": 10,
    "max_subsections": 20,
    "max_refined_text_sentences": 5,
    "output_filename": "challenge1b_output.json",
}

# Performance constraints
CONSTRAINTS = {
    "max_model_size_mb": 1000,  # 1GB limit
    "max_processing_time_seconds": 60,
    "cpu_only": True,
    "no_internet": True,
}

# Test cases
TEST_CASES = {
    "academic_research": {
        "persona": "PhD Researcher in Computational Biology",
        "job": "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks"
    },
    "business_analysis": {
        "persona": "Investment Analyst", 
        "job": "Analyze revenue trends, R&D investments, and market positioning strategies"
    },
    "educational_content": {
        "persona": "Undergraduate Chemistry Student",
        "job": "Identify key concepts and mechanisms for exam preparation on reaction kinetics"
    }
}

# Zero-shot classification labels
ZERO_SHOT_LABELS = [
    "methods", "results", "limitations", "conclusion", "introduction",
    "background", "discussion", "future_work", "related_work", "experiments"
]

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "doc_intelligence.log"
}

# Model file paths
MODEL_PATHS = {
    "enhanced_model": "enhanced_persona_document_model.pth",
    "base_model": "persona_document_model.pth",
}

# Feature flags
FEATURES = {
    "few_shot_learning": True,
    "hierarchical_attention": True,
    "contrastive_learning": True,
    "explainable_ai": True,
    "abstractive_summarization": True,
    "cross_document_aggregation": True,
    "ai_question_answering": True,
    "zero_shot_classification": True,
    "sentiment_analysis": True,
}
