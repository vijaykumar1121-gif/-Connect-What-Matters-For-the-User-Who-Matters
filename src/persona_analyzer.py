# src/persona_analyzer.py

import re
from typing import Dict, List, Any
# from nltk.tokenize import word_tokenize # Uncomment if you install NLTK
# from nltk.stem import PorterStemmer # Uncomment if you install NLTK
from src.heuristics import CONCEPT_LEXICON, get_heuristics # Import our heuristics

class PersonaAnalyzer:
    def __init__(self):
        # self.stemmer = PorterStemmer() # Uncomment if using NLTK stemmer

        # A basic set of stopwords (can be expanded)
        self.stopwords = set([
            "a", "an", "the", "and", "or", "is", "are", "was", "were", "be", "been", "being",
            "of", "in", "on", "at", "for", "with", "as", "by", "from", "to", "but", "what",
            "how", "which", "who", "when", "where", "why", "about", "this", "that", "these",
            "those", "its", "it's", "their", "they", "them", "their", "you", "your", "yours",
            "he", "she", "it", "his", "her", "him", "my", "me", "we", "us", "our", "ours",
            "i", "am", "have", "has", "had", "do", "does", "did", "not", "no", "yes", "so",
            "can", "will", "would", "should", "could", "get", "go", "just", "like", "make",
            "see", "take", "up", "down", "out", "in", "on", "off", "over", "under", "again",
            "further", "then", "once", "here", "there", "when", "where", "why", "all", "any",
            "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "only",
            "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "don", "should", "now"
        ])


    def _preprocess_text(self, text: str) -> List[str]:
        """Tokenizes text and converts to lowercase."""
        text = text.lower()
        # Remove non-alphanumeric characters, keep spaces, then split
        text = re.sub(r'[^a-z0-9\s]', '', text)
        tokens = text.split()
        # tokens = [self.stemmer.stem(word) for word in tokens if word not in self.stopwords] # If using NLTK
        tokens = [word for word in tokens if word not in self.stopwords] # Simple removal
        return tokens

    def _get_ngrams(self, tokens: List[str], n: int) -> List[str]:
        """Generates n-grams from a list of tokens."""
        if len(tokens) < n:
            return []
        return ["_".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    def extract_persona_job_features(self, persona_description: str, job_to_be_done: str) -> Dict[str, Any]:
        """
        Extracts keywords, concepts, and loads heuristics for a given persona and job.
        """
        # 1. Process Job-to-be-Done and Persona Description
        jtd_tokens = self._preprocess_text(job_to_be_done)
        persona_tokens = self._preprocess_text(persona_description)

        # Combine for a comprehensive set of keywords
        combined_keywords_raw = list(set(jtd_tokens + persona_tokens))
        
        # Add common n-grams that might be important (e.g., "financial statements")
        # You might want to filter these later based on CONCEPT_LEXICON or PERSONA_HEURISTICS
        combined_keywords_raw.extend(self._get_ngrams(jtd_tokens, 2))
        combined_keywords_raw.extend(self._get_ngrams(jtd_tokens, 3))
        combined_keywords_raw.extend(self._get_ngrams(persona_tokens, 2))
        combined_keywords_raw.extend(self._get_ngrams(persona_tokens, 3))


        # 2. Map Keywords to Concepts
        extracted_concepts = set()
        for keyword in combined_keywords_raw:
            if keyword in CONCEPT_LEXICON:
                extracted_concepts.add(CONCEPT_LEXICON[keyword])
        
        # 3. Load Persona-Specific Heuristics
        specific_heuristics = get_heuristics(persona_description, job_to_be_done)

        return {
            "keywords": combined_keywords_raw, # All extracted keywords and n-grams
            "concepts": list(extracted_concepts), # Unique high-level concepts
            "heuristics": specific_heuristics # Persona-specific rules/weights
        }