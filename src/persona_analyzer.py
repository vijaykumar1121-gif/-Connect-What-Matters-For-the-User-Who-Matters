import re
from typing import Dict, List, Any
from src.heuristics import CONCEPT_LEXICON, get_heuristics
from src.nlp_utils import extract_keywords, classify_intent
class PersonaAnalyzer:
    def __init__(self):
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
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        tokens = text.split()
        tokens = [word for word in tokens if word not in self.stopwords]
        return tokens
    def _get_ngrams(self, tokens: List[str], n: int) -> List[str]:
        if len(tokens) < n:
            return []
        return ["_".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    def extract_persona_job_features(self, persona_description: str, job_to_be_done: str) -> dict:
        features = {}
        combined_text = persona_description + ' ' + job_to_be_done
        features['keywords'] = extract_keywords(combined_text, top_n=15)
        features['intent'] = classify_intent(job_to_be_done)
        jtd_tokens = self._preprocess_text(job_to_be_done)
        persona_tokens = self._preprocess_text(persona_description)
        combined_keywords_raw = list(set(jtd_tokens + persona_tokens))
        combined_keywords_raw.extend(self._get_ngrams(jtd_tokens, 2))
        combined_keywords_raw.extend(self._get_ngrams(jtd_tokens, 3))
        combined_keywords_raw.extend(self._get_ngrams(persona_tokens, 2))
        combined_keywords_raw.extend(self._get_ngrams(persona_tokens, 3))
        extracted_concepts = set()
        for keyword in combined_keywords_raw:
            if keyword in CONCEPT_LEXICON:
                extracted_concepts.add(CONCEPT_LEXICON[keyword])
        specific_heuristics = get_heuristics(persona_description, job_to_be_done)
        features["keywords"] = combined_keywords_raw
        features["concepts"] = list(extracted_concepts)
        features["heuristics"] = specific_heuristics
        return features