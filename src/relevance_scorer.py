# src/relevance_scorer.py

import re
from typing import List, Dict, Any, Tuple
from src.data_models import Document, Section, Subsection
from collections import Counter
from src.nlp_utils import semantic_similarity, extract_entities, extract_topics

class RelevanceScorer:
    def __init__(self):
        # You can define default weights here, or pass them in if dynamic
        self.default_weights = {
            "keyword_match": 1.0,
            "concept_match": 0.5,
            "structural_importance": 0.7,
            "persona_heuristic_boost": 2.0, # High weight for explicit persona rules
            "proximity_to_jtd": 1.5,
            "title_match_multiplier": 3.0, # Titles are very important
        }

    def _normalize_text(self, text: str) -> str:
        """Converts text to lowercase and removes non-alphanumeric chars for consistent matching."""
        return re.sub(r'[^a-z0-9\s]', '', text.lower())

    def _calculate_keyword_score(self, text: str, keywords: List[str]) -> float:
        """Calculates a score based on keyword presence and frequency."""
        normalized_text = self._normalize_text(text)
        text_words = normalized_text.split()
        score = 0.0
        for keyword in keywords:
            if "_" in keyword: # Handle n-grams
                if keyword.replace("_", " ") in normalized_text:
                    score += 1.5 # Boost for n-gram matches
            elif keyword in text_words:
                score += text_words.count(keyword) # Count occurrences
        return score

    def _calculate_concept_score(self, text: str, concepts: List[str], concept_lexicon: Dict[str, str]) -> float:
        """Calculates a score based on presence of mapped concepts."""
        normalized_text = self._normalize_text(text)
        score = 0.0
        # Iterate through the concept lexicon to find associated terms in text
        for term, concept_tag in concept_lexicon.items():
            if concept_tag in concepts and term in normalized_text:
                score += normalized_text.count(term) * 0.5 # Less weight than direct keywords
        return score

    def _apply_structural_boost(self, section_or_subsection: Any, heuristics: Dict[str, Any]) -> float:
        """Applies boosts based on structural elements."""
        boost = 1.0
        structural_boosts = heuristics.get("structural_boosts", {})

        if isinstance(section_or_subsection, Section):
            level_key = f"H{section_or_subsection.level}"
            boost *= structural_boosts.get(level_key, 1.0)    
            normalized_title = self._normalize_text(section_or_subsection.title)
            for title_keyword in heuristics.get("priority_section_titles", []):
                if self._normalize_text(title_keyword) in normalized_title:
                    boost *= 1.5 
        
        elif isinstance(section_or_subsection, Subsection):
            if section_or_subsection.is_list_item:
                boost *= structural_boosts.get("lists", 1.0)
            if section_or_subsection.is_table_candidate: 
                boost *= structural_boosts.get("tables", 1.0)
        
        return boost

    def _apply_persona_heuristics(self, text: str, heuristics: Dict[str, Any]) -> float:
        """Applies boosts/penalties based on persona-specific content keywords."""
        normalized_text = self._normalize_text(text)
        score_modifier = 0.0

        for keyword in heuristics.get("high_value_content_keywords", []):
            if self._normalize_text(keyword) in normalized_text:
                score_modifier += 1.0 

        for keyword in heuristics.get("de_emphasize_content_keywords", []):
            if self._normalize_text(keyword) in normalized_text:
                score_modifier -= 0.5 # Penalize for de-emphasized terms

        return score_modifier

    def score_and_rank(self, processed_documents: List[Document], persona_features: Dict[str, Any], concept_lexicon: Dict[str, str]) -> Tuple[List[Dict], List[Dict]]:
        """
        Calculates relevance scores for sections and subsections and ranks them.
        Now also returns an 'explanation' for each item.
        """
        all_scored_sections = []
        all_scored_subsections = []

        keywords = persona_features.get("keywords", [])
        concepts = persona_features.get("concepts", [])
        heuristics = persona_features.get("heuristics", {})
        jtd_raw = persona_features.get("job_to_be_done_raw_text", "").lower()

        persona_job_text = persona_features.get("persona_job_text", None)
        if not persona_job_text:
            persona_job_text = persona_features.get("persona_description", "") + ' ' + persona_features.get("job_to_be_done_raw_text", "")
        persona_entities = set(extract_entities(persona_job_text))
        persona_intent = persona_features.get("intent", "general")
        persona_topics = set()
        try:
            persona_topics = set([w for t in extract_topics([persona_job_text], num_topics=1, num_words=5) for w in t[1].split(' + ')])
        except Exception:
            pass

        for doc_obj in processed_documents:
            for section in doc_obj.sections:
                section_score = 0.0
                explanation_parts = []

                # 1. Keyword and Concept Score for the whole section
                kw_score = self._calculate_keyword_score(section.text_content, keywords) * self.default_weights["keyword_match"]
                if kw_score > 0:
                    explanation_parts.append(f"Matched persona/job keywords (score: {kw_score:.2f})")
                section_score += kw_score

                concept_score = self._calculate_concept_score(section.text_content, concepts, concept_lexicon) * self.default_weights["concept_match"]
                if concept_score > 0:
                    explanation_parts.append(f"Matched domain concepts (score: {concept_score:.2f})")
                section_score += concept_score

                # Semantic similarity
                sem_score = semantic_similarity(persona_job_text, section.text_content)
                section_score += sem_score * 2.0  # Weight can be tuned
                explanation_parts.append(f"Semantic similarity: {sem_score:.2f}")

                # NER entity overlap
                section_entities = set(extract_entities(section.text_content))
                common_entities = persona_entities & section_entities
                if common_entities:
                    section_score += len(common_entities) * 1.5  # Boost per entity
                    explanation_parts.append(f"Entity overlap: {', '.join(common_entities)}")

                
                section_topics = set()
                try:
                    section_topics = set([w for t in extract_topics([section.text_content], num_topics=1, num_words=5) for w in t[1].split(' + ')])
                except Exception:
                    pass
                common_topics = persona_topics & section_topics
                if common_topics:
                    section_score += len(common_topics) * 1.0
                    explanation_parts.append(f"Topic overlap: {', '.join(common_topics)}")

                # 2. Structural Importance Boost
                struct_boost = self._apply_structural_boost(section, heuristics) * self.default_weights["structural_importance"]
                if struct_boost != 1.0:
                    explanation_parts.append(f"Structural boost applied (x{struct_boost:.2f})")
                section_score *= struct_boost

                
                heur_score = self._apply_persona_heuristics(section.text_content, heuristics) * self.default_weights["persona_heuristic_boost"]
                if heur_score > 0:
                    explanation_parts.append(f"Persona-specific high-value content found (score: {heur_score:.2f})")
                elif heur_score < 0:
                    explanation_parts.append(f"De-emphasized content found (score: {heur_score:.2f})")
                section_score += heur_score
                normalized_title = self._normalize_text(section.title)
                jtd_keywords_in_title = sum(1 for kw in keywords if kw in normalized_title)
                if jtd_keywords_in_title > 0:
                    explanation_parts.append(f"Section title matches job keywords (boost: {jtd_keywords_in_title * self.default_weights['title_match_multiplier']:.2f})")
                section_score += jtd_keywords_in_title * self.default_weights["title_match_multiplier"]

                if section.title == "Initial Content" and heuristics:
                    section_score *= heuristics.get("initial_content_focus", 1.0)
                    explanation_parts.append("Initial content focus boost applied")

                section.importance_score = section_score
                all_scored_sections.append({
                    "document": doc_obj.file_name,
                    "page_range": section.page_range,
                    "section_title": section.title,
                    "importance_score": section.importance_score,
                    "raw_text_content": section.text_content, # For summarization
                    "explanation": "; ".join(explanation_parts) if explanation_parts else "General relevance to persona/job."
                })
                for subsection in section.subsections:
                    subsection_score = 0.0
                    ss_explanation_parts = []

                    kw_score = self._calculate_keyword_score(subsection.text_content, keywords) * self.default_weights["keyword_match"]
                    if kw_score > 0:
                        ss_explanation_parts.append(f"Matched persona/job keywords (score: {kw_score:.2f})")
                    subsection_score += kw_score

                    concept_score = self._calculate_concept_score(subsection.text_content, concepts, concept_lexicon) * self.default_weights["concept_match"]
                    if concept_score > 0:
                        ss_explanation_parts.append(f"Matched domain concepts (score: {concept_score:.2f})")
                    subsection_score += concept_score

                    heur_score = self._apply_persona_heuristics(subsection.text_content, heuristics) * self.default_weights["persona_heuristic_boost"]
                    if heur_score > 0:
                        ss_explanation_parts.append(f"Persona-specific high-value content found (score: {heur_score:.2f})")
                    elif heur_score < 0:
                        ss_explanation_parts.append(f"De-emphasized content found (score: {heur_score:.2f})")
                    subsection_score += heur_score

                    struct_boost = self._apply_structural_boost(subsection, heuristics) * self.default_weights["structural_importance"]
                    if struct_boost != 1.0:
                        ss_explanation_parts.append(f"Structural boost applied (x{struct_boost:.2f})")
                    subsection_score *= struct_boost
                    sem_score_ss = semantic_similarity(persona_job_text, subsection.text_content)
                    subsection_score += sem_score_ss * 2.0
                    ss_explanation_parts.append(f"Semantic similarity: {sem_score_ss:.2f}")

                    ss_entities = set(extract_entities(subsection.text_content))
                    common_ss_entities = persona_entities & ss_entities
                    if common_ss_entities:
                        subsection_score += len(common_ss_entities) * 1.5
                        ss_explanation_parts.append(f"Entity overlap: {', '.join(common_ss_entities)}")

                    normalized_ss_text = self._normalize_text(subsection.text_content)
                    if jtd_raw:
                        jtd_word_count = len(self._normalize_text(jtd_raw).split())
                        if jtd_word_count > 0:
                            common_words = len(set(normalized_ss_text.split()) & set(self._normalize_text(jtd_raw).split()))
                            if common_words > 0:
                                boost = (common_words / jtd_word_count) * self.default_weights["proximity_to_jtd"]
                                ss_explanation_parts.append(f"Job-to-be-done keyword density boost ({boost:.2f})")
                                subsection_score += boost

                    subsection.importance_score = subsection_score
                    all_scored_subsections.append({
                        "document": doc_obj.file_name,
                        "page_number": subsection.page_number,
                        "section_title": section.title,
                        "subsection_text": subsection.text_content,
                        "importance_score": subsection.importance_score,
                        "explanation": "; ".join(ss_explanation_parts) if ss_explanation_parts else "General relevance to persona/job."
                    })

        # Sort and rank sections
        ranked_sections = sorted(all_scored_sections, key=lambda x: x["importance_score"], reverse=True)
        for i, item in enumerate(ranked_sections):
            item["importance_rank"] = i + 1

        ranked_subsections = sorted(all_scored_subsections, key=lambda x: x["importance_score"], reverse=True)
        for i, item in enumerate(ranked_subsections):
            item["importance_rank"] = i + 1

        return ranked_sections, ranked_subsections