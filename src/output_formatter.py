# src/output_formatter.py

import re
from typing import List, Dict, Any, Tuple
from datetime import datetime
from src.nlp_utils import abstractive_summary

class OutputFormatter:
    def __init__(self):
        self.max_sections_to_output = 5      # Top N sections to include in final output
        self.max_subsections_per_section = 3 # Top M subsections from each relevant section
        self.max_refined_text_sentences = 3  # Max sentences for refined text

    def _get_sentences(self, text: str) -> List[str]:
        """Simple sentence tokenizer."""
        # Split by common sentence terminators, keeping the terminator
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _select_relevant_sentences(self, text: str, keywords: List[str], max_sentences: int) -> str:
        """
        Selects sentences most relevant to keywords using a simple scoring.
        Prioritizes sentences with more keywords, and first/last sentences.
        """
        sentences = self._get_sentences(text)
        if not sentences:
            return ""

        # Normalize keywords for matching
        normalized_keywords = [re.sub(r'[^a-z0-9\s]', '', k.lower()) for k in keywords]

        scored_sentences = []
        for i, sentence in enumerate(sentences):
            normalized_sentence = re.sub(r'[^a-z0-9\s]', '', sentence.lower())
            
            score = 0
            # Keyword presence count
            for keyword in normalized_keywords:
                if keyword in normalized_sentence:
                    score += normalized_sentence.count(keyword)
            
            # Boost for first/last sentences (often summary or concluding thoughts)
            if i == 0 or i == len(sentences) - 1:
                score += 0.5 # Small boost

            scored_sentences.append((score, i, sentence)) # Store score, original index, and sentence

        # Sort by score (descending), then by original index (ascending for order)
        scored_sentences.sort(key=lambda x: (x[0], -x[1]), reverse=True) # Sort by score, then reverse index to keep original order if scores are same
        
        # Select top N sentences, maintaining original order
        selected_sentences_with_indices = sorted(scored_sentences[:max_sentences], key=lambda x: x[1])
        
        return " ".join([s[2] for s in selected_sentences_with_indices])


    def format_output(
        self,
        document_paths: List[str],
        persona_description: str,
        job_to_be_done: str,
        ranked_sections: List[Dict],
        ranked_subsections: List[Dict],
        processing_start_time: float,
        persona_features: Dict[str, Any] # Need persona features for keywords
    ) -> Dict[str, Any]:
        """
        Formats the processed data into the specified challenge1b_output.json structure.
        """
        # --- Metadata ---
        output_metadata = {
            "input_documents": [doc_path.split('/')[-1] for doc_path in document_paths],
            "persona": persona_description,
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": datetime.fromtimestamp(processing_start_time).strftime("%Y-%m-%d %H:%M:%S IST")
        }

        # --- Extracted Sections ---
        # Select top N relevant sections
        final_extracted_sections = []
        for i, section_data in enumerate(ranked_sections[:self.max_sections_to_output]):
            summary = self._select_relevant_sentences(
                section_data.get("raw_text_content", ""),
                persona_features.get("keywords", []),
                self.max_refined_text_sentences
            )
            abs_summary = abstractive_summary(section_data.get("raw_text_content", ""))
            final_extracted_sections.append({
                "document": section_data["document"],
                "page_number": f"{section_data['page_range'][0]}-{section_data['page_range'][1]}" if section_data['page_range'][0] != section_data['page_range'][1] else str(section_data['page_range'][0]),
                "section_title": section_data["section_title"],
                "importance_rank": i + 1, # Re-rank based on final selection
                "explanation": section_data.get("explanation", ""),
                "summary": summary,
                "abstractive_summary": abs_summary
            })

        # --- Sub-section Analysis ---
        final_subsection_analysis = []
        relevant_section_titles = {s['section_title'] for s in final_extracted_sections}
        filtered_subsections = [
            ss for ss in ranked_subsections
            if ss['section_title'] in relevant_section_titles
        ]
        subsections_by_parent_section = {}
        for ss in filtered_subsections:
            key = (ss['document'], ss['section_title'])
            if key not in subsections_by_parent_section:
                subsections_by_parent_section[key] = []
            subsections_by_parent_section[key].append(ss)
        for key, subsections in subsections_by_parent_section.items():
            subsections.sort(key=lambda x: x["importance_score"], reverse=True)
            for ss_data in subsections[:self.max_subsections_per_section]:
                refined_text = self._select_relevant_sentences(
                    ss_data['subsection_text'],
                    persona_features.get("keywords", []),
                    self.max_refined_text_sentences
                )
                abs_summary = abstractive_summary(ss_data['subsection_text'])
                final_subsection_analysis.append({
                    "document": ss_data["document"],
                    "page_number": ss_data["page_number"],
                    "refined_text": refined_text,
                    "explanation": ss_data.get("explanation", ""),
                    "summary": refined_text,
                    "abstractive_summary": abs_summary
                })
        final_subsection_analysis.sort(key=lambda x: (x['document'], x['page_number']))
        final_output_json = {
            "metadata": output_metadata,
            "extracted_sections": final_extracted_sections,
            "sub_section_analysis": final_subsection_analysis
        }
        return final_output_json