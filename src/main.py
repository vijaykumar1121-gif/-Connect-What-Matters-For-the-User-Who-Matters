# src/main.py

import os
import json
import time
from typing import List, Dict, Any
from src.pdf_parser import PDFParser
from src.data_models import Document, Section, Subsection
from src.persona_analyzer import PersonaAnalyzer
from src.relevance_scorer import RelevanceScorer
from src.heuristics import CONCEPT_LEXICON
from src.output_formatter import OutputFormatter # <--- ADD THIS LINE

def run_analysis(document_paths: List[str], persona_description: str, job_to_be_done: str) -> Dict[str, Any]:
    """
    Main function to run the persona-driven document analysis.
    """
    start_time = time.time()
    
    # 1. Document Pre-processing & Structural Analysis
    print("Stage 1: Processing documents and extracting structure...")
    parser = PDFParser()
    processed_documents: List[Document] = []
    for doc_path in document_paths:
        print(f"  Parsing: {doc_path}")
        doc_data = parser.parse_pdf(doc_path)
        processed_documents.append(doc_data)
    
    # 2. Persona-Job Feature Engineering
    print("Stage 2: Analyzing persona and job-to-be-done...")
    persona_analyzer = PersonaAnalyzer()
    
    persona_features: Dict[str, Any] = persona_analyzer.extract_persona_job_features(
        persona_description, job_to_be_done
    )
    persona_features["job_to_be_done_raw_text"] = job_to_be_done # Add raw JTD for proximity scoring
    
    # 3. Hybrid Scoring & Hierarchical Prioritization
    print("Stage 3: Scoring and prioritizing sections...")
    scorer = RelevanceScorer()
    ranked_sections, ranked_subsections = scorer.score_and_rank(
        processed_documents, persona_features, CONCEPT_LEXICON
    )

    # 4. Output Generation (Final JSON Format)
    print("Stage 4: Generating final output in specified JSON format...")
    formatter = OutputFormatter()
    final_output = formatter.format_output(
        document_paths,
        persona_description,
        job_to_be_done,
        ranked_sections,
        ranked_subsections,
        start_time,
        persona_features # Pass persona_features for refined text generation
    )
    
    end_time = time.time()
    print(f"Analysis completed in {end_time - start_time:.2f} seconds.")
    
    return final_output # Return the final formatted output

if __name__ == "__main__":
    # --- SAMPLE INPUTS FOR TESTING ---
    dummy_pdf_path_1 = "test_docs/research_paper_sample.pdf"
    
    if not os.path.exists("test_docs"):
        os.makedirs("test_docs")
        print("Created 'test_docs' directory. Please place some PDF files inside for testing.")

    sample_document_paths = [dummy_pdf_path_1]
    
    for path in sample_document_paths:
        if not os.path.exists(path):
            print(f"WARNING: Dummy PDF '{path}' not found. Please create it or place real PDFs for testing.")
            sample_document_paths = [p for p in sample_document_paths if os.path.exists(p)]

    # Test Case 1: Academic Research
    persona_tc1 = "PhD Researcher in Computational Biology"
    job_tc1 = "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks"

    # Test Case 2: Business Analysis
    persona_tc2 = "Investment Analyst"
    job_tc2 = "Analyze revenue trends, R&D investments, and market positioning strategies"

    # Test Case 3: Educational Content
    persona_tc3 = "Undergraduate Chemistry Student"
    job_tc3 = "Identify key concepts and mechanisms for exam preparation on reaction kinetics"

    # Choose which test case to run (uncomment one)
    current_persona = persona_tc1
    current_job = job_tc1
    # current_persona = persona_tc2
    # current_job = job_tc2
    # current_persona = persona_tc3
    # current_job = job_tc3


    if sample_document_paths:
        output = run_analysis(sample_document_paths, current_persona, current_job)
        output_filename = "challenge1b_output.json" # Final output name as per brief
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)
        print(f"\nFinal formatted output saved to '{output_filename}'")
    else:
        print("\nNo PDF documents found to process. Please add PDFs to the 'test_docs' directory.")