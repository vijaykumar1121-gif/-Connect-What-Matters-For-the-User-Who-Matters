import os
import json
import time
from typing import List, Dict, Any
from src.pdf_parser import PDFParser
from src.data_models import Document, Section, Subsection
from src.persona_analyzer import PersonaAnalyzer
from src.relevance_scorer import RelevanceScorer
from src.heuristics import CONCEPT_LEXICON
from src.output_formatter import OutputFormatter
from src.nlp_utils import aggregate_entities, aggregate_topics, answer_question, zero_shot_classify, sentiment_subjectivity
from src.model import PersonaDocumentModel
from src.trainer import train_enhanced_model

def run_analysis(document_paths: List[str], persona_description: str, job_to_be_done: str) -> Dict[str, Any]:
    start_time = time.time()
    print("Stage 1: Processing documents and extracting structure...")
    parser = PDFParser()
    processed_documents: List[Document] = []
    for doc_path in document_paths:
        print(f"  Parsing: {doc_path}")
        doc_data = parser.parse_pdf(doc_path)
        processed_documents.append(doc_data)
    print("Stage 2: Analyzing persona and job-to-be-done...")
    persona_analyzer = PersonaAnalyzer()
    persona_features: Dict[str, Any] = persona_analyzer.extract_persona_job_features(
        persona_description, job_to_be_done
    )
    persona_features["job_to_be_done_raw_text"] = job_to_be_done
    print("Stage 3: Scoring and prioritizing sections...")
    scorer = RelevanceScorer()
    ranked_sections, ranked_subsections = scorer.score_and_rank(
        processed_documents, persona_features, CONCEPT_LEXICON
    )
    print("Stage 4: Generating final output in specified JSON format...")
    formatter = OutputFormatter()
    final_output = formatter.format_output(
        document_paths,
        persona_description,
        job_to_be_done,
        ranked_sections,
        ranked_subsections,
        start_time,
        persona_features
    )
    end_time = time.time()
    print(f"Analysis completed in {end_time - start_time:.2f} seconds.")
    return final_output

if __name__ == "__main__":
    dummy_pdf_path_1 = "test_docs/research_paper_sample.pdf"
    if not os.path.exists("test_docs"):
        os.makedirs("test_docs")
        print("Created 'test_docs' directory. Please place some PDF files inside for testing.")
    sample_document_paths = [dummy_pdf_path_1]
    for path in sample_document_paths:
        if not os.path.exists(path):
            print(f"WARNING: Dummy PDF '{path}' not found. Please create it or place real PDFs for testing.")
            sample_document_paths = [p for p in sample_document_paths if os.path.exists(p)]
    persona_tc1 = "PhD Researcher in Computational Biology"
    job_tc1 = "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks"
    persona_tc2 = "Investment Analyst"
    job_tc2 = "Analyze revenue trends, R&D investments, and market positioning strategies"
    persona_tc3 = "Undergraduate Chemistry Student"
    job_tc3 = "Identify key concepts and mechanisms for exam preparation on reaction kinetics"
    current_persona = persona_tc1
    current_job = job_tc1
    if sample_document_paths:
        model_path = "enhanced_persona_document_model.pth"
        if not os.path.exists(model_path):
            print("Training enhanced custom NLP model...")
            print("Features: Few-shot learning, hierarchical attention, contrastive learning")
            model = train_enhanced_model(epochs=5, batch_size=2)
        else:
            print("Loading pre-trained enhanced model...")
            model = PersonaDocumentModel()
        output = run_analysis(sample_document_paths, current_persona, current_job)
        support_examples = [
            {
                'persona_text': current_persona,
                'job_text': current_job,
                'document_text': 'This section contains highly relevant methodology and experimental setup information.',
                'relevance_score': 0.95
            },
            {
                'persona_text': current_persona,
                'job_text': current_job,
                'document_text': 'The results section presents key findings and performance metrics.',
                'relevance_score': 0.90
            }
        ]
        print("\nMaking enhanced predictions with custom NLP model...")
        print("Using: Few-shot learning, hierarchical attention, contrastive learning")
        for section in output.get('extracted_sections', [])[:3]:
            prediction = model.predict(
                current_persona,
                current_job,
                section.get('summary', ''),
                support_examples=support_examples
            )
            section['enhanced_model_score'] = prediction['relevance_score']
            section['base_score'] = prediction['base_score']
            section['few_shot_score'] = prediction['few_shot_score']
            section['enhanced_explanation'] = prediction['explanation']
        all_texts = []
        for doc_path in sample_document_paths:
            with open(doc_path, 'rb') as f:
                try:
                    from PyPDF2 import PdfReader
                    reader = PdfReader(f)
                    all_texts.append(" ".join(page.extract_text() or '' for page in reader.pages))
                except Exception:
                    pass
        entity_summary = aggregate_entities(all_texts)
        topic_summary = aggregate_topics(all_texts, num_topics=3, num_words=5)
        example_question = "What are the main findings?"
        qa_answer = answer_question(example_question, " ".join(all_texts))
        labels = ["methods", "results", "limitations", "conclusion", "introduction"]
        for section in output.get('extracted_sections', []):
            section['zero_shot_labels'] = zero_shot_classify(section.get('summary', ''), labels)
            section['sentiment_subjectivity'] = sentiment_subjectivity(section.get('summary', ''))
        for sub in output.get('sub_section_analysis', []):
            sub['zero_shot_labels'] = zero_shot_classify(sub.get('refined_text', ''), labels)
            sub['sentiment_subjectivity'] = sentiment_subjectivity(sub.get('refined_text', ''))
        output['entity_summary'] = entity_summary
        output['topic_summary'] = topic_summary
        output['qa_example'] = {"question": example_question, "answer": qa_answer}
        output_filename = "challenge1b_output.json"
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)
        print(f"\nFinal formatted output saved to '{output_filename}'")
        print("Enhanced features included: Few-shot learning, hierarchical attention, contrastive learning")
    else:
        print("\nNo PDF documents found to process. Please add PDFs to the 'test_docs' directory.")