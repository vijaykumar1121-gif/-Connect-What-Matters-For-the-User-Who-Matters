# Approach Explanation: Persona-Driven Document Intelligence

## Overview
Our solution intelligently analyzes a collection of documents and extracts the most relevant sections and sub-sections based on a given persona and their job-to-be-done. The system is generic and can handle diverse document types, personas, and tasks.

## Pipeline Steps

1. **Document Parsing & Structure Extraction**
   - Each PDF is parsed to extract its hierarchical structure (sections, sub-sections, page numbers, and text).
   - This is handled by the `PDFParser` class, which outputs structured `Document` objects.

2. **Persona & Job Feature Engineering**
   - The persona description and job-to-be-done are analyzed to extract key features, focus areas, and intent.
   - The `PersonaAnalyzer` class processes these inputs to generate a feature set that guides relevance scoring.

3. **Relevance Scoring & Ranking**
   - Each section and sub-section of the documents is scored for relevance to the persona and job.
   - The `RelevanceScorer` uses a hybrid approach:
     - **Keyword/Concept Matching:** Uses a domain-agnostic concept lexicon to match important terms.
     - **Proximity Scoring:** Considers how closely the text matches the job-to-be-done.
     - **Heuristics:** Additional rules to prioritize sections like "Conclusion" or "Summary" for certain personas.
   - Sections and sub-sections are ranked by their computed importance.

4. **Output Formatting**
   - The top-ranked sections and sub-sections are formatted into a JSON output as per the challenge requirements.
   - The output includes metadata, extracted sections (with document, page, title, rank), and sub-section analysis (with refined text).

## Performance & Constraints
- The solution is optimized to run on CPU only and completes processing of 3–5 documents in under 60 seconds.
- All models and libraries used are lightweight and do not exceed the 1GB memory constraint.
- No internet access is required during execution.

## Generality
- The pipeline is fully generic and can process documents from any domain.
- The persona and job analysis is flexible, allowing the system to adapt to different user needs and document types.

## Conclusion
This approach ensures that users receive highly relevant, persona-tailored insights from large document collections, regardless of the domain or task. 