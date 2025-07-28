from typing import Dict, List, Any
CONCEPT_LEXICON: Dict[str, str] = {
    "methodology": "RESEARCH_METHOD",
    "methods": "RESEARCH_METHOD",
    "approach": "RESEARCH_METHOD",
    "algorithm": "RESEARCH_METHOD",
    "model": "RESEARCH_METHOD",
    "data set": "DATA_INFO",
    "datasets": "DATA_INFO",
    "corpus": "DATA_INFO",
    "performance": "EVALUATION_METRIC",
    "benchmark": "EVALUATION_METRIC",
    "benchmarks": "EVALUATION_METRIC",
    "accuracy": "EVALUATION_METRIC",
    "precision": "EVALUATION_METRIC",
    "recall": "EVALUATION_METRIC",
    "f1-score": "EVALUATION_METRIC",
    "results": "STUDY_OUTCOME",
    "discussion": "STUDY_OUTCOME",
    "conclusion": "STUDY_OUTCOME",
    "abstract": "SUMMARY",
    "introduction": "OVERVIEW",
    "future work": "FUTURE_DIRECTION",
    "limitations": "FUTURE_DIRECTION",
    "revenue": "FINANCIALS",
    "profit": "FINANCIALS",
    "income": "FINANCIALS",
    "expenditure": "FINANCIALS",
    "investments": "FINANCIALS",
    "rd": "FINANCIALS",
    "market positioning": "BUSINESS_STRATEGY",
    "strategy": "BUSINESS_STRATEGY",
    "trends": "TREND_ANALYSIS",
    "outlook": "TREND_ANALYSIS",
    "growth": "TREND_ANALYSIS",
    "risk": "RISK_ANALYSIS",
    "reaction": "CHEM_MECHANISM",
    "mechanisms": "CHEM_MECHANISM",
    "kinetics": "CHEM_KINETICS",
    "catalysis": "CHEM_KINETICS",
    "molecules": "CHEM_COMPOUND",
    "compounds": "CHEM_COMPOUND",
    "organic": "CHEM_DOMAIN",
    "inorganic": "CHEM_DOMAIN",
    "concepts": "BASIC_CONCEPT",
    "definitions": "BASIC_CONCEPT",
    "equations": "BASIC_CONCEPT",
    "examples": "ILLUSTRATION",
}
PERSONA_HEURISTICS: Dict[str, Dict[str, Any]] = {
    "PhD Researcher in Computational Biology_Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks": {
        "priority_section_titles": ["Methodology", "Methods", "Experimental Setup", "Results", "Evaluation", "Performance", "Data", "Datasets", "Discussion"],
        "high_value_content_keywords": [
            "proposed method", "novel approach", "architecture", "algorithm", "model design",
            "dataset used", "data collection", "data preprocessing", "benchmark dataset",
            "accuracy", "precision", "recall", "f1-score", "AUC", "ROC", "loss", "metrics",
            "ablation study", "comparison with", "state-of-the-art"
        ],
        "de_emphasize_content_keywords": ["general overview", "historical context of X", "simple definition of"],
        "structural_boosts": {
            "H1": 1.5,
            "H2": 1.2,
            "lists": 0.8,
            "tables": 1.0,
        },
        "initial_content_focus": 0.5,
    },
    "Investment Analyst_Analyze revenue trends, R&D investments, and market positioning strategies": {
        "priority_section_titles": ["Financial Statements", "Income Statement", "Balance Sheet", "Cash Flow", "Management Discussion & Analysis", "Revenue", "R&D Expenses", "Market Share", "Strategy", "Outlook", "Risk Factors"],
        "high_value_content_keywords": [
            "revenue growth", "net income", "gross margin", "operating expenses", "capex",
            "R&D spend", "investment in", "market share", "competitive landscape",
            "strategic initiatives", "forward-looking statements", "CAGR", "EBITDA", "EPS",
            "year-over-year", "quarterly results", "trend", "outlook"
        ],
        "de_emphasize_content_keywords": ["employee benefits", "corporate social responsibility (CSR) overview"],
        "structural_boosts": {
            "H1": 1.2,
            "H2": 1.5,
            "lists": 1.0,
            "tables": 2.0,
        },
        "initial_content_focus": 0.2,
    },
    "Undergraduate Chemistry Student_Identify key concepts and mechanisms for exam preparation on reaction kinetics": {
        "priority_section_titles": ["Reaction Mechanisms", "Reaction Kinetics", "Key Concepts", "Definitions", "Examples", "Problem Solving", "Reaction Types"],
        "high_value_content_keywords": [
            "mechanism", "rate law", "order of reaction", "activation energy", "transition state",
            "catalyst", "intermediate", "rate-determining step", "equilibrium constant",
            "Arrhenius equation", "reaction pathway", "bimolecular", "unimolecular", "stereochemistry"
        ],
        "de_emphasize_content_keywords": ["historical overview of chemistry", "biographical notes"],
        "structural_boosts": {
            "H1": 1.0,
            "H2": 1.2,
            "lists": 1.5,
            "tables": 1.0,
        },
        "initial_content_focus": 0.8,
    },
}
def get_persona_job_key(persona_description: str, job_to_be_done: str) -> str:
    return f"{persona_description}_{job_to_be_done}"
def get_heuristics(persona_description: str, job_to_be_done: str) -> Dict[str, Any]:
    key = get_persona_job_key(persona_description, job_to_be_done)
    return PERSONA_HEURISTICS.get(key, {})