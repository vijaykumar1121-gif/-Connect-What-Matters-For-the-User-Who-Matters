#!/usr/bin/env python3
"""
Create Test PDFs Script
Generate proper test PDFs for the official test cases.
"""

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import os

def create_academic_research_pdfs():
    """Create PDFs for academic research test case."""
    print("📄 Creating Academic Research PDFs...")
    
    # PDF 1: Methodology Paper
    doc1 = SimpleDocTemplate("test_docs/graph_nn_methodology.pdf", pagesize=letter)
    story1 = []
    styles = getSampleStyleSheet()
    
    story1.append(Paragraph("Graph Neural Networks for Drug Discovery: A Comprehensive Methodology", styles['Title']))
    story1.append(Spacer(1, 12))
    
    story1.append(Paragraph("Abstract", styles['Heading1']))
    story1.append(Paragraph("This paper presents a novel methodology for applying Graph Neural Networks (GNNs) to drug discovery applications. We introduce a comprehensive framework that leverages molecular graph representations to predict drug-target interactions and molecular properties.", styles['Normal']))
    story1.append(Spacer(1, 12))
    
    story1.append(Paragraph("1. Introduction", styles['Heading1']))
    story1.append(Paragraph("Drug discovery is a complex process that requires understanding molecular interactions and predicting biological activities. Traditional methods rely on experimental screening, which is time-consuming and expensive. Graph Neural Networks offer a promising alternative by learning from molecular graph representations.", styles['Normal']))
    story1.append(Spacer(1, 12))
    
    story1.append(Paragraph("2. Methodology", styles['Heading1']))
    story1.append(Paragraph("Our methodology consists of three main components: molecular graph construction, GNN architecture design, and training strategy. We represent molecules as graphs where nodes represent atoms and edges represent chemical bonds.", styles['Normal']))
    story1.append(Spacer(1, 12))
    
    story1.append(Paragraph("2.1 Molecular Graph Construction", styles['Heading2']))
    story1.append(Paragraph("We construct molecular graphs using RDKit, where each atom is represented as a node with features including atomic number, formal charge, and hybridization state. Chemical bonds are represented as edges with bond type and stereochemistry information.", styles['Normal']))
    story1.append(Spacer(1, 12))
    
    story1.append(Paragraph("2.2 GNN Architecture", styles['Heading2']))
    story1.append(Paragraph("We employ a Graph Convolutional Network (GCN) with attention mechanisms. The network consists of multiple graph convolution layers followed by global pooling and fully connected layers for final prediction.", styles['Normal']))
    story1.append(Spacer(1, 12))
    
    story1.append(Paragraph("3. Datasets", styles['Heading1']))
    story1.append(Paragraph("We evaluate our methodology on three benchmark datasets: ChEMBL, BindingDB, and PubChem. These datasets contain millions of drug-target interaction pairs with associated experimental data.", styles['Normal']))
    story1.append(Spacer(1, 12))
    
    story1.append(Paragraph("4. Performance Benchmarks", styles['Heading1']))
    story1.append(Paragraph("Our GNN-based approach achieves state-of-the-art performance on drug-target interaction prediction, with AUC-ROC scores of 0.89 on ChEMBL and 0.91 on BindingDB. We also demonstrate superior performance in molecular property prediction tasks.", styles['Normal']))
    
    doc1.build(story1)
    
    # PDF 2: Dataset Paper
    doc2 = SimpleDocTemplate("test_docs/graph_nn_datasets.pdf", pagesize=letter)
    story2 = []
    
    story2.append(Paragraph("Comprehensive Datasets for Graph Neural Networks in Drug Discovery", styles['Title']))
    story2.append(Spacer(1, 12))
    
    story2.append(Paragraph("Abstract", styles['Heading1']))
    story2.append(Paragraph("We present a comprehensive collection of datasets specifically designed for training and evaluating Graph Neural Networks in drug discovery applications. These datasets cover various aspects of molecular property prediction and drug-target interaction.", styles['Normal']))
    story2.append(Spacer(1, 12))
    
    story2.append(Paragraph("1. Dataset Overview", styles['Heading1']))
    story2.append(Paragraph("Our dataset collection includes molecular structures, biological activities, and physicochemical properties. Each dataset is carefully curated and annotated for machine learning applications.", styles['Normal']))
    story2.append(Spacer(1, 12))
    
    story2.append(Paragraph("2. Molecular Property Datasets", styles['Heading1']))
    story2.append(Paragraph("We provide datasets for predicting molecular properties such as solubility, toxicity, and drug-likeness. These properties are crucial for drug development and safety assessment.", styles['Normal']))
    story2.append(Spacer(1, 12))
    
    story2.append(Paragraph("3. Drug-Target Interaction Datasets", styles['Heading1']))
    story2.append(Paragraph("Our interaction datasets contain experimentally validated drug-target pairs with associated binding affinities and inhibition constants. These datasets enable training of models for predicting novel drug-target interactions.", styles['Normal']))
    
    doc2.build(story2)
    
    # PDF 3: Benchmark Paper
    doc3 = SimpleDocTemplate("test_docs/graph_nn_benchmarks.pdf", pagesize=letter)
    story3 = []
    
    story3.append(Paragraph("Benchmarking Graph Neural Networks for Drug Discovery Applications", styles['Title']))
    story3.append(Spacer(1, 12))
    
    story3.append(Paragraph("Abstract", styles['Heading1']))
    story3.append(Paragraph("We present a comprehensive benchmarking study comparing various Graph Neural Network architectures for drug discovery tasks. Our evaluation covers multiple datasets and performance metrics to provide insights into the effectiveness of different approaches.", styles['Normal']))
    story3.append(Spacer(1, 12))
    
    story3.append(Paragraph("1. Benchmark Framework", styles['Heading1']))
    story3.append(Paragraph("We establish a standardized framework for evaluating GNN performance in drug discovery. This framework includes consistent data preprocessing, model training protocols, and evaluation metrics.", styles['Normal']))
    story3.append(Spacer(1, 12))
    
    story3.append(Paragraph("2. Model Comparison", styles['Heading1']))
    story3.append(Paragraph("We compare various GNN architectures including Graph Convolutional Networks, Graph Attention Networks, and GraphSAGE. Our results show that attention-based models generally outperform traditional convolution-based approaches.", styles['Normal']))
    story3.append(Spacer(1, 12))
    
    story3.append(Paragraph("3. Performance Metrics", styles['Heading1']))
    story3.append(Paragraph("We evaluate models using multiple metrics including AUC-ROC, precision-recall curves, and mean squared error. These metrics provide comprehensive assessment of model performance across different aspects of drug discovery.", styles['Normal']))
    
    doc3.build(story3)
    
    print("✅ Created 3 academic research PDFs")

def create_business_analysis_pdfs():
    """Create PDFs for business analysis test case."""
    print("📄 Creating Business Analysis PDFs...")
    
    # PDF 1: Tech Company Annual Report 2022
    doc1 = SimpleDocTemplate("test_docs/tech_company_2022.pdf", pagesize=letter)
    story1 = []
    styles = getSampleStyleSheet()
    
    story1.append(Paragraph("TechCorp Inc. Annual Report 2022", styles['Title']))
    story1.append(Spacer(1, 12))
    
    story1.append(Paragraph("Executive Summary", styles['Heading1']))
    story1.append(Paragraph("TechCorp Inc. achieved record revenue growth of 25% in 2022, driven by strong performance in cloud services and AI solutions. Our strategic investments in R&D have positioned us for continued market leadership.", styles['Normal']))
    story1.append(Spacer(1, 12))
    
    story1.append(Paragraph("Financial Performance", styles['Heading1']))
    story1.append(Paragraph("Total revenue reached $15.2 billion in 2022, up from $12.1 billion in 2021. Cloud services revenue grew 40% year-over-year, while AI solutions revenue increased 60%. Operating margin improved to 28% from 25% in 2021.", styles['Normal']))
    story1.append(Spacer(1, 12))
    
    story1.append(Paragraph("R&D Investments", styles['Heading1']))
    story1.append(Paragraph("We invested $2.3 billion in research and development in 2022, representing 15% of total revenue. Key focus areas include artificial intelligence, quantum computing, and cybersecurity. Our R&D team grew to 5,000 engineers and scientists.", styles['Normal']))
    story1.append(Spacer(1, 12))
    
    story1.append(Paragraph("Market Positioning", styles['Heading1']))
    story1.append(Paragraph("TechCorp maintains strong market positions in cloud infrastructure (35% market share), AI platforms (28% market share), and enterprise software (22% market share). We continue to expand into emerging markets and new technology sectors.", styles['Normal']))
    
    doc1.build(story1)
    
    # PDF 2: Tech Company Annual Report 2023
    doc2 = SimpleDocTemplate("test_docs/tech_company_2023.pdf", pagesize=letter)
    story2 = []
    
    story2.append(Paragraph("TechCorp Inc. Annual Report 2023", styles['Title']))
    story2.append(Spacer(1, 12))
    
    story2.append(Paragraph("Financial Highlights", styles['Heading1']))
    story2.append(Paragraph("Revenue grew 30% to $19.8 billion in 2023. Cloud services revenue reached $8.5 billion, up 45% year-over-year. AI solutions revenue doubled to $3.2 billion. Operating margin expanded to 32%.", styles['Normal']))
    story2.append(Spacer(1, 12))
    
    story2.append(Paragraph("Strategic Initiatives", styles['Heading1']))
    story2.append(Paragraph("We launched new AI-powered products and expanded our cloud infrastructure globally. Strategic acquisitions strengthened our position in cybersecurity and data analytics. International markets now represent 45% of total revenue.", styles['Normal']))
    story2.append(Spacer(1, 12))
    
    story2.append(Paragraph("R&D Strategy", styles['Heading1']))
    story2.append(Paragraph("R&D investment increased to $3.1 billion (16% of revenue). We established new research centers in Asia and Europe. Key breakthroughs in quantum computing and generative AI were achieved.", styles['Normal']))
    
    doc2.build(story2)
    
    # PDF 3: Tech Company Annual Report 2024
    doc3 = SimpleDocTemplate("test_docs/tech_company_2024.pdf", pagesize=letter)
    story3 = []
    
    story3.append(Paragraph("TechCorp Inc. Annual Report 2024", styles['Title']))
    story3.append(Spacer(1, 12))
    
    story3.append(Paragraph("Market Leadership", styles['Heading1']))
    story3.append(Paragraph("TechCorp achieved market leadership in AI platforms with 35% market share. Cloud infrastructure market share grew to 38%. We are the preferred partner for enterprise digital transformation initiatives.", styles['Normal']))
    story3.append(Spacer(1, 12))
    
    story3.append(Paragraph("Revenue Growth", styles['Heading1']))
    story3.append(Paragraph("Total revenue reached $25.7 billion, up 30% from 2023. AI solutions revenue grew 80% to $5.8 billion. Cloud services revenue increased 50% to $12.7 billion. Operating margin reached 35%.", styles['Normal']))
    story3.append(Spacer(1, 12))
    
    story3.append(Paragraph("Innovation Pipeline", styles['Heading1']))
    story3.append(Paragraph("Our innovation pipeline includes next-generation AI models, quantum computing applications, and advanced cybersecurity solutions. We expect these innovations to drive continued growth in 2025 and beyond.", styles['Normal']))
    
    doc3.build(story3)
    
    print("✅ Created 3 business analysis PDFs")

def create_educational_content_pdfs():
    """Create PDFs for educational content test case."""
    print("📄 Creating Educational Content PDFs...")
    
    # PDF 1: Organic Chemistry Chapter 1
    doc1 = SimpleDocTemplate("test_docs/chemistry_chapter1.pdf", pagesize=letter)
    story1 = []
    styles = getSampleStyleSheet()
    
    story1.append(Paragraph("Organic Chemistry: Chapter 1 - Introduction to Reaction Kinetics", styles['Title']))
    story1.append(Spacer(1, 12))
    
    story1.append(Paragraph("1.1 What is Reaction Kinetics?", styles['Heading1']))
    story1.append(Paragraph("Reaction kinetics is the study of the rates of chemical reactions and the factors that influence them. Understanding kinetics is crucial for predicting reaction outcomes and optimizing reaction conditions.", styles['Normal']))
    story1.append(Spacer(1, 12))
    
    story1.append(Paragraph("1.2 Rate Laws and Rate Constants", styles['Heading1']))
    story1.append(Paragraph("The rate law expresses the relationship between reaction rate and reactant concentrations. For a reaction A + B → C, the rate law is typically: rate = k[A]^m[B]^n, where k is the rate constant and m, n are reaction orders.", styles['Normal']))
    story1.append(Spacer(1, 12))
    
    story1.append(Paragraph("1.3 Key Concepts for Exam Preparation", styles['Heading1']))
    story1.append(Paragraph("Students should understand: (1) How to determine reaction order from experimental data, (2) The relationship between rate constant and temperature (Arrhenius equation), (3) How catalysts affect reaction rates, and (4) The concept of activation energy.", styles['Normal']))
    
    doc1.build(story1)
    
    # PDF 2: Organic Chemistry Chapter 2
    doc2 = SimpleDocTemplate("test_docs/chemistry_chapter2.pdf", pagesize=letter)
    story2 = []
    
    story2.append(Paragraph("Organic Chemistry: Chapter 2 - Reaction Mechanisms", styles['Title']))
    story2.append(Spacer(1, 12))
    
    story2.append(Paragraph("2.1 Understanding Reaction Mechanisms", styles['Heading1']))
    story2.append(Paragraph("A reaction mechanism describes the step-by-step process by which reactants are converted to products. Each step involves the formation or breaking of chemical bonds, and understanding these steps is essential for predicting reaction outcomes.", styles['Normal']))
    story2.append(Spacer(1, 12))
    
    story2.append(Paragraph("2.2 SN1 and SN2 Mechanisms", styles['Heading1']))
    story2.append(Paragraph("Nucleophilic substitution reactions can proceed via SN1 (unimolecular) or SN2 (bimolecular) mechanisms. SN1 reactions involve a carbocation intermediate and show first-order kinetics, while SN2 reactions occur in one step and show second-order kinetics.", styles['Normal']))
    story2.append(Spacer(1, 12))
    
    story2.append(Paragraph("2.3 Factors Affecting Mechanism Choice", styles['Heading1']))
    story2.append(Paragraph("The choice between SN1 and SN2 mechanisms depends on: (1) The nature of the leaving group, (2) The structure of the alkyl halide (primary, secondary, or tertiary), (3) The nucleophilicity of the nucleophile, and (4) The solvent used.", styles['Normal']))
    
    doc2.build(story2)
    
    # PDF 3: Organic Chemistry Chapter 3
    doc3 = SimpleDocTemplate("test_docs/chemistry_chapter3.pdf", pagesize=letter)
    story3 = []
    
    story3.append(Paragraph("Organic Chemistry: Chapter 3 - Energy Diagrams and Transition States", styles['Title']))
    story3.append(Spacer(1, 12))
    
    story3.append(Paragraph("3.1 Energy Diagrams", styles['Heading1']))
    story3.append(Paragraph("Energy diagrams show the energy changes that occur during a chemical reaction. They help visualize the activation energy, transition states, and overall energy change of the reaction.", styles['Normal']))
    story3.append(Spacer(1, 12))
    
    story3.append(Paragraph("3.2 Transition State Theory", styles['Heading1']))
    story3.append(Paragraph("Transition state theory explains how reactions proceed through high-energy intermediate states. The transition state represents the highest energy point along the reaction coordinate and determines the reaction rate.", styles['Normal']))
    story3.append(Spacer(1, 12))
    
    story3.append(Paragraph("3.3 Exam Preparation Focus", styles['Heading1']))
    story3.append(Paragraph("For exam preparation, students should be able to: (1) Draw energy diagrams for simple reactions, (2) Identify transition states and intermediates, (3) Calculate activation energies from rate data, and (4) Predict how changes in conditions affect reaction rates.", styles['Normal']))
    
    doc3.build(story3)
    
    print("✅ Created 3 educational content PDFs")

def main():
    """Create all test PDFs."""
    print("📄 Creating Test PDFs for Official Test Cases")
    print("=" * 60)
    
    # Ensure test_docs directory exists
    if not os.path.exists("test_docs"):
        os.makedirs("test_docs")
        print("Created test_docs directory")
    
    # Create PDFs for each test case
    create_academic_research_pdfs()
    create_business_analysis_pdfs()
    create_educational_content_pdfs()
    
    print("\n🎉 All test PDFs created successfully!")
    print("📁 Check the test_docs/ directory for the new PDF files")
    print("\n💡 Now you can run the official test cases with proper PDFs:")
    print("   python test_official_cases.py")

if __name__ == "__main__":
    main() 