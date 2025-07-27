#!/usr/bin/env python3
"""
Quick Start Script for Persona-Driven Document Intelligence System.
Automatically sets up the environment and runs the system with sample data.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        "torch", "transformers", "spacy", "nltk", "sentence_transformers",
        "scikit-learn", "textblob", "PyPDF2", "fitz", "psutil"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    return missing_packages

def install_dependencies():
    """Install missing dependencies."""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def download_models():
    """Download required NLP models."""
    print("\n🤖 Downloading NLP models...")
    
    try:
        # Download spaCy model
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("✅ spaCy model downloaded")
        
        # Download NLTK data
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK data downloaded")
        
        return True
    except Exception as e:
        print(f"❌ Failed to download models: {e}")
        return False

def create_sample_pdf():
    """Create a sample PDF for testing if none exists."""
    test_docs_dir = Path("test_docs")
    test_docs_dir.mkdir(exist_ok=True)
    
    pdf_files = list(test_docs_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("\n📄 Creating sample PDF for testing...")
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            
            sample_pdf_path = test_docs_dir / "sample_research_paper.pdf"
            
            c = canvas.Canvas(str(sample_pdf_path), pagesize=letter)
            c.drawString(100, 750, "Sample Research Paper")
            c.drawString(100, 730, "Abstract")
            c.drawString(100, 710, "This is a sample research paper for testing the document intelligence system.")
            c.drawString(100, 690, "It contains various sections that can be analyzed by the custom NLP model.")
            
            c.drawString(100, 650, "1. Introduction")
            c.drawString(100, 630, "This section introduces the research methodology and experimental setup.")
            c.drawString(100, 610, "The methodology section describes the experimental procedures used in this study.")
            
            c.drawString(100, 570, "2. Methodology")
            c.drawString(100, 550, "We employed advanced statistical analysis techniques to process the data.")
            c.drawString(100, 530, "The experimental setup included multiple validation steps and quality controls.")
            
            c.drawString(100, 490, "3. Results")
            c.drawString(100, 470, "Our analysis revealed significant improvements in performance metrics.")
            c.drawString(100, 450, "The results demonstrate the effectiveness of the proposed approach.")
            
            c.drawString(100, 410, "4. Conclusion")
            c.drawString(100, 390, "This study provides valuable insights into the research domain.")
            c.drawString(100, 370, "Future work will focus on extending these findings to other applications.")
            
            c.save()
            print("✅ Sample PDF created")
            return True
        except ImportError:
            print("⚠️  reportlab not available, skipping sample PDF creation")
            return False
        except Exception as e:
            print(f"❌ Failed to create sample PDF: {e}")
            return False
    
    return True

def run_system():
    """Run the document intelligence system."""
    print("\n🚀 Running the Document Intelligence System...")
    
    try:
        from src.main import run_analysis
        from config import TEST_CASES
        
        # Use the first test case
        test_case_name = list(TEST_CASES.keys())[0]
        test_case = TEST_CASES[test_case_name]
        
        print(f"📋 Test Case: {test_case_name}")
        print(f"👤 Persona: {test_case['persona']}")
        print(f"🎯 Job: {test_case['job']}")
        
        # Get PDF files
        test_docs_dir = Path("test_docs")
        pdf_files = list(test_docs_dir.glob("*.pdf"))
        
        if not pdf_files:
            print("❌ No PDF files found in test_docs/")
            return False
        
        print(f"📄 Processing {len(pdf_files)} document(s)...")
        
        # Run analysis
        output = run_analysis(
            [str(f) for f in pdf_files],
            test_case["persona"],
            test_case["job"]
        )
        
        print("✅ Analysis completed successfully!")
        print(f"📊 Output saved to: challenge1b_output.json")
        
        # Show summary
        if "extracted_sections" in output:
            print(f"📑 Extracted {len(output['extracted_sections'])} relevant sections")
        
        if "sub_section_analysis" in output:
            print(f"📝 Analyzed {len(output['sub_section_analysis'])} subsections")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to run system: {e}")
        return False

def main():
    """Main quick start function."""
    print("🎯 Persona-Driven Document Intelligence - Quick Start")
    print("=" * 60)
    
    # Check dependencies
    missing_packages = check_dependencies()
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        install_choice = input("Install missing dependencies? (y/n): ").lower()
        
        if install_choice == 'y':
            if not install_dependencies():
                print("❌ Setup failed. Please install dependencies manually.")
                return
        else:
            print("❌ Setup cancelled.")
            return
    
    # Download models
    if not download_models():
        print("❌ Failed to download required models.")
        return
    
    # Create sample PDF if needed
    create_sample_pdf()
    
    # Run the system
    if run_system():
        print("\n🎉 Quick start completed successfully!")
        print("\n📚 Next steps:")
        print("1. Add your own PDF documents to the 'test_docs/' directory")
        print("2. Modify test cases in 'config.py' for different scenarios")
        print("3. Run 'python -m src.main' to process your documents")
        print("4. Run 'python benchmark.py' to test performance")
    else:
        print("\n❌ Quick start failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 