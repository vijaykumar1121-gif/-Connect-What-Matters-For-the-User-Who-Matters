#!/usr/bin/env python3
"""
Custom Dataset Testing Script
Test the system with your own documents, persona, and job-to-be-done.
"""

import os
import json
from pathlib import Path
from src.main import run_analysis

def test_custom_dataset(persona, job, documents_dir="test_docs"):
    """
    Test the system with custom persona, job, and documents.
    
    Args:
        persona (str): Your persona description
        job (str): Your job-to-be-done
        documents_dir (str): Directory containing PDF documents
    """
    print("🎯 Testing Custom Dataset")
    print("=" * 50)
    print(f"👤 Persona: {persona}")
    print(f"🎯 Job: {job}")
    print(f"📁 Documents: {documents_dir}")
    
    # Get PDF files
    docs_path = Path(documents_dir)
    if not docs_path.exists():
        print(f"❌ Directory '{documents_dir}' not found!")
        return
    
    pdf_files = list(docs_path.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in '{documents_dir}'!")
        return
    
    print(f"📄 Found {len(pdf_files)} PDF document(s)")
    
    # Run analysis
    try:
        output = run_analysis(
            [str(f) for f in pdf_files],
            persona,
            job
        )
        
        # Save custom output
        output_filename = f"custom_output_{persona.replace(' ', '_').lower()}.json"
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)
        
        print(f"✅ Analysis completed!")
        print(f"📊 Results saved to: {output_filename}")
        
        # Show summary
        if "extracted_sections" in output:
            print(f"📑 Extracted {len(output['extracted_sections'])} relevant sections")
            
            # Show top 3 sections
            print("\n🏆 Top 3 Most Relevant Sections:")
            for i, section in enumerate(output['extracted_sections'][:3], 1):
                print(f"{i}. {section.get('section_title', 'Untitled')}")
                print(f"   Document: {section.get('document', 'Unknown')}")
                print(f"   Score: {section.get('enhanced_model_score', 0):.3f}")
                print(f"   Explanation: {section.get('enhanced_explanation', 'N/A')[:100]}...")
                print()
        
        if "sub_section_analysis" in output:
            print(f"📝 Analyzed {len(output['sub_section_analysis'])} subsections")
        
        return output
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return None

def interactive_test():
    """Interactive testing with user input."""
    print("🎯 Interactive Custom Dataset Testing")
    print("=" * 50)
    
    # Get persona
    print("\n👤 Enter your persona (e.g., 'Medical Researcher', 'Financial Analyst'):")
    persona = input("Persona: ").strip()
    
    # Get job
    print("\n🎯 Enter your job-to-be-done (e.g., 'Extract clinical trial results'):")
    job = input("Job: ").strip()
    
    # Get documents directory
    print("\n📁 Enter documents directory (default: 'test_docs'):")
    docs_dir = input("Directory: ").strip() or "test_docs"
    
    # Run test
    return test_custom_dataset(persona, job, docs_dir)

def example_tests():
    """Run example tests with different domains."""
    examples = [
        {
            "name": "Medical Research",
            "persona": "Medical Researcher",
            "job": "Extract clinical trial methodologies and patient outcomes"
        },
        {
            "name": "Financial Analysis", 
            "persona": "Investment Analyst",
            "job": "Analyze revenue trends and market positioning strategies"
        },
        {
            "name": "Legal Review",
            "persona": "Legal Consultant", 
            "job": "Identify key legal precedents and regulatory requirements"
        },
        {
            "name": "Technical Analysis",
            "persona": "Software Engineer",
            "job": "Extract implementation details and API specifications"
        }
    ]
    
    print("🧪 Running Example Tests")
    print("=" * 50)
    
    for example in examples:
        print(f"\n📋 Testing: {example['name']}")
        test_custom_dataset(example['persona'], example['job'])

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "interactive":
            interactive_test()
        elif sys.argv[1] == "examples":
            example_tests()
        else:
            print("Usage:")
            print("  python test_custom_dataset.py interactive  # Interactive mode")
            print("  python test_custom_dataset.py examples     # Run example tests")
    else:
        print("🎯 Custom Dataset Testing")
        print("=" * 50)
        print("Choose an option:")
        print("1. Interactive testing (enter your own persona/job)")
        print("2. Example tests (predefined scenarios)")
        
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == "1":
            interactive_test()
        elif choice == "2":
            example_tests()
        else:
            print("Invalid choice. Running interactive mode...")
            interactive_test() 