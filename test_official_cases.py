#!/usr/bin/env python3
"""
Official Test Cases Script
Test the system with the three official test cases from the challenge requirements.
"""

import os
import json
from pathlib import Path
from src.main import run_analysis

# Official test cases from challenge requirements
OFFICIAL_TEST_CASES = {
    "academic_research": {
        "name": "Academic Research - Graph Neural Networks for Drug Discovery",
        "documents": "4 research papers on 'Graph Neural Networks for Drug Discovery'",
        "persona": "PhD Researcher in Computational Biology",
        "job": "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks"
    },
    "business_analysis": {
        "name": "Business Analysis - Tech Company Annual Reports",
        "documents": "3 annual reports from competing tech companies (2022-2024)",
        "persona": "Investment Analyst", 
        "job": "Analyze revenue trends, R&D investments, and market positioning strategies"
    },
    "educational_content": {
        "name": "Educational Content - Organic Chemistry Textbooks",
        "documents": "5 chapters from organic chemistry textbooks",
        "persona": "Undergraduate Chemistry Student",
        "job": "Identify key concepts and mechanisms for exam preparation on reaction kinetics"
    }
}

def test_official_case(case_name, case_data, documents_dir="test_docs"):
    """
    Test an official test case.
    
    Args:
        case_name (str): Name of the test case
        case_data (dict): Test case data
        documents_dir (str): Directory containing PDF documents
    """
    print(f"\n🎯 Testing Official Case: {case_data['name']}")
    print("=" * 70)
    print(f"📄 Documents: {case_data['documents']}")
    print(f"👤 Persona: {case_data['persona']}")
    print(f"🎯 Job: {case_data['job']}")
    
    # Get PDF files
    docs_path = Path(documents_dir)
    if not docs_path.exists():
        print(f"❌ Directory '{documents_dir}' not found!")
        return None
    
    pdf_files = list(docs_path.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in '{documents_dir}'!")
        print("💡 Add PDF documents to test_docs/ directory to test this case")
        return None
    
    print(f"📄 Found {len(pdf_files)} PDF document(s)")
    
    # Run analysis
    try:
        output = run_analysis(
            [str(f) for f in pdf_files],
            case_data['persona'],
            case_data['job']
        )
        
        # Save output with case name
        output_filename = f"official_case_{case_name}_output.json"
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)
        
        print(f"✅ Analysis completed!")
        print(f"📊 Results saved to: {output_filename}")
        
        # Validate output format
        validate_output_format(output, case_name)
        
        # Show summary
        show_case_summary(output, case_name)
        
        return output
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return None

def validate_output_format(output, case_name):
    """Validate that output meets challenge requirements."""
    print(f"\n🔍 Validating Output Format for {case_name}")
    print("-" * 50)
    
    required_fields = {
        "metadata": ["input_documents", "persona", "job_to_be_done", "processing_timestamp"],
        "extracted_sections": ["document", "page_number", "section_title", "importance_rank"],
        "sub_section_analysis": ["document", "page_number", "refined_text"]
    }
    
    all_valid = True
    
    for section, fields in required_fields.items():
        if section not in output:
            print(f"❌ Missing required section: {section}")
            all_valid = False
            continue
            
        print(f"✅ {section} section present")
        
        if section == "metadata":
            for field in fields:
                if field in output[section]:
                    print(f"  ✅ {field}: {output[section][field]}")
                else:
                    print(f"  ❌ Missing field: {field}")
                    all_valid = False
        else:
            # Check if sections have required fields
            if output[section] and len(output[section]) > 0:
                first_item = output[section][0]
                for field in fields:
                    if field in first_item:
                        print(f"  ✅ {field} field present")
                    else:
                        print(f"  ❌ Missing field: {field}")
                        all_valid = False
            else:
                print(f"  ⚠️  {section} is empty")
    
    if all_valid:
        print(f"🎉 Output format validation PASSED for {case_name}")
    else:
        print(f"❌ Output format validation FAILED for {case_name}")
    
    return all_valid

def show_case_summary(output, case_name):
    """Show a summary of the analysis results."""
    print(f"\n📊 Analysis Summary for {case_name}")
    print("-" * 50)
    
    if "extracted_sections" in output:
        sections = output["extracted_sections"]
        print(f"📑 Extracted {len(sections)} relevant sections")
        
        if sections:
            print("\n🏆 Top 3 Most Relevant Sections:")
            for i, section in enumerate(sections[:3], 1):
                print(f"{i}. {section.get('section_title', 'Untitled')}")
                print(f"   Document: {section.get('document', 'Unknown')}")
                print(f"   Page: {section.get('page_number', 'N/A')}")
                print(f"   Rank: {section.get('importance_rank', 'N/A')}")
                print(f"   Score: {section.get('enhanced_model_score', 0):.3f}")
                print(f"   Explanation: {section.get('enhanced_explanation', 'N/A')[:80]}...")
                print()
    
    if "sub_section_analysis" in output:
        subsections = output["sub_section_analysis"]
        print(f"📝 Analyzed {len(subsections)} subsections")
        
        if subsections:
            print("\n🔍 Sample Subsection Analysis:")
            sample = subsections[0]
            print(f"   Document: {sample.get('document', 'Unknown')}")
            print(f"   Page: {sample.get('page_number', 'N/A')}")
            print(f"   Text: {sample.get('refined_text', 'N/A')[:100]}...")
    
    # Show advanced features
    advanced_features = []
    if "entity_summary" in output:
        advanced_features.append("Entity Aggregation")
    if "topic_summary" in output:
        advanced_features.append("Topic Modeling")
    if "qa_example" in output:
        advanced_features.append("AI Question Answering")
    
    if advanced_features:
        print(f"\n🚀 Advanced Features Used: {', '.join(advanced_features)}")

def run_all_official_tests():
    """Run all three official test cases."""
    print("🎯 Running All Official Test Cases")
    print("=" * 70)
    print("This will test the three official cases from the challenge requirements:")
    print("1. Academic Research - Graph Neural Networks for Drug Discovery")
    print("2. Business Analysis - Tech Company Annual Reports") 
    print("3. Educational Content - Organic Chemistry Textbooks")
    print("\n💡 Make sure you have PDF documents in the test_docs/ directory")
    
    results = {}
    
    for case_name, case_data in OFFICIAL_TEST_CASES.items():
        result = test_official_case(case_name, case_data)
        results[case_name] = result
    
    # Summary
    print(f"\n🎉 Official Test Cases Summary")
    print("=" * 70)
    successful_cases = sum(1 for r in results.values() if r is not None)
    total_cases = len(OFFICIAL_TEST_CASES)
    
    print(f"✅ Successful: {successful_cases}/{total_cases}")
    
    for case_name, result in results.items():
        status = "✅ PASSED" if result is not None else "❌ FAILED"
        print(f"   {case_name}: {status}")
    
    if successful_cases == total_cases:
        print(f"\n🎊 All official test cases PASSED!")
        print("Your system is ready for the challenge submission!")
    else:
        print(f"\n⚠️  Some test cases failed. Check the error messages above.")

def main():
    """Main function."""
    import sys
    
    if len(sys.argv) > 1:
        case_name = sys.argv[1]
        if case_name in OFFICIAL_TEST_CASES:
            test_official_case(case_name, OFFICIAL_TEST_CASES[case_name])
        else:
            print(f"Unknown test case: {case_name}")
            print(f"Available cases: {list(OFFICIAL_TEST_CASES.keys())}")
    else:
        run_all_official_tests()

if __name__ == "__main__":
    main() 