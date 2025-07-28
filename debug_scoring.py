import os
import sys
from src.main import run_analysis

def test_scoring():
    print("🔍 Debugging Scoring System")
    print("=" * 50)
    
    # Test with a simple case
    test_docs = ['test_docs/chemistry_chapter1.pdf']
    persona = "PhD Researcher in Computational Biology"
    job = "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks"
    
    print(f"📄 Testing with: {test_docs[0]}")
    print(f"👤 Persona: {persona}")
    print(f"🎯 Job: {job}")
    
    try:
        result = run_analysis(test_docs, persona, job)
        
        print("\n📊 Results:")
        print("=" * 30)
        
        if 'extracted_sections' in result:
            print(f"Found {len(result['extracted_sections'])} sections")
            for i, section in enumerate(result['extracted_sections'][:5]):
                print(f"Rank {i+1}: {section['section_title']}")
                print(f"  Score: {section['importance_score']:.4f}")
                print(f"  Explanation: {section.get('explanation', 'N/A')}")
                print()
        else:
            print("❌ No extracted_sections found in result")
            
        if 'sub_section_analysis' in result:
            print(f"Found {len(result['sub_section_analysis'])} subsections")
            for i, sub in enumerate(result['sub_section_analysis'][:3]):
                print(f"Subsection {i+1}: Score {sub['importance_score']:.4f}")
        else:
            print("❌ No sub_section_analysis found in result")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_scoring() 