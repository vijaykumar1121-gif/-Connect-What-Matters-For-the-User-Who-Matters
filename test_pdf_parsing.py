import os
from src.pdf_parser import PDFParser

def test_pdf_parsing():
    print("🔍 Testing PDF Parsing")
    print("=" * 30)
    
    pdf_path = "test_docs/research_paper_sample.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        return
    
    try:
        parser = PDFParser()
        doc = parser.parse_pdf(pdf_path)
        
        print(f"📄 Parsed: {pdf_path}")
        print(f"📊 Found {len(doc.sections)} sections")
        
        for i, section in enumerate(doc.sections):
            print(f"\nSection {i+1}: {section.title}")
            print(f"  Level: {section.level}")
            print(f"  Pages: {section.page_range}")
            print(f"  Subsections: {len(section.subsections)}")
            
            for j, sub in enumerate(section.subsections[:3]):
                print(f"    Subsection {j+1}: {sub.text_content[:100]}...")
                
    except Exception as e:
        print(f"❌ Error parsing PDF: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pdf_parsing() 