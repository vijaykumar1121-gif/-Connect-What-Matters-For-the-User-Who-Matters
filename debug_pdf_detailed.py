import os
import fitz
from src.pdf_parser import PDFParser

def debug_pdf_detailed():
    print("🔍 Detailed PDF Debugging")
    print("=" * 40)
    
    pdf_path = "test_docs/chemistry_chapter1.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        return
    
    try:
        # First, let's see what's in the PDF
        doc = fitz.open(pdf_path)
        print(f"📄 PDF: {pdf_path}")
        print(f"📊 Pages: {doc.page_count}")
        
        # Check first page
        page = doc[0]
        print(f"\n📄 Page 1 content:")
        print("=" * 30)
        
        # Get raw text
        raw_text = page.get_text()
        print(f"Raw text length: {len(raw_text)}")
        print(f"First 200 chars: {raw_text[:200]}")
        
        # Get structured text
        blocks = page.get_text("dict")["blocks"]
        print(f"\n📊 Found {len(blocks)} blocks")
        
        text_blocks = 0
        for i, block in enumerate(blocks):
            if block['type'] == 0:  # text block
                text_blocks += 1
                print(f"\nBlock {i+1} (text):")
                for line in block['lines']:
                    for span in line['spans']:
                        text = span['text']
                        size = span['size']
                        flags = span['flags']
                        is_bold = bool(flags & 16)
                        print(f"  Text: '{text}' | Size: {size} | Bold: {is_bold}")
        
        print(f"\n📊 Total text blocks: {text_blocks}")
        
        # Now test our parser
        print(f"\n🔧 Testing our parser:")
        parser = PDFParser()
        parsed_doc = parser.parse_pdf(pdf_path)
        
        print(f"📊 Parsed sections: {len(parsed_doc.sections)}")
        
        for i, section in enumerate(parsed_doc.sections):
            print(f"\nSection {i+1}: {section.title}")
            print(f"  Level: {section.level}")
            print(f"  Pages: {section.page_range}")
            print(f"  Subsections: {len(section.subsections)}")
            
            for j, sub in enumerate(section.subsections[:2]):
                print(f"    Sub {j+1}: {sub.text_content[:80]}...")
        
        doc.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_pdf_detailed() 