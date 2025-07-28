import fitz
from typing import List, Dict, Tuple, Optional
from src.data_models import Document, Section, Subsection

class PDFParser:
    def __init__(self):
        self.FONT_SIZE_THRESHOLD_H1 = 16.0
        self.FONT_SIZE_THRESHOLD_H2 = 14.0
        self.FONT_SIZE_THRESHOLD_H3 = 12.0
        self.FONT_SIZE_THRESHOLD_BODY = 10.0
        self.MAX_HEADING_LENGTH = 150
        self.HEADING_MIN_CHARS = 3

    def _get_text_blocks_with_font_info(self, page) -> List[Dict]:
        blocks = page.get_text("dict")["blocks"]
        parsed_blocks = []
        for b in blocks:
            if b['type'] == 0:
                for line in b['lines']:
                    line_text = ""
                    font_size = 0.0
                    font_flags = 0
                    if line['spans']:
                        span = line['spans'][0]
                        line_text = " ".join([s['text'] for s in line['spans']])
                        font_size = span['size']
                        font_flags = span['flags']
                    if line_text.strip():
                        parsed_blocks.append({
                            "text": line_text.strip(),
                            "size": font_size,
                            "is_bold": bool(font_flags & 16),
                            "page": page.number + 1
                        })
        return parsed_blocks

    def _determine_heading_level(self, block_info: Dict) -> int:
        text = block_info['text']
        size = block_info['size']
        is_bold = block_info['is_bold']
        
        # Skip very short or very long text
        if len(text) < self.HEADING_MIN_CHARS or len(text) > self.MAX_HEADING_LENGTH:
            return 0
        
        # Check for numbered headings (1., 2., 1.1, etc.)
        if text and text[0].isdigit():
            if '.' in text[:5] or ')' in text[:5]:
                if size >= self.FONT_SIZE_THRESHOLD_H2:
                    return 1
                elif size >= self.FONT_SIZE_THRESHOLD_H3:
                    return 2
        
        # Check for lettered headings (A., B., a), etc.)
        if text and text[0].isalpha() and len(text) > 1:
            if text[1] in ['.', ')']:
                if size >= self.FONT_SIZE_THRESHOLD_H2:
                    return 2
                elif size >= self.FONT_SIZE_THRESHOLD_H3:
                    return 3
        
        # Check for bold text with large font
        if is_bold:
            if size >= self.FONT_SIZE_THRESHOLD_H1:
                return 1
            elif size >= self.FONT_SIZE_THRESHOLD_H2:
                return 2
            elif size >= self.FONT_SIZE_THRESHOLD_H3:
                return 3
        
        # Check for large font without bold
        if size >= self.FONT_SIZE_THRESHOLD_H1:
            return 1
        elif size >= self.FONT_SIZE_THRESHOLD_H2:
            return 2
        elif size >= self.FONT_SIZE_THRESHOLD_H3:
            return 3
        
        # Check for common heading patterns
        text_lower = text.lower()
        heading_indicators = ['introduction', 'conclusion', 'method', 'methodology', 'results', 'discussion', 'abstract', 'summary', 'background', 'related work', 'experiments', 'evaluation', 'analysis']
        if any(indicator in text_lower for indicator in heading_indicators):
            if size >= self.FONT_SIZE_THRESHOLD_H2:
                return 1
            elif size >= self.FONT_SIZE_THRESHOLD_H3:
                return 2
        
        return 0

    def parse_pdf(self, file_path: str) -> Document:
        doc_obj = Document(file_name=file_path.split('/')[-1])
        
        try:
            doc = fitz.open(file_path)
        except Exception as e:
            print(f"Error opening PDF {file_path}: {e}")
            return doc_obj
        
        current_section: Optional[Section] = None
        current_section_level = 0
        all_text = ""
        
        for page_num in range(doc.page_count):
            try:
                page = doc[page_num]
                blocks = self._get_text_blocks_with_font_info(page)
                
                for block in blocks:
                    text = block['text']
                    page_actual = block['page']
                    all_text += text + " "
                    
                    heading_level = self._determine_heading_level(block)
                    
                    if heading_level > 0:
                        # Found a heading
                        if current_section is None or heading_level <= current_section_level:
                            if current_section:
                                doc_obj.add_section(current_section)
                            current_section = Section(
                                title=text,
                                level=heading_level,
                                page_range=[page_actual, page_actual],
                                subsections=[]
                            )
                            current_section_level = heading_level
                        else:
                            # This is a subsection heading
                            if current_section:
                                current_section.add_subsection(
                                    Subsection(text_content=text, page_number=page_actual)
                                )
                    else:
                        # Regular text content
                        if current_section:
                            is_list_item = text.startswith(('-', '*', '\u2022', '1.', 'a)', '•'))
                            current_section.add_subsection(
                                Subsection(text_content=text, page_number=page_actual, is_list_item=is_list_item)
                            )
                        else:
                            # No section yet, create initial content section
                            if not doc_obj.sections:
                                current_section = Section(
                                    title="Initial Content",
                                    level=0,
                                    page_range=[page_actual, page_actual],
                                    subsections=[]
                                )
                                current_section_level = 0
                                doc_obj.add_section(current_section)
                            
                            if current_section:
                                is_list_item = text.startswith(('-', '*', '\u2022', '1.', 'a)', '•'))
                                current_section.add_subsection(
                                    Subsection(text_content=text, page_number=page_actual, is_list_item=is_list_item)
                                )
            except Exception as e:
                print(f"Error processing page {page_num} in {file_path}: {e}")
                continue
        
        # Add the last section if it exists
        if current_section and current_section not in doc_obj.sections:
            doc_obj.add_section(current_section)
        
        # If no sections were found, create a default section with all text
        if not doc_obj.sections and all_text.strip():
            default_section = Section(
                title="Document Content",
                level=1,
                page_range=[1, doc.page_count],
                subsections=[]
            )
            # Split text into paragraphs and add as subsections
            paragraphs = [p.strip() for p in all_text.split('\n') if p.strip()]
            for i, para in enumerate(paragraphs[:10]):  # Limit to first 10 paragraphs
                default_section.add_subsection(
                    Subsection(text_content=para, page_number=1)
                )
            doc_obj.add_section(default_section)
        
        doc.close()
        return doc_obj