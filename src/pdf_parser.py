import fitz
from typing import List, Dict, Tuple, Optional
from src.data_models import Document, Section, Subsection
class PDFParser:
    def __init__(self):
        self.FONT_SIZE_THRESHOLD_H1 = 18.0
        self.FONT_SIZE_THRESHOLD_H2 = 14.0
        self.FONT_SIZE_THRESHOLD_H3 = 12.0
        self.FONT_SIZE_THRESHOLD_BODY = 10.0
        self.MAX_HEADING_LENGTH = 100
        self.HEADING_MIN_CHARS = 5
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
        if len(text) < self.HEADING_MIN_CHARS or len(text) > self.MAX_HEADING_LENGTH:
            return 0
        if is_bold:
            if size >= self.FONT_SIZE_THRESHOLD_H1:
                return 1
            elif size >= self.FONT_SIZE_THRESHOLD_H2:
                return 2
            elif size >= self.FONT_SIZE_THRESHOLD_H3:
                return 3
        if size > self.FONT_SIZE_THRESHOLD_H1:
             return 1
        elif size > self.FONT_SIZE_THRESHOLD_H2:
             return 2
        if text and (text[0].isdigit() or text[0] in ['A', 'I']) and ('.' in text or ')' in text):
             if ' ' in text and (text.split(' ')[0].replace('.', '').isdigit() or text.split(' ')[0].isupper()):
                 if size >= self.FONT_SIZE_THRESHOLD_H3:
                     return 3 if not is_bold else (2 if size < self.FONT_SIZE_THRESHOLD_H2 else 1)
        return 0
    def parse_pdf(self, file_path: str) -> Document:
        doc_obj = Document(file_name=file_path.split('/')[-1])
        doc = fitz.open(file_path)
        current_section: Optional[Section] = None
        current_section_level = 0
        for page_num in range(doc.page_count):
            page = doc[page_num]
            blocks = self._get_text_blocks_with_font_info(page)
            for block in blocks:
                text = block['text']
                page_actual = block['page']
                heading_level = self._determine_heading_level(block)
                if heading_level > 0:
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
                        if current_section:
                            current_section.add_subsection(
                                Subsection(text_content=text, page_number=page_actual)
                            )
                else:
                    if current_section:
                        is_list_item = text.startswith(('-', '*', '\u2022', '1.', 'a)'))
                        current_section.add_subsection(
                            Subsection(text_content=text, page_number=page_actual, is_list_item=is_list_item)
                        )
                    else:
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
                            is_list_item = text.startswith(('-', '*', '\u2022', '1.', 'a)'))
                            current_section.add_subsection(
                                Subsection(text_content=text, page_number=page_actual, is_list_item=is_list_item)
                            )
        if current_section and current_section not in doc_obj.sections:
             doc_obj.add_section(current_section)
        return doc_obj