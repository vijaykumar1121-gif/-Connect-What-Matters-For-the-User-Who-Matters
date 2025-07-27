# src/pdf_parser.py

import fitz # PyMuPDF
from typing import List, Dict, Tuple, Optional
from src.data_models import Document, Section, Subsection

class PDFParser:
    def __init__(self):
        # Configuration for heading detection heuristics
        # Adjust these thresholds based on typical document types
        self.FONT_SIZE_THRESHOLD_H1 = 18.0 # Example: Large font for main headings
        self.FONT_SIZE_THRESHOLD_H2 = 14.0 # Example: Smaller for sub-headings
        self.FONT_SIZE_THRESHOLD_H3 = 12.0 # Example: Even smaller
        self.FONT_SIZE_THRESHOLD_BODY = 10.0 # Example: Normal body text size
        self.MAX_HEADING_LENGTH = 100 # Prevent very long lines from being headings
        self.HEADING_MIN_CHARS = 5 # Minimum characters to be considered a heading

    def _get_text_blocks_with_font_info(self, page) -> List[Dict]:
        """
        Extracts text blocks with basic font information (size, flags for bold/italic).
        PyMuPDF's get_text("dict") provides this.
        """
        blocks = page.get_text("dict")["blocks"]
        parsed_blocks = []
        for b in blocks:
            if b['type'] == 0:  # Text block
                for line in b['lines']:
                    line_text = ""
                    font_size = 0.0
                    font_flags = 0 # 1=superscript, 2=italic, 4=serif, 8=monospaced, 16=bold
                    
                    if line['spans']:
                        # Use info from the first span for simplicity, or average for more accuracy
                        span = line['spans'][0]
                        line_text = " ".join([s['text'] for s in line['spans']])
                        font_size = span['size']
                        font_flags = span['flags']
                        
                    if line_text.strip(): # Only process non-empty lines
                        parsed_blocks.append({
                            "text": line_text.strip(),
                            "size": font_size,
                            "is_bold": bool(font_flags & 16), # Check if bold flag is set
                            "page": page.number + 1 # 1-indexed page number
                        })
        return parsed_blocks

    def _determine_heading_level(self, block_info: Dict) -> int:
        """
        Determines the heading level based on font size and bold status.
        Returns 0 for body text, 1 for H1, 2 for H2, etc.
        """
        text = block_info['text']
        size = block_info['size']
        is_bold = block_info['is_bold']

        # Heuristics for heading detection
        if len(text) < self.HEADING_MIN_CHARS or len(text) > self.MAX_HEADING_LENGTH:
            return 0 # Too short or too long for a typical heading

        if is_bold:
            if size >= self.FONT_SIZE_THRESHOLD_H1:
                return 1
            elif size >= self.FONT_SIZE_THRESHOLD_H2:
                return 2
            elif size >= self.FONT_SIZE_THRESHOLD_H3:
                return 3
        
        # Fallback for non-bold but large text (e.g., some titles)
        if size > self.FONT_SIZE_THRESHOLD_H1:
             return 1
        elif size > self.FONT_SIZE_THRESHOLD_H2:
             return 2

        # Check for common numbering patterns for headings (e.g., "1. Introduction")
        if text and (text[0].isdigit() or text[0] in ['A', 'I']) and ('.' in text or ')' in text):
             if ' ' in text and (text.split(' ')[0].replace('.', '').isdigit() or text.split(' ')[0].isupper()):
                 # This is a very basic check, can be refined
                 if size >= self.FONT_SIZE_THRESHOLD_H3: # Even smaller headings often numbered
                     return 3 if not is_bold else (2 if size < self.FONT_SIZE_THRESHOLD_H2 else 1) # Prioritize bold
        
        return 0 # Default to body text

    def parse_pdf(self, file_path: str) -> Document:
        doc_obj = Document(file_name=file_path.split('/')[-1]) # Extract just the file name
        
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

                if heading_level > 0: # This is a potential heading
                    # A new section starts, or a higher-level section starts
                    if current_section is None or heading_level <= current_section_level:
                        # Close previous section if exists and add to document
                        if current_section:
                            doc_obj.add_section(current_section)
                        
                        # Start a new main section
                        current_section = Section(
                            title=text,
                            level=heading_level,
                            page_range=[page_actual, page_actual],
                            subsections=[]
                        )
                        current_section_level = heading_level
                    else:
                        # This is a sub-heading within the current section's scope
                        # We'll treat it as a new subsection for simplicity in this initial stage,
                        # but in a more advanced version, you'd nest Sections further.
                        if current_section:
                            current_section.add_subsection(
                                Subsection(text_content=text, page_number=page_actual)
                            )
                else: # Body text / subsection content
                    if current_section:
                        is_list_item = text.startswith(('-', '*', '•', '1.', 'a)')) # Basic list detection
                        current_section.add_subsection(
                            Subsection(text_content=text, page_number=page_actual, is_list_item=is_list_item)
                        )
                    else:
                        # If content appears before any heading, treat as part of an initial, unnamed section
                        if not doc_obj.sections: # Only create if no sections exist yet
                            current_section = Section(
                                title="Initial Content",
                                level=0, # Special level for initial content
                                page_range=[page_actual, page_actual],
                                subsections=[]
                            )
                            current_section_level = 0
                            doc_obj.add_section(current_section) # Add immediately
                        
                        # Add to the initial section
                        if current_section: # Ensure current_section is not None
                            is_list_item = text.startswith(('-', '*', '•', '1.', 'a)'))
                            current_section.add_subsection(
                                Subsection(text_content=text, page_number=page_actual, is_list_item=is_list_item)
                            )
        
        # Add the last section if it exists
        if current_section and current_section not in doc_obj.sections: # Avoid double adding if already added
             doc_obj.add_section(current_section)

        return doc_obj