# src/data_models.py

from typing import List, Dict, Any, Optional

class Subsection:
    """Represents a granular part of a document, like a paragraph or list."""
    def __init__(self,
                 text_content: str,
                 page_number: int,
                 is_list_item: bool = False,
                 is_table_candidate: bool = False,
                 importance_score: float = 0.0): # Will be calculated later
        self.text_content = text_content.strip()
        self.page_number = page_number
        self.is_list_item = is_list_item
        self.is_table_candidate = is_table_candidate
        self.importance_score = importance_score
        self.refined_text: str = "" # To be filled during final refinement

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text_content": self.text_content,
            "page_number": self.page_number,
            "is_list_item": self.is_list_item,
            "is_table_candidate": self.is_table_candidate,
            "importance_score": self.importance_score,
            "refined_text": self.refined_text
        }

class Section:
    """Represents a logical section of a document (e.g., Introduction, Methodology)."""
    def __init__(self,
                 title: str,
                 level: int, # H1=1, H2=2, etc. based on font size/pattern
                 page_range: List[int], # [start_page, end_page]
                 subsections: Optional[List[Subsection]] = None,
                 text_content: str = "", # Aggregated text for the whole section
                 importance_score: float = 0.0): # Will be calculated later
        self.title = title.strip()
        self.level = level
        self.page_range = page_range
        self.subsections = subsections if subsections is not None else []
        self.text_content = text_content.strip()
        self.importance_score = importance_score

    def add_subsection(self, subsection: Subsection):
        self.subsections.append(subsection)
        # Update page range
        self.page_range[0] = min(self.page_range[0], subsection.page_number)
        self.page_range[1] = max(self.page_range[1], subsection.page_number)
        self.text_content += ("\n" + subsection.text_content)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "level": self.level,
            "page_range": self.page_range,
            "text_content": self.text_content,
            "importance_score": self.importance_score,
            "subsections": [ss.to_dict() for ss in self.subsections]
        }

class Document:
    """Represents a processed PDF document."""
    def __init__(self,
                 file_name: str,
                 sections: Optional[List[Section]] = None):
        self.file_name = file_name
        self.sections = sections if sections is not None else []

    def add_section(self, section: Section):
        self.sections.append(section)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_name": self.file_name,
            "sections": [s.to_dict() for s in self.sections]
        }