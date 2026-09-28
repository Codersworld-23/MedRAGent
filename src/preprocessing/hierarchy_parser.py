"""
Phase 2 -- Hierarchical Textbook Parser

Heuristically extracts chapters and sections from raw
medical textbook text to create a hierarchical document tree.
"""

import re
from pathlib import Path
from src.utils.io import save_json


def _is_chapter_heading(line):
    """Detect if a line is likely a chapter heading."""
    
    line = line.strip()
    
    # E.g. "CHAPTER 1: Introduction", "Chapter 1", "PART I"
    chapter_pattern = r"^(?:chapter|part)\s+[0-9ivx]+[\.:]?\s*(.*)$"
    
    if re.match(chapter_pattern, line, re.IGNORECASE):
        return True
        
    return False


def _chapter_number(line):
    match = re.match(r"^(?:chapter|part)\s+([0-9ivx]+)", line.strip(), re.IGNORECASE)
    return match.group(1) if match else None


def _is_section_heading(line):
    """Detect if a line is likely a section heading."""
    
    line = line.strip()
    
    # E.g. "1.1 General Principles", "1.1.2 Sub-topic"
    section_pattern = r"^[0-9]+\.[0-9]+(?:\.[0-9]+)?[\.:]?\s+(.*)$"
    
    if re.match(section_pattern, line):
        return True
        
    return False


def _section_number(line):
    match = re.match(r"^([0-9]+\.[0-9]+(?:\.[0-9]+)?)", line.strip())
    return match.group(1) if match else None


def parse_hierarchy(text, book_id):
    """
    Parse a raw textbook string into a hierarchical JSON structure.
    
    Returns:
        dict: {
            "book_id": str,
            "chapters": [
                {
                    "title": str,
                    "sections": [
                        {
                            "title": str,
                            "content": str
                        }
                    ]
                }
            ]
        }
    """
    
    lines = text.split("\n")
    
    hierarchy = {
        "book_id": book_id,
        "schema_version": 1,
        "source": "medical_textbook",
        "chapters": []
    }
    
    # Initialize with a default chapter if content starts immediately
    current_chapter = {
        "title": "Front Matter",
        "chapter_id": f"{book_id}_front_matter",
        "number": None,
        "level": "chapter",
        "sections": []
    }
    
    current_section = {
        "title": "Introduction",
        "section_id": f"{book_id}_front_matter_introduction",
        "number": None,
        "level": "section",
        "content_lines": []
    }
    
    for line in lines:
        if not line.strip():
            # Add blank lines back to content if we are collecting
            current_section["content_lines"].append(line)
            continue
            
        if _is_chapter_heading(line):
            # Save the previous section
            if current_section["content_lines"] or current_section["title"] != "Introduction":
                current_section["content"] = "\n".join(current_section["content_lines"]).strip()
                del current_section["content_lines"]
                current_chapter["sections"].append(current_section)
                
            # Save the previous chapter
            if current_chapter["sections"]:
                hierarchy["chapters"].append(current_chapter)
                
            # Start new chapter
            chapter_number = _chapter_number(line)
            chapter_index = len(hierarchy["chapters"])
            chapter_id = f"{book_id}_chapter_{chapter_index}"
            current_chapter = {
                "title": line.strip(),
                "chapter_id": chapter_id,
                "number": chapter_number,
                "level": "chapter",
                "sections": []
            }
            # Start new section for this chapter
            current_section = {
                "title": "Chapter Introduction",
                "section_id": f"{chapter_id}_introduction",
                "number": None,
                "level": "section",
                "content_lines": []
            }
            
        elif _is_section_heading(line):
            # Save previous section
            if current_section["content_lines"] or current_section["title"] != "Chapter Introduction":
                current_section["content"] = "\n".join(current_section["content_lines"]).strip()
                del current_section["content_lines"]
                current_chapter["sections"].append(current_section)
                
            # Start new section
            section_number = _section_number(line)
            section_index = len(current_chapter["sections"])
            current_section = {
                "title": line.strip(),
                "section_id": f"{current_chapter['chapter_id']}_section_{section_index}",
                "number": section_number,
                "level": "section",
                "content_lines": []
            }
            
        else:
            current_section["content_lines"].append(line)
            
    # Save the final section and chapter
    if current_section["content_lines"] or current_section["title"] != "Chapter Introduction":
        current_section["content"] = "\n".join(current_section["content_lines"]).strip()
        del current_section["content_lines"]
        current_chapter["sections"].append(current_section)
        
    if current_chapter["sections"]:
        hierarchy["chapters"].append(current_chapter)
        
    return hierarchy


def parse_all_textbooks(textbook_records, output_dir):
    """
    Process a list of textbook records into hierarchies.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_hierarchies = []
    
    print("\n--- Parsing Textbook Hierarchies ---")
    
    for record in textbook_records:
        book_id = record["document_id"]
        text = record["text"]
        
        print(f"  Parsing: {book_id}...")
        
        hierarchy = parse_hierarchy(text, book_id)
        all_hierarchies.append(hierarchy)
        
        # Save individual hierarchy
        save_json(hierarchy, output_dir / f"{book_id}_hierarchy.json")
        
        # Stats
        num_chapters = len(hierarchy["chapters"])
        num_sections = sum(len(c["sections"]) for c in hierarchy["chapters"])
        
        print(f"    Found {num_chapters} chapters, {num_sections} sections")
        
    return all_hierarchies
