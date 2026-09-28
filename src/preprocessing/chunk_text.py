"""Phase 2 chunking strategies for the medical textbook corpus."""

import json
from pathlib import Path
from src.utils.io import save_jsonl

def _chunk_words(text, chunk_size=500, overlap=50):
    """Split text into deterministic word windows with explicit overlap."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller")

    words = text.split()
    chunks = []

    if not words:
        return chunks

    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap

    return chunks


def baseline_chunking(textbook_records, chunk_size=500, overlap=50):
    """
    Standard naive chunking over raw text, ignoring hierarchy.
    """
    print(f"\n--- Running Baseline Chunking ({chunk_size} words, {overlap} overlap) ---")

    all_chunks = []
    
    for record in textbook_records:
        book_id = record["document_id"]
        text = record["text"]
        
        chunks = _chunk_words(text, chunk_size, overlap)
            
        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{book_id}_chunk_{i}",
                "book_id": book_id,
                "text": chunk_text,
                "metadata": {
                    "strategy": "baseline",
                    "chunk_size_words": chunk_size,
                    "overlap_words": overlap,
                    "chunk_index": i,
                    "word_count": len(chunk_text.split()),
                    "source": "medical_textbook"
                }
            })
            
    print(f"  Generated {len(all_chunks):,} baseline chunks.")
    return all_chunks


def hierarchical_chunking(hierarchies, max_chunk_words=500):
    """
    Semantic chunking that respects section boundaries and embeds 
    hierarchy metadata directly into the chunks.
    """
    print(f"\n--- Running Hierarchical Chunking (Max {max_chunk_words} words) ---")
    
    all_chunks = []
    global_chunk_idx = 0
    
    for book in hierarchies:
        book_id = book["book_id"]
        
        for ch_idx, chapter in enumerate(book["chapters"]):
            chapter_title = chapter["title"]
            
            for sec_idx, section in enumerate(chapter["sections"]):
                section_title = section["title"]
                content = section.get("content", "")
                
                if not content:
                    continue
                    
                # Chunk within the section if it's too large
                words = content.split()
                
                if len(words) <= max_chunk_words:
                    sub_chunks = [content]
                else:
                    # Simple split within the section
                    sub_chunks = _chunk_words(content, max_chunk_words, overlap=50)
                    
                for i, chunk_text in enumerate(sub_chunks):
                    all_chunks.append({
                        "chunk_id": f"{book_id}_hier_{global_chunk_idx}",
                        "book_id": book_id,
                        "text": chunk_text,
                        "metadata": {
                            "strategy": "hierarchical",
                            "chapter": chapter_title,
                            "chapter_id": chapter.get("chapter_id"),
                            "section": section_title,
                            "section_id": section.get("section_id"),
                            "chapter_index": ch_idx,
                            "section_index": sec_idx,
                            "chunk_index": i,
                            "chunk_size_words": max_chunk_words,
                            "overlap_words": 50 if len(words) > max_chunk_words else 0,
                            "word_count": len(chunk_text.split()),
                            "source": "medical_textbook"
                        }
                    })
                    global_chunk_idx += 1
                    
    print(f"  Generated {len(all_chunks):,} hierarchical chunks.")
    return all_chunks


def run_chunking_pipeline(textbook_records, hierarchies, output_dir):
    """Run all chunking strategies and save them."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Keep the experiment matrix explicit so retrieval results can be compared fairly.
    configurations = [
        (250, 50, "baseline_chunks_250.jsonl"),
        (500, 50, "baseline_chunks_500.jsonl"),
        (500, 100, "baseline_chunks_500_overlap100.jsonl"),
        (1000, 100, "baseline_chunks_1000.jsonl"),
        (1500, 200, "baseline_chunks_1500.jsonl"),
    ]
    manifest = {"baseline": [], "hierarchical": {"chunk_size_words": 500, "overlap_words": 50}}

    for chunk_size, overlap, filename in configurations:
        chunks = baseline_chunking(textbook_records, chunk_size, overlap)
        save_jsonl(chunks, output_dir / filename)
        manifest["baseline"].append({
            "chunk_size_words": chunk_size,
            "overlap_words": overlap,
            "filename": filename,
            "chunk_count": len(chunks),
        })

    hier = hierarchical_chunking(hierarchies, max_chunk_words=500)
    save_jsonl(hier, output_dir / "hierarchical_chunks.jsonl")
    manifest["hierarchical"]["filename"] = "hierarchical_chunks.jsonl"
    manifest["hierarchical"]["chunk_count"] = len(hier)

    with open(output_dir / "chunk_experiment_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
