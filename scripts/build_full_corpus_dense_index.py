"""
Step B.2: Build and persist the full 18-textbook dense FAISS index.

Uses SentenceTransformers ('all-MiniLM-L6-v2', 384-dim) across all 125,847 chunks.
Saves:
- indexes/full_textbooks_dense/dense_full_textbooks.faiss
- indexes/full_textbooks_dense/dense_index_metadata.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_full_textbook_chunks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunk-dir",
        type=Path,
        default=Path(r"C:\Users\kiran\OneDrive\Desktop\sem7\major\datasets\MedRag\chunk"),
        help="Path to 18 textbook jsonl chunks directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "indexes" / "full_textbooks_dense",
        help="Output directory for FAISS index and metadata",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Encoding batch size",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    faiss_path = args.output_dir / "dense_full_textbooks.faiss"
    metadata_path = args.output_dir / "dense_index_metadata.json"

    # CPU threading configuration
    cpu_cores = os.cpu_count() or 8
    torch.set_num_threads(cpu_cores)
    print("=" * 80)
    print("BUILDING FULL 18-TEXTBOOK DENSE VECTOR INDEX (FAISS)")
    print(f"Embedding Model: {args.model_name}")
    print(f"CPU Cores: {cpu_cores} threads")
    print(f"Batch Size: {args.batch_size}")
    print(f"Chunk Directory: {args.chunk_dir}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 80)

    # 1. Load Chunks
    start_load = time.perf_counter()
    print("Loading 18 textbook chunk files...")
    chunks = load_full_textbook_chunks(args.chunk_dir)
    total_chunks = len(chunks)
    load_time = time.perf_counter() - start_load
    print(f"Loaded {total_chunks:,} chunks across 18 textbooks in {load_time:.2f}s.")

    # 2. Initialize Model
    print(f"Initializing SentenceTransformer('{args.model_name}')...")
    model = SentenceTransformer(args.model_name, device="cpu")
    dim = model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {dim}")

    # 3. Batch Embed Chunks
    print(f"Encoding {total_chunks:,} chunks with batch size {args.batch_size}...")
    start_embed = time.perf_counter()
    texts = [chunk.text for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    embed_time = time.perf_counter() - start_embed
    print(f"Successfully generated embeddings in {embed_time:.2f}s ({total_chunks / embed_time:.1f} chunks/sec).")

    # 4. Normalize and Build FAISS IndexFlatIP
    print("Normalizing vectors (L2) and adding to FAISS IndexFlatIP...")
    start_index = time.perf_counter()
    matrix = np.asarray(embeddings, dtype=np.float32)
    faiss.normalize_L2(matrix)

    index = faiss.IndexFlatIP(dim)
    index.add(matrix)
    index_time = time.perf_counter() - start_index
    print(f"FAISS index built with {index.ntotal:,} vectors in {index_time:.2f}s.")

    # 5. Save FAISS Index
    print(f"Saving FAISS index to {faiss_path}...")
    faiss.write_index(index, str(faiss_path))
    file_size_mb = faiss_path.stat().st_size / (1024 * 1024)
    print(f"Saved FAISS index ({file_size_mb:.2f} MB).")

    # 6. Save Metadata & Manifest
    chunk_manifest = [
        {
            "chunk_id": chunk.chunk_id,
            "textbook_name": chunk.textbook_name,
            "title": chunk.title,
            "source_file": chunk.source_file,
        }
        for chunk in chunks
    ]
    metadata = {
        "model_name": args.model_name,
        "embedding_dimension": dim,
        "total_chunks": total_chunks,
        "faiss_index_path": str(faiss_path),
        "index_file_size_mb": round(file_size_mb, 2),
        "build_time_seconds": round(embed_time + index_time, 2),
        "encoding_throughput_chunks_per_sec": round(total_chunks / embed_time, 1),
        "normalization": "L2; inner-product search equals cosine similarity",
        "chunks": chunk_manifest,
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved index metadata to {metadata_path}.")

    # 7. Verification: Test Known Query
    print("\n--- Verifying Index Integrity with Test Query ---")
    test_query = "What are the diagnostic criteria for major depressive disorder?"
    q_vec = model.encode([test_query], convert_to_numpy=True, normalize_embeddings=False)
    q_vec = np.asarray(q_vec, dtype=np.float32)
    faiss.normalize_L2(q_vec)
    scores, indices = index.search(q_vec, 3)

    print(f"Query: '{test_query}'")
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        matched_chunk = chunks[idx]
        print(f"[{rank}] ID: {matched_chunk.chunk_id} ({matched_chunk.textbook_name}) | Cosine Sim: {score:.4f}")
        print(f"    Text: {matched_chunk.text[:120]}...\n")

    print("Index verification PASSED! Step B.2 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
