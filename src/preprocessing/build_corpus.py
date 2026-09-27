from pathlib import Path

from src.preprocessing.clean_text import (
    clean_text
)

from src.utils.io import (
    save_jsonl,
    ensure_dir
)


def build_textbook_corpus(
    textbook_dir,
    output_file
):

    textbook_dir = Path(
        textbook_dir
    )

    documents = []

    files = sorted(
        textbook_dir.glob("*.txt")
    )

    print(
        f"Found {len(files)} English textbooks."
    )

    for book_path in files:

        print(
            f"Processing: {book_path.name}"
        )

        with open(
            book_path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:

            raw_text = f.read()

        text = clean_text(
            raw_text
        )

        if not text:

            print(
                f"WARNING: empty file "
                f"{book_path.name}"
            )

            continue

        documents.append({

            "document_id":
                book_path.stem,

            "filename":
                book_path.name,

            "language":
                "en",

            "source":
                "medical_textbook",

            "text":
                text,

            "character_count":
                len(text),

            "word_count":
                len(text.split())
        })

    ensure_dir(
        Path(output_file).parent
    )

    save_jsonl(
        documents,
        output_file
    )

    print(
        f"\nSaved {len(documents)} textbooks."
    )

    return documents
