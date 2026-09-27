"""
Phase 1 -- Normalize Questions

Contains per-dataset normalizer functions that convert
each dataset's native record format into a common schema:

    {
        "id":       str,      # unique identifier
        "dataset":  str,      # source dataset name
        "split":    str,      # train / dev / test / validation
        "subset":   str|None, # subject or subset name
        "question": str,      # the question text
        "options":  dict|None,# {"A": "...", "B": "...", ...}
        "answer":   str|None, # answer key (letter) or text
        "context":  any|None, # explanation / snippets / context
    }
"""


def normalize_medqa(record, index, split, subset="US"):
    """
    Normalize a MedQA record.

    MedQA native format:
        question, answer (text), options (dict), answer_idx (letter),
        meta_info
    """

    options = record.get("options", {})

    # answer_idx is the letter key (e.g. "E")
    answer_idx = record.get("answer_idx")

    # answer is the full answer text -- we prefer the letter
    answer = answer_idx

    if not answer:
        # Fallback: try to find the letter from the text
        answer_text = record.get("answer", "")

        for key, val in options.items():
            if val == answer_text:
                answer = key
                break
        else:
            answer = answer_text

    return {
        "id": (
            record.get("id")
            or f"medqa_{subset}_{split}_{index}"
        ),
        "dataset": "MedQA",
        "split": split,
        "subset": subset,
        "question": record.get("question", ""),
        "options": options,
        "answer": answer,
        "context": record.get("meta_info"),
    }


def normalize_mmlu(record, index, split, subject):
    """
    Normalize an MMLU-Medical record.

    MMLU native format:
        question, choices (list), answer (int index), subject
    """

    choices = record.get("choices", [])

    options = {}

    for i, choice in enumerate(choices):

        letter = chr(ord("A") + i)

        options[letter] = choice

    answer = record.get("answer")

    if isinstance(answer, int):

        answer = chr(ord("A") + answer)

    return {
        "id": f"mmlu_{subject}_{split}_{index}",
        "dataset": "MMLU-Medical",
        "split": split,
        "subset": subject,
        "question": record.get("question", ""),
        "options": options,
        "answer": answer,
        "context": None,
    }


def normalize_medmcqa(record, index, split, subject=None):
    """
    Normalize a MedMCQA record.

    MedMCQA native format:
        question, opa/opb/opc/opd, cop (1-indexed int or None),
        subject_name, topic_name, id, choice_type, exp
    """

    options = {
        "A": record.get("opa"),
        "B": record.get("opb"),
        "C": record.get("opc"),
        "D": record.get("opd"),
    }

    cop = record.get("cop")

    answer = None

    if cop is not None:

        # MedMCQA uses 1-indexed: 1=A, 2=B, 3=C, 4=D
        answer_map = {1: "A", 2: "B", 3: "C", 4: "D"}

        try:
            answer = answer_map.get(int(cop))
        except (ValueError, TypeError):
            answer = str(cop)

    return {
        "id": (
            record.get("id")
            or f"medmcqa_{split}_{index}"
        ),
        "dataset": "MedMCQA",
        "split": split,
        "subset": (
            subject
            or record.get("subject_name")
        ),
        "question": record.get("question", ""),
        "options": options,
        "answer": answer,
        "context": record.get("exp"),
    }


def normalize_pubmedqa(record, index, split):
    """
    Normalize a PubMedQA record.

    PubMedQA native format:
        QUESTION, CONTEXTS, LONG_ANSWER, final_decision,
        MESHES, YEAR, LABELS (for labeled subset)
    """

    question = (
        record.get("QUESTION")
        or record.get("question")
        or ""
    )

    # Build context from CONTEXTS list
    contexts = record.get("CONTEXTS")

    if isinstance(contexts, list):
        context_text = "\n".join(contexts)
    else:
        context_text = contexts

    # Also include LONG_ANSWER if available
    long_answer = record.get("LONG_ANSWER")

    if long_answer and context_text:
        context_text = f"{context_text}\n\n{long_answer}"
    elif long_answer:
        context_text = long_answer

    answer = (
        record.get("final_decision")
        or record.get("LABEL")
        or record.get("label")
    )

    # PubMedQA PMID comes from the dict key
    pmid = record.get("__pmid__")

    return {
        "id": (
            pmid
            or record.get("pubid")
            or f"pubmedqa_{split}_{index}"
        ),
        "dataset": "PubMedQA",
        "split": split,
        "subset": None,
        "question": question,
        "options": {
            "A": "yes",
            "B": "no",
            "C": "maybe",
        },
        "answer": answer,
        "context": context_text,
    }


def normalize_bioasq(record, index, split):
    """
    Normalize a BioASQ record.

    BioASQ native format:
        body, type, id, ideal_answer, exact_answer,
        snippets, documents, concepts
    """

    question = (
        record.get("body")
        or record.get("question")
        or ""
    )

    exact_answer = record.get("exact_answer")
    ideal_answer = record.get("ideal_answer")

    # For ideal_answer, take the first one if it's a list
    if isinstance(ideal_answer, list) and ideal_answer:
        ideal_answer = ideal_answer[0]

    # Build context from snippets
    snippets = record.get("snippets")

    context = None

    if snippets and isinstance(snippets, list):

        snippet_texts = []

        for s in snippets:
            if isinstance(s, dict):
                snippet_texts.append(
                    s.get("text", "")
                )
            elif isinstance(s, str):
                snippet_texts.append(s)

        context = "\n---\n".join(snippet_texts)

    qtype = record.get("type", "unknown")

    # For yesno questions, structure options
    options = None

    if qtype == "yesno":
        options = {"A": "yes", "B": "no"}

    return {
        "id": (
            record.get("id")
            or f"bioasq_{split}_{index}"
        ),
        "dataset": "BioASQ",
        "split": split,
        "subset": qtype,
        "question": question,
        "options": options,
        "answer": exact_answer,
        "ideal_answer": ideal_answer,
        "context": context,
    }
