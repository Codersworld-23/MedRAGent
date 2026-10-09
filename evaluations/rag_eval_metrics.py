import re
from typing import List, Dict, Any, Union

class _LLMHelper:
    @staticmethod
    def _parse_score(response: str) -> float:
        """Extract a numeric score from the LLM response."""
        match = re.search(r"[-+]?\d*\.\d+|\d+", response)
        if match:
            return float(match.group())
        return 0.0

class RetrievalMetrics:
    """Metrics for evaluating the retrieval component of RAG."""

    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        """Calculate Precision at k."""
        retrieved_k = retrieved[:k]
        if not retrieved_k:
            return 0.0
        relevant_set = set(relevant)
        hits = sum(1 for doc_id in retrieved_k if doc_id in relevant_set)
        return hits / len(retrieved_k)

    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        """Calculate Recall at k."""
        if not relevant:
            return 0.0
        retrieved_k = retrieved[:k]
        relevant_set = set(relevant)
        hits = sum(1 for doc_id in retrieved_k if doc_id in relevant_set)
        return hits / len(relevant)

    @staticmethod
    def f1_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        """Calculate F1 score at k."""
        p = RetrievalMetrics.precision_at_k(retrieved, relevant, k)
        r = RetrievalMetrics.recall_at_k(retrieved, relevant, k)
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)

    @staticmethod
    def mrr(retrieved_lists: List[List[str]], relevant_lists: List[List[str]]) -> float:
        """Calculate Mean Reciprocal Rank over multiple queries."""
        if not retrieved_lists:
            return 0.0
        rr_sum = 0.0
        for retrieved, relevant in zip(retrieved_lists, relevant_lists):
            relevant_set = set(relevant)
            for i, doc_id in enumerate(retrieved):
                if doc_id in relevant_set:
                    rr_sum += 1.0 / (i + 1)
                    break
        return rr_sum / len(retrieved_lists)

    @staticmethod
    def map_at_k(retrieved_lists: List[List[str]], relevant_lists: List[List[str]], k: int) -> float:
        """Calculate Mean Average Precision at k."""
        if not retrieved_lists:
            return 0.0
        ap_sum = 0.0
        for retrieved, relevant in zip(retrieved_lists, relevant_lists):
            relevant_set = set(relevant)
            retrieved_k = retrieved[:k]
            hits = 0
            score_sum = 0.0
            for i, doc_id in enumerate(retrieved_k):
                if doc_id in relevant_set:
                    hits += 1
                    score_sum += hits / (i + 1)
            if relevant_set:
                ap_sum += score_sum / min(len(relevant_set), k)
        return ap_sum / len(retrieved_lists)

    @staticmethod
    def hit_at_k(retrieved: List[str], relevant: List[str], k: int) -> bool:
        """Determine if there is at least one relevant document in top k."""
        retrieved_k = retrieved[:k]
        relevant_set = set(relevant)
        return any(doc_id in relevant_set for doc_id in retrieved_k)

    @staticmethod
    def context_relevance_score(query: str, retrieved_docs: List[str], llm: Any) -> float:
        """Use LLM-as-Judge to score context relevance 1-5, return normalized score 0-1."""
        context = "\n".join(retrieved_docs)
        prompt = f"Rate the relevance of the following context to the query on a scale of 1 to 5, where 1 is completely irrelevant and 5 is perfectly relevant. Only output the number.\n\nQuery: {query}\n\nContext:\n{context}\n\nScore (1-5):"
        response = llm.generate(prompt)
        score = _LLMHelper._parse_score(response)
        score = max(1.0, min(5.0, score))
        return (score - 1) / 4.0

class GenerationMetrics:
    """Metrics for evaluating the generation component of RAG."""

    @staticmethod
    def exact_match(prediction: str, ground_truth: str) -> float:
        """Calculate Exact Match score."""
        return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0

    @staticmethod
    def f1_score(prediction: str, ground_truth: str) -> float:
        """Calculate token-level F1 score."""
        pred_tokens = prediction.strip().lower().split()
        truth_tokens = ground_truth.strip().lower().split()
        
        common = sum(1 for token in set(pred_tokens) if token in truth_tokens)
        if not common:
            return 0.0
            
        precision = common / len(pred_tokens)
        recall = common / len(truth_tokens)
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def bleu_score(prediction: str, reference: str) -> float:
        """Calculate BLEU score using NLTK."""
        import nltk
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        pred_tokens = prediction.strip().lower().split()
        ref_tokens = [reference.strip().lower().split()]
        smooth = SmoothingFunction().method1
        return sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smooth)

    @staticmethod
    def rouge_l_score(prediction: str, reference: str) -> float:
        """Calculate ROUGE-L score."""
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        return scores['rougeL'].fmeasure

    @staticmethod
    def bert_score(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate BERTScore."""
        try:
            import bert_score
            p, r, f = bert_score.score(predictions, references, model_type='microsoft/deberta-xlarge-mnli', lang="en", verbose=False)
            return {
                'precision': p.mean().item(),
                'recall': r.mean().item(),
                'f1': f.mean().item()
            }
        except ImportError:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        except Exception:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

class FaithfulnessMetrics:
    """Metrics to evaluate the faithfulness of generated answers to the context."""

    @staticmethod
    def faithfulness_score(answer: str, context: str, llm: Any) -> float:
        """Calculate faithfulness score using LLM-as-Judge (0 to 1)."""
        prompt = f"Given the following context and an answer, rate from 0.0 to 1.0 how faithful the answer is to the context. A score of 1.0 means all claims in the answer are supported by the context. A score of 0.0 means none are supported. Only output the numeric score.\n\nContext:\n{context}\n\nAnswer:\n{answer}\n\nScore (0-1):"
        response = llm.generate(prompt)
        score = _LLMHelper._parse_score(response)
        return max(0.0, min(1.0, score))

    @staticmethod
    def answer_relevance_score(answer: str, question: str, llm: Any) -> float:
        """Calculate answer relevance score using LLM-as-Judge (0 to 1)."""
        prompt = f"Rate the relevance of the following answer to the given question on a scale of 0.0 to 1.0, where 1.0 is perfectly relevant and directly answers the question, and 0.0 is completely irrelevant. Only output the numeric score.\n\nQuestion:\n{question}\n\nAnswer:\n{answer}\n\nScore (0-1):"
        response = llm.generate(prompt)
        score = _LLMHelper._parse_score(response)
        return max(0.0, min(1.0, score))

    @staticmethod
    def hallucination_score(answer: str, context: str, llm: Any) -> float:
        """Calculate hallucination score using LLM-as-Judge (0 to 1)."""
        prompt = f"Analyze the answer based on the context. Provide a score from 0.0 to 1.0 representing the fraction of claims in the answer that are NOT supported by the context (i.e., hallucinations). 0.0 means no hallucination, 1.0 means fully hallucinated. Only output the numeric score.\n\nContext:\n{context}\n\nAnswer:\n{answer}\n\nScore (0-1):"
        response = llm.generate(prompt)
        score = _LLMHelper._parse_score(response)
        return max(0.0, min(1.0, score))

def compute_all_metrics(predictions: List[str], references: List[str], questions: List[str], contexts: List[str], llm: Any = None) -> Dict[str, Any]:
    """Compute all generation and faithfulness metrics."""
    results = {}
    
    # Generation metrics
    exact_matches = [GenerationMetrics.exact_match(p, r) for p, r in zip(predictions, references)]
    f1_scores = [GenerationMetrics.f1_score(p, r) for p, r in zip(predictions, references)]
    bleu_scores = [GenerationMetrics.bleu_score(p, r) for p, r in zip(predictions, references)]
    rouge_l_scores = [GenerationMetrics.rouge_l_score(p, r) for p, r in zip(predictions, references)]
    
    results['exact_match'] = sum(exact_matches) / len(exact_matches) if exact_matches else 0.0
    results['f1_score'] = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    results['bleu_score'] = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    results['rouge_l_score'] = sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0.0
    
    bert_results = GenerationMetrics.bert_score(predictions, references)
    results['bert_precision'] = bert_results.get('precision', 0.0)
    results['bert_recall'] = bert_results.get('recall', 0.0)
    results['bert_f1'] = bert_results.get('f1', 0.0)

    # Faithfulness/Relevance metrics if LLM is provided
    if llm:
        faithfulness = [FaithfulnessMetrics.faithfulness_score(p, c, llm) for p, c in zip(predictions, contexts)]
        answer_relevance = [FaithfulnessMetrics.answer_relevance_score(p, q, llm) for p, q in zip(predictions, questions)]
        hallucination = [FaithfulnessMetrics.hallucination_score(p, c, llm) for p, c in zip(predictions, contexts)]
        
        results['faithfulness_score'] = sum(faithfulness) / len(faithfulness) if faithfulness else 0.0
        results['answer_relevance_score'] = sum(answer_relevance) / len(answer_relevance) if answer_relevance else 0.0
        results['hallucination_score'] = sum(hallucination) / len(hallucination) if hallucination else 0.0
        
    return results
