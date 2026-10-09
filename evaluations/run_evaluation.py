"""
This is the MAIN ENTRY POINT script that ties everything together. 
It takes an LLM as input and evaluates it on the MedQA dataset using a RAG pipeline.

Usage Examples:
    # OpenAI
    python run_evaluation.py --provider openai --model gpt-3.5-turbo --api-key sk-xxx --num-questions 50
    
    # Gemini
    python run_evaluation.py --provider gemini --model gemini-pro --api-key AIza-xxx
    
    # Local Ollama
    python run_evaluation.py --provider ollama --model llama3 --num-questions 20
    
    # HuggingFace (local)
    python run_evaluation.py --provider huggingface --model meta-llama/Meta-Llama-3-8B-Instruct
    
    # Custom OpenAI-compatible endpoint (vLLM, LM Studio)
    python run_evaluation.py --provider openai --model my-model --base-url http://localhost:8000/v1 --api-key dummy
    
    # Quick test (10 questions, no faithfulness)
    python run_evaluation.py --provider ollama --model llama3 --num-questions 10 --skip-faithfulness
"""

import os
import json
import argparse
import random
import re
import numpy as np
from tqdm import tqdm

try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from llm_interface import create_llm, BaseLLM, OpenAILLM, OllamaLLM, HuggingFaceLLM, GeminiLLM
from rag_eval_metrics import RetrievalMetrics, GenerationMetrics, FaithfulnessMetrics, compute_all_metrics
from rag_pipeline import TextChunker, VectorStore, MedQADataset, RAGPipeline

def extract_answer_letter(response: str) -> str:
    """Extract answer letter (A/B/C/D/E) from LLM response."""
    if not response:
        return ''
    
    # Try exact answer format first
    match = re.search(r'(?i)(?:answer is|answer:|correct option is)\s*([A-E])', response)
    if match:
        return match.group(1).upper()
        
    # Check if the response starts with a letter and a period/parenthesis
    match = re.search(r'^\s*([A-E])[\.\)]', response)
    if match:
        return match.group(1).upper()
        
    # Just find the first A-E uppercase letter
    match = re.search(r'\b([A-E])\b', response)
    if match:
        return match.group(1).upper()
        
    return ''

def print_header(text: str):
    """Print a formatted section header."""
    if RICH_AVAILABLE:
        console.print(f"\n[bold cyan]=== {text} ===[/bold cyan]\n")
    else:
        print(f"\n=== {text} ===\n")

def print_results_table(gen_metrics, faith_metrics, accuracy, total, correct):
    """Print a nice summary table."""
    if RICH_AVAILABLE:
        table = Table(title="Evaluation Results")
        table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right", style="magenta")

        table.add_row("Accuracy (MCQ)", f"{accuracy*100:.1f}% ({correct}/{total})")
        
        if gen_metrics.get('exact_match') is not None:
            table.add_row("Exact Match", f"{gen_metrics['exact_match']:.4f}")
        if gen_metrics.get('f1') is not None:
            table.add_row("F1 Score", f"{gen_metrics['f1']:.4f}")
        if gen_metrics.get('bleu') is not None:
            table.add_row("BLEU", f"{gen_metrics['bleu']:.4f}")
        if gen_metrics.get('rouge_l') is not None:
            table.add_row("ROUGE-L", f"{gen_metrics['rouge_l']:.4f}")
        if gen_metrics.get('bert_score_f1') is not None:
            table.add_row("BERTScore F1", f"{gen_metrics['bert_score_f1']:.4f}")
            
        if faith_metrics.get('faithfulness') is not None:
            table.add_row("Faithfulness", f"{faith_metrics['faithfulness']:.4f}")
        if faith_metrics.get('answer_relevance') is not None:
            table.add_row("Answer Relevance", f"{faith_metrics['answer_relevance']:.4f}")
        if faith_metrics.get('hallucination_rate') is not None:
            table.add_row("Hallucination Rate", f"{faith_metrics['hallucination_rate']:.4f}")
            
        console.print(table)
    else:
        print("\n" + "="*40)
        print("EVALUATION RESULTS")
        print("="*40)
        print(f"Accuracy (MCQ): {accuracy*100:.1f}% ({correct}/{total})")
        for k, v in gen_metrics.items():
            if k != 'accuracy' and v is not None:
                print(f"{k.replace('_', ' ').title()}: {v:.4f}")
        for k, v in faith_metrics.items():
            if v is not None:
                print(f"{k.replace('_', ' ').title()}: {v:.4f}")
        print("="*40 + "\n")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LLMs on MedQA using RAG")
    parser.add_argument("--provider", type=str, choices=['openai', 'gemini', 'huggingface', 'ollama'], required=True, help="LLM provider")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., 'gpt-3.5-turbo', 'gemini-pro')")
    parser.add_argument("--api-key", type=str, default=None, help="API key")
    parser.add_argument("--base-url", type=str, default=None, help="Custom API endpoint")
    parser.add_argument("--num-questions", type=int, default=50, help="Number of questions to evaluate")
    parser.add_argument("--top-k", type=int, default=5, help="Number of documents to retrieve")
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size for index")
    parser.add_argument("--chunk-overlap", type=int, default=64, help="Chunk overlap for index")
    parser.add_argument("--embedding-model", type=str, default='all-MiniLM-L6-v2', help="Embedding model name")
    parser.add_argument("--output", type=str, default='eval_results.json', help="Output JSON file")
    parser.add_argument("--index-path", type=str, default='medqa_index', help="Path to FAISS index")
    parser.add_argument("--skip-faithfulness", action="store_true", help="Skip LLM-as-judge metrics")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Step 1: Initialize LLM
    print_header('Step 1: Initializing LLM')
    llm = create_llm(args.provider, args.model, api_key=args.api_key, base_url=args.base_url)
    
    # Step 2: Load dataset
    print_header('Step 2: Loading MedQA Dataset')
    dataset = MedQADataset(zip_path='data_clean.zip')  # same folder
    questions = dataset.load_questions(split='test', max_questions=args.num_questions)
    print(f'Loaded {len(questions)} questions')
    
    # Step 3: Build or load vector index
    print_header('Step 3: Building/Loading Vector Index')
    vector_store = VectorStore(embedding_model=args.embedding_model)
    index_file = os.path.join(args.index_path, 'index.faiss')
    if os.path.exists(index_file):
        print('Loading existing index...')
        vector_store.load(args.index_path)
    else:
        print('Building index from textbooks (this may take a few minutes)...')
        textbooks = dataset.load_textbooks()
        chunker = TextChunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        all_chunks = []
        for book in textbooks:
            chunks = chunker.chunk_text(book['text'], source=book['source'])
            all_chunks.extend(chunks)
            print(f"  Chunked {book['source']}: {len(chunks)} chunks")
        print(f'Total chunks: {len(all_chunks)}')
        vector_store.build_index(all_chunks)
        os.makedirs(args.index_path, exist_ok=True)
        vector_store.save(args.index_path)
        print(f'Index saved to {args.index_path}')
    
    # Step 4: Initialize RAG pipeline
    print_header('Step 4: Running RAG Evaluation')
    rag = RAGPipeline(llm=llm, vector_store=vector_store, top_k=args.top_k)
    
    # Step 5: Run evaluation loop
    results = []
    predictions = []
    references = []
    all_questions = []
    all_contexts = []
    correct = 0
    
    for i, q in enumerate(tqdm(questions, desc='Evaluating')):
        formatted_q = MedQADataset.format_question_with_options(q)
        result = rag.run(query=formatted_q, options=q.get('options'))
        
        # Extract predicted answer letter
        pred_letter = extract_answer_letter(result['response'])
        gt_letter = q['answer_idx']
        is_correct = (pred_letter == gt_letter)
        if is_correct:
            correct += 1
        
        # Store for metrics
        predictions.append(result['response'])
        references.append(q['answer'])
        all_questions.append(formatted_q)
        all_contexts.append(result['context'])
        
        results.append({
            'question_idx': i,
            'question': q['question'],
            'options': q.get('options', {}),
            'ground_truth': gt_letter,
            'ground_truth_text': q['answer'],
            'predicted_letter': pred_letter,
            'response': result['response'],
            'correct': is_correct,
            'num_docs_retrieved': len(result['retrieved_docs']),
            'retrieved_sources': [d['source'] for d in result['retrieved_docs']],
        })
    
    # Step 6: Compute metrics
    print_header('Step 5: Computing Metrics')
    accuracy = correct / len(questions)
    
    # Generation metrics
    gen_metrics = {}
    gen_metrics['accuracy'] = accuracy
    try:
        gen_metrics['exact_match'] = np.mean([GenerationMetrics.exact_match(p, r) for p, r in zip(predictions, references)])
    except: gen_metrics['exact_match'] = None
    
    try:
        gen_metrics['f1'] = np.mean([GenerationMetrics.f1_score(p, r) for p, r in zip(predictions, references)])
    except: gen_metrics['f1'] = None
    
    try:
        gen_metrics['bleu'] = np.mean([GenerationMetrics.bleu_score(p, r) for p, r in zip(predictions, references)])
    except: gen_metrics['bleu'] = None
    
    try:
        gen_metrics['rouge_l'] = np.mean([GenerationMetrics.rouge_l_score(p, r) for p, r in zip(predictions, references)])
    except: gen_metrics['rouge_l'] = None
    
    try:
        bert_scores = GenerationMetrics.bert_score(predictions, references)
        gen_metrics['bert_score_f1'] = bert_scores['f1']
    except: gen_metrics['bert_score_f1'] = None
    
    # Faithfulness metrics (optional, uses API calls)
    faith_metrics = {}
    if not args.skip_faithfulness:
        print('Computing faithfulness metrics (LLM-as-Judge)...')
        sample_size = min(20, len(predictions))  # Sample to save API costs
        indices = random.sample(range(len(predictions)), sample_size)
        
        faith_scores = []
        relevance_scores = []
        halluc_scores = []
        for idx in tqdm(indices, desc='Faithfulness'):
            try:
                fs = FaithfulnessMetrics.faithfulness_score(predictions[idx], all_contexts[idx], llm)
                if fs is not None: faith_scores.append(fs)
                
                rs = FaithfulnessMetrics.answer_relevance_score(predictions[idx], all_questions[idx], llm)
                if rs is not None: relevance_scores.append(rs)
                
                hs = FaithfulnessMetrics.hallucination_score(predictions[idx], all_contexts[idx], llm)
                if hs is not None: halluc_scores.append(hs)
            except Exception as e: 
                pass
        
        faith_metrics['faithfulness'] = np.mean(faith_scores) if faith_scores else None
        faith_metrics['answer_relevance'] = np.mean(relevance_scores) if relevance_scores else None
        faith_metrics['hallucination_rate'] = np.mean(halluc_scores) if halluc_scores else None
    
    # Step 7: Generate report
    print_header('Step 6: Results Summary')
    final_report = {
        'config': {
            'provider': args.provider,
            'model': args.model,
            'num_questions': len(questions),
            'top_k': args.top_k,
            'chunk_size': args.chunk_size,
            'embedding_model': args.embedding_model,
        },
        'generation_metrics': gen_metrics,
        'faithfulness_metrics': faith_metrics,
        'per_question_results': results,
    }
    
    # Pretty print summary
    print_results_table(gen_metrics, faith_metrics, accuracy, len(questions), correct)
    
    # Save
    with open(args.output, 'w') as f:
        json.dump(final_report, f, indent=2, default=str)
    print(f'\nFull results saved to {args.output}')

if __name__ == '__main__':
    main()
