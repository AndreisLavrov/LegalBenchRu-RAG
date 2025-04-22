import pandas as pd
import os
import re
from typing import List, Dict, Tuple, Any
import numpy as np
from tqdm.auto import tqdm
import logging

from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter

from sentence_transformers import SentenceTransformer

import faiss

from sentence_transformers.cross_encoder import CrossEncoder

# --- Configuration ---

# --- Data and Corpus Paths ---
CSV_PATH = r'C:\Users\funny\OneDrive\Desktop\4Course_Project_HSE\Diploma\Data_For_RAG\RAG_data.csv'
# IMPORTANT: Set this to the directory containing your .txt legal documents
CORPUS_DIR = r'C:\Users\funny\OneDrive\Desktop\4Course_Project_HSE\Diploma\Data_For_RAG\corpus_files' # <--- CHANGE THIS TO YOUR CORPUS FOLDER

# --- Model Selection ---
# Choose a multilingual model suitable for Russian
EMBEDDING_MODEL_NAME = 'ai-forever/sbert_large_nlu_ru'
# Optional: Cross-encoder for reranking
RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2' # Or a multilingual one if needed

# --- RAG Pipeline Parameters ---
CHUNK_SIZE_NAIVE = 500 # Characters for naive chunking
CHUNK_OVERLAP_NAIVE = 50 # Characters overlap for naive chunking
CHUNK_SIZE_RCTS = 500 # Target chunk size for RCTS
CHUNK_OVERLAP_RCTS = 50 # Overlap for RCTS
# Увеличиваем K для поиска/реранкинга, если нужно
TOP_K_RETRIEVAL = 100 # Больше кандидатов для реранкера или BM25
TOP_K_RERANK = 64    # Если используется реранкер, смотрим на топ-64 после него
K_VALUES_EVAL = [1, 2, 4, 8, 16, 32, 64] # Считаем метрики до K=64

# Для BM25 тоже нужно изменить TOP_K_RETRIEVAL
# TOP_K_RETRIEVAL = 64 # Для BM25 - получаем сразу нужное число

# --- Flags ---
USE_RERANKER = True # Set to True to enable reranking step
CHUNK_STRATEGY = 'naive' # Options: 'naive', 'rcts'

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def load_corpus(corpus_dir: str, filenames: List[str]) -> Dict[str, str]:
    """Loads text content from specified files in the corpus directory."""
    corpus = {}
    logging.info(f"Loading corpus from {corpus_dir}...")
    missing_files = []
    for filename in tqdm(filenames, desc="Loading documents"):
        filepath = os.path.join(corpus_dir, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    corpus[filename] = f.read()
            except Exception as e:
                logging.warning(f"Could not read file {filename}: {e}")
                corpus[filename] = "" # Add empty string if reading fails
        else:
            logging.warning(f"File not found: {filepath}")
            missing_files.append(filename)
            corpus[filename] = "" # Add empty string for missing files
    if missing_files:
         logging.warning(f"Total missing files: {len(missing_files)}")
    return corpus

def chunk_text_naive(doc_text: str, chunk_size: int, chunk_overlap: int) -> List[Tuple[str, int, int]]:
    """Naive fixed-size chunking with overlap."""
    chunks = []
    start = 0
    doc_len = len(doc_text)
    while start < doc_len:
        end = start + chunk_size
        chunk_text = doc_text[start:end]
        chunks.append((chunk_text, start, min(end, doc_len)))
        # Move start for the next chunk, considering overlap
        start += chunk_size - chunk_overlap
        if start + chunk_overlap >= doc_len: # Avoid infinite loop on very small overlap/size
            break
    return chunks

class CharacterRecursiveTextSplitter(RecursiveCharacterTextSplitter):
    """Override to keep track of character indices."""
    def split_text(self, text: str) -> List[str]:
        # Use the parent class's splitting logic
        chunks = super().split_text(text)
        logging.warning("RCTS index tracking is approximate in this basic implementation.")
        return chunks

    def create_documents(
        self, texts: List[str], metadatas: List[dict] | None = None
    ) -> List[Dict[str, Any]]:
        """Create documents from texts with approximate index tracking."""
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        start_index = 0
        original_text = texts[0] # Assume first text is the full original for searching
        current_pos = 0
        split_texts = super().split_text(original_text) # Split the actual text

        for i, chunk_text in enumerate(split_texts):
            # Find the start of the chunk in the *remaining* text
            try:
                chunk_start_in_doc = original_text.index(chunk_text, current_pos)
                chunk_end_in_doc = chunk_start_in_doc + len(chunk_text)
                metadata = _metadatas[0].copy() # Use metadata of the original doc
                metadata["start_char"] = chunk_start_in_doc
                metadata["end_char"] = chunk_end_in_doc
                documents.append({"page_content": chunk_text, "metadata": metadata})
                current_pos = chunk_end_in_doc # Move search position forward
            except ValueError:
                logging.warning(f"Could not find chunk {i+1} in original text. Skipping.")
                pass


        return documents


def create_chunks(
    corpus: Dict[str, str],
    strategy: str = 'rcts',
    chunk_size: int = 500,
    chunk_overlap: int = 50
    ) -> List[Dict[str, Any]]:
    """Creates chunks from the corpus using the specified strategy."""
    all_chunks = []
    logging.info(f"Creating chunks using strategy: {strategy}")

    if strategy == 'naive':
        for filename, text in tqdm(corpus.items(), desc="Chunking documents (naive)"):
            if not text: continue # Skip empty docs
            doc_chunks = chunk_text_naive(text, chunk_size, chunk_overlap)
            for i, (chunk_text, start, end) in enumerate(doc_chunks):
                all_chunks.append({
                    'text': chunk_text,
                    'source_file': filename,
                    'chunk_id': f"{filename}_chunk_{i}",
                    'start_char': start,
                    'end_char': end
                })
    elif strategy == 'rcts':
        # RCTS needs careful index handling - simplified approach here
        text_splitter = RecursiveCharacterTextSplitter( # Or CharacterRecursiveTextSplitter for custom
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", " ", ""] # Common separators
        )
        chunk_counter = 0
        for filename, text in tqdm(corpus.items(), desc="Chunking documents (RCTS)"):
             if not text: continue # Skip empty docs
             # LangChain's split_text returns list of strings
             doc_chunk_texts = text_splitter.split_text(text)

             # We need to find the indices of these chunks in the original text
             current_pos = 0
             for chunk_text in doc_chunk_texts:
                 try:
                    start_index = text.index(chunk_text, current_pos)
                    end_index = start_index + len(chunk_text)
                    all_chunks.append({
                        'text': chunk_text,
                        'source_file': filename,
                        'chunk_id': f"{filename}_chunk_{chunk_counter}",
                        'start_char': start_index,
                        'end_char': end_index
                    })
                    current_pos = start_index + 1 # Move search start slightly forward
                    chunk_counter += 1
                 except ValueError:
                     logging.warning(f"Chunk not found sequentially in {filename}. May indicate splitter issues or text repetition. Skipping chunk.")
                     # If chunk not found, we might skip it or use approximate indices
                     current_pos += len(chunk_text) # Advance position anyway to avoid getting stuck


    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    logging.info(f"Created {len(all_chunks)} chunks.")
    return all_chunks

def embed_chunks(chunks: List[Dict[str, Any]], model: SentenceTransformer) -> np.ndarray:
    """Embeds the text of each chunk."""
    logging.info(f"Embedding {len(chunks)} chunks using {model}...")
    chunk_texts = [chunk['text'] for chunk in chunks]
    embeddings = model.encode(chunk_texts, show_progress_bar=True, normalize_embeddings=True)
    return embeddings.astype('float32') # FAISS uses float32

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Builds a FAISS index for the given embeddings."""
    logging.info("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension) # Better for cosine similarity with normalized embeddings
    index.add(embeddings)
    logging.info(f"FAISS index built. Contains {index.ntotal} vectors.")
    return index

def retrieve_top_k(
    query: str,
    embed_model: SentenceTransformer,
    index: faiss.Index,
    chunks_metadata: List[Dict[str, Any]],
    k: int
    ) -> List[Dict[str, Any]]:
    """Retrieves the top-k most relevant chunks for a query."""
    query_embedding = embed_model.encode([query], normalize_embeddings=True).astype('float32')
    # Search returns distances (L2) or similarities (IP) and indices
    similarities, indices = index.search(query_embedding, k)
    
    top_k_chunks = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        sim = similarities[0][i]
        if idx >= 0 and idx < len(chunks_metadata): # Basic check for valid index
            chunk_info = chunks_metadata[idx].copy() # Get metadata
            chunk_info['retrieval_score'] = sim
            top_k_chunks.append(chunk_info)
        else:
            logging.warning(f"Invalid index {idx} returned by FAISS search for query '{query[:50]}...'")

    return top_k_chunks

def rerank_chunks(
    query: str,
    chunks_to_rerank: List[Dict[str, Any]],
    rerank_model: CrossEncoder,
    top_n: int
    ) -> List[Dict[str, Any]]:
    """Reranks the retrieved chunks using a cross-encoder model."""
    if not chunks_to_rerank:
        return []
        
    pairs = [[query, chunk['text']] for chunk in chunks_to_rerank]
    logging.debug(f"Reranking {len(pairs)} pairs...")
    scores = rerank_model.predict(pairs)

    # Add scores to chunks and sort
    for i, chunk in enumerate(chunks_to_rerank):
        chunk['rerank_score'] = scores[i]

    reranked_chunks = sorted(chunks_to_rerank, key=lambda x: x['rerank_score'], reverse=True)

    return reranked_chunks[:top_n]


def check_overlap(retrieved_span: Tuple[int, int], ground_truth_span: Tuple[int, int]) -> bool:
    """Checks if two character spans overlap."""
    ret_start, ret_end = retrieved_span
    gt_start, gt_end = ground_truth_span
    # Check for any overlap: max of starts < min of ends
    return max(ret_start, gt_start) < min(ret_end, gt_end)

def evaluate_retrieval(
    retrieved_chunks: List[Dict[str, Any]],
    ground_truth_spans: List[Dict[str, Any]],
    k_values: List[int]
    ) -> Dict[str, Dict[int, float]]:
    """Calculates Precision@k and Recall@k based on character span overlap."""
    metrics = {'precision': {k: 0.0 for k in k_values}, 'recall': {k: 0.0 for k in k_values}}
    max_k = max(k_values)
    
    # Limit retrieved chunks to max_k for evaluation
    retrieved_at_max_k = retrieved_chunks[:max_k]
    if not ground_truth_spans: # No ground truth for this query
        return metrics
        
    total_gt_spans = len(ground_truth_spans)
    
    # Map retrieved chunks to their spans (file, start, end)
    retrieved_spans_info = [
        (c['source_file'], c['start_char'], c['end_char'])
        for c in retrieved_at_max_k
    ]
    
    # Map ground truth to spans (file, start, end)
    gt_spans_info = [
        (gt['ИсточникФайл'], gt['НачалоСимвола'], gt['КонецСимвола'])
        for gt in ground_truth_spans
    ]

    gt_hit = [False] * total_gt_spans # Track which GT spans have been hit

    relevant_retrieved_count = 0
    for i, retrieved_info in enumerate(retrieved_spans_info):
        ret_file, ret_start, ret_end = retrieved_info
        k = i + 1 # Current rank (1-based)
        
        is_relevant_hit = False # Did *this* retrieved chunk hit *any* GT span?
        
        for j, gt_info in enumerate(gt_spans_info):
            gt_file, gt_start, gt_end = gt_info
            
            # Check if spans are in the same file and overlap
            if ret_file == gt_file and check_overlap((ret_start, ret_end), (gt_start, gt_end)):
                is_relevant_hit = True
                gt_hit[j] = True # Mark this ground truth span as recalled by top-k

        if is_relevant_hit:
            relevant_retrieved_count += 1

        # Calculate cumulative metrics at k
        if k in k_values:
            metrics['precision'][k] = relevant_retrieved_count / k
            recalled_gt_count = sum(gt_hit)
            metrics['recall'][k] = recalled_gt_count / total_gt_spans

    last_precision = metrics['precision'][max_k] if max_k <= len(retrieved_spans_info) else 0
    last_recall = metrics['recall'][max_k] if max_k <= len(retrieved_spans_info) else 0

    for k in k_values:
         if k > len(retrieved_spans_info):
             metrics['precision'][k] = relevant_retrieved_count / k if k > 0 else 0 # Precision calculation includes k in denominator
             metrics['recall'][k] = last_recall # Recall doesn't change if no more items are retrieved


    return metrics

# --- Main Execution ---

if __name__ == "__main__":
    logging.info("Starting RAG Benchmark Script")

    # 1. Load Data
    try:
        df = pd.read_csv(CSV_PATH)
        logging.info(f"Loaded dataset: {CSV_PATH} with {len(df)} rows.")
        # Basic cleaning/validation
        df = df.dropna(subset=['Вопрос', 'ИсточникФайл', 'НачалоСимвола', 'КонецСимвола', 'ИзвлеченныйСниппет'])
        df['НачалоСимвола'] = pd.to_numeric(df['НачалоСимвола'], errors='coerce').astype('Int64')
        df['КонецСимвола'] = pd.to_numeric(df['КонецСимвола'], errors='coerce').astype('Int64')
        df = df.dropna(subset=['НачалоСимвола', 'КонецСимвола'])
        logging.info(f"Dataset after cleaning NAs: {len(df)} rows.")
        if len(df) == 0:
            raise ValueError("No valid data found after cleaning. Check CSV format and content.")
    except FileNotFoundError:
        logging.error(f"Dataset file not found: {CSV_PATH}")
        exit(1)
    except Exception as e:
        logging.error(f"Error loading or processing CSV: {e}")
        exit(1)

    # Group ground truth by query
    queries_data = {}
    for _, row in df.iterrows():
        query = row['Вопрос']
        if query not in queries_data:
            queries_data[query] = []
        queries_data[query].append(row.to_dict())

    unique_queries = list(queries_data.keys())
    logging.info(f"Found {len(unique_queries)} unique queries.")

    # 2. Load Corpus
    corpus_files = df['ИсточникФайл'].unique().tolist()
    corpus = load_corpus(CORPUS_DIR, corpus_files)
    if not any(corpus.values()): # Check if all files were missing or empty
        logging.error(f"Corpus directory '{CORPUS_DIR}' seems empty or files are unreadable.")
        exit(1)


    # 3. Create Chunks
    chunks_metadata = create_chunks(
        corpus,
        strategy=CHUNK_STRATEGY,
        chunk_size=CHUNK_SIZE_NAIVE if CHUNK_STRATEGY == 'naive' else CHUNK_SIZE_RCTS,
        chunk_overlap=CHUNK_OVERLAP_NAIVE if CHUNK_STRATEGY == 'naive' else CHUNK_OVERLAP_RCTS
    )
    if not chunks_metadata:
        logging.error("No chunks were created. Check corpus files and chunking strategy.")
        exit(1)

    # 4. Load Models and Embed Chunks
    logging.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    chunk_embeddings = embed_chunks(chunks_metadata, embed_model)

    # Optional: Load Reranker Model
    rerank_model = None
    if USE_RERANKER:
        try:
            logging.info(f"Loading reranker model: {RERANKER_MODEL_NAME}")
            rerank_model = CrossEncoder(RERANKER_MODEL_NAME)
        except Exception as e:
            logging.error(f"Failed to load reranker model {RERANKER_MODEL_NAME}: {e}")
            USE_RERANKER = False # Disable reranking if model loading fails

    # 5. Build Vector Index
    index = build_faiss_index(chunk_embeddings)

    # 6. Run Benchmark Loop
    all_results = []
    logging.info("Running benchmark...")
    for query in tqdm(unique_queries, desc="Processing queries"):
        # Retrieve
        retrieved_chunks = retrieve_top_k(query, embed_model, index, chunks_metadata, TOP_K_RETRIEVAL)

        # Rerank (Optional)
        final_chunks = retrieved_chunks
        if USE_RERANKER and rerank_model:
             final_chunks = rerank_chunks(query, retrieved_chunks, rerank_model, TOP_K_RERANK)


        # Evaluate
        ground_truth = queries_data[query]
        query_metrics = evaluate_retrieval(final_chunks, ground_truth, K_VALUES_EVAL)
        all_results.append(query_metrics)

    # 7. Aggregate and Report Results
    if not all_results:
        logging.error("No results were generated during the benchmark.")
        exit(1)
        
    avg_precision = {k: 0.0 for k in K_VALUES_EVAL}
    avg_recall = {k: 0.0 for k in K_VALUES_EVAL}
    num_queries_evaluated = len(all_results)

    for result in all_results:
        for k in K_VALUES_EVAL:
            avg_precision[k] += result['precision'][k]
            avg_recall[k] += result['recall'][k]

    for k in K_VALUES_EVAL:
        avg_precision[k] /= num_queries_evaluated
        avg_recall[k] /= num_queries_evaluated

    print("\n--- Benchmark Results ---")
    print(f"Strategy: {CHUNK_STRATEGY}")
    print(f"Embedding Model: {EMBEDDING_MODEL_NAME}")
    print(f"Reranker Used: {USE_RERANKER} ({RERANKER_MODEL_NAME if USE_RERANKER else 'N/A'})")
    print(f"Top K Retrieved: {TOP_K_RETRIEVAL}")
    if USE_RERANKER:
        print(f"Top K After Rerank: {TOP_K_RERANK}")
    print(f"Number of Queries: {num_queries_evaluated}")
    print("-" * 25)
    print("Average Precision@k:")
    for k in K_VALUES_EVAL:
        print(f"  P@{k:<2}: {avg_precision[k]:.4f}")
    print("-" * 25)
    print("Average Recall@k:")
    for k in K_VALUES_EVAL:
        print(f"  R@{k:<2}: {avg_recall[k]:.4f}")
    print("-" * 25)

    logging.info("Benchmark script finished.")