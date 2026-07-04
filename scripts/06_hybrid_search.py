# scripts/06_hybrid_search.py
import os
import torch
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K_FETCH = 50   # Скільки документів діставати для RRF
TOP_K_DISPLAY = 5  # Скільки виводити у консоль

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def build_bm25_corpus(df):
    """Підготовка текстів для BM25: токенізація заголовків та анотацій."""
    print("Побудова локального індексу BM25...")
    tokenized_corpus = []
    # Об'єднуємо title та abstract, переводимо в нижній регістр і розбиваємо по пробілах
    for _, row in df.iterrows():
        text = str(row['title']) + " " + str(row['abstract'])
        tokens = text.lower().split()
        tokenized_corpus.append(tokens)
    return tokenized_corpus

def search_bm25(bm25, df, query, top_k):
    """Повертає список словників з arxiv_id та рангом."""
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    
    # Отримуємо індекси топ документів
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for rank, idx in enumerate(top_indices):
        # Якщо score == 0, значить збігів немає взагалі, можемо ігнорувати
        if scores[idx] > 0:
            arxiv_id = str(df.iloc[idx]['id'])
            results.append({
                'arxiv_id': arxiv_id,
                'rank': rank + 1,  # Ранг починається з 1
                'bm25_score': scores[idx]
            })
    return results

def search_vector(index, model, query, top_k):
    """Повертає список словників з arxiv_id та рангом від Pinecone."""
    query_vector = model.encode(query, normalize_embeddings=True).tolist()
    
    response = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    
    results = []
    for rank, match in enumerate(response['matches']):
        results.append({
            'arxiv_id': match['metadata']['arxiv_id'],
            'rank': rank + 1,
            'vector_score': match['score']
        })
    return results

def reciprocal_rank_fusion(bm25_results, vector_results, k=60):
    """
    Об'єднує результати за допомогою RRF.
    Повертає відсортований список словників.
    """
    rrf_scores = {}
    
    # Обробка результатів BM25
    for item in bm25_results:
        arxiv_id = item['arxiv_id']
        rank = item['rank']
        rrf_scores[arxiv_id] = rrf_scores.get(arxiv_id, 0.0) + (1.0 / (k + rank))
        
    # Обробка результатів Векторного пошуку
    for item in vector_results:
        arxiv_id = item['arxiv_id']
        rank = item['rank']
        rrf_scores[arxiv_id] = rrf_scores.get(arxiv_id, 0.0) + (1.0 / (k + rank))
        
    # Сортування за RRF-скором (за спаданням)
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results

def print_results(results, df, method_name, max_items=5):
    """Форматований вивід результатів."""
    print(f"\n--- {method_name} ---")
    if not results:
        print("Нічого не знайдено.")
        return
        
    for i, item in enumerate(results[:max_items]):
        # Обробка різних структур даних залежно від методу
        if isinstance(item, tuple): # RRF повертає tuple (arxiv_id, score)
            arxiv_id, score = item[0], item[1]
            score_str = f"RRF Score: {score:.5f}"
        else: # BM25 та Векторний повертають словник
            arxiv_id = item['arxiv_id']
            if 'bm25_score' in item:
                score_str = f"BM25 Score: {item['bm25_score']:.2f}"
            else:
                score_str = f"Vector Score: {item['vector_score']:.4f}"
                
        # Отримуємо назву статті з DataFrame
        article = df[df['id'] == arxiv_id].iloc[0]
        title = article['title']
        
        print(f"[{i+1}] {score_str} | ID: {arxiv_id}")
        print(f"    {title[:100]}...")

def execute_hybrid_search(bm25, vector_index, model, df, query):
    """Виконує всі три типи пошуку для запиту і виводить результати."""
    print(f"\n{'='*60}\nЗАПИТ: '{query}'\n{'='*60}")
    
    # 1. Пошук BM25 (додаємо більше документів для кращого перетину в RRF)
    bm25_res = search_bm25(bm25, df, query, top_k=TOP_K_FETCH)
    print_results(bm25_res, df, "ТОП-5 BM25 (Лексичний)", TOP_K_DISPLAY)
    
    # 2. Векторний пошук
    vector_res = search_vector(vector_index, model, query, top_k=TOP_K_FETCH)
    print_results(vector_res, df, "ТОП-5 ВЕКТОРНИЙ (Семантичний)", TOP_K_DISPLAY)
    
    # 3. Гібридний пошук (RRF)
    hybrid_res = reciprocal_rank_fusion(bm25_res, vector_res)
    print_results(hybrid_res, df, "ТОП-5 ГІБРИДНИЙ (RRF)", TOP_K_DISPLAY)

def main():
    print("Ініціалізація інфраструктури...")
    device = get_device()
    
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    # Для цього скрипта ми підключаємось до оригінального індексу arxiv-papers
    # Якщо ви його видалили на попередньому кроці, доведеться запустити 03_load_to_pinecone.py ще раз
    index = pc.Index(INDEX_NAME) 
    
    model = SentenceTransformer(MODEL_NAME, device=device)
    df = pd.read_parquet("data/arxiv_subset.parquet").reset_index(drop=True)
    
    # Побудова BM25
    tokenized_corpus = build_bm25_corpus(df)
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Виконання тестових запитів
    queries = [
        "BERT fine-tuning",
        "Yann LeCun convolutional networks",
        "making computers understand human emotions from text"
    ]
    
    for query in queries:
        execute_hybrid_search(bm25, index, model, df, query)

if __name__ == "__main__":
    main()