# scripts/04_search.py
import os
import torch
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5
EMBEDDINGS_PATH = "embeddings/embeddings.npy"

def encode_query(model, query: str) -> list:
    """Генерує нормалізований ембеддинг для текстового запиту."""
    embedding = model.encode(query, normalize_embeddings=True)
    return embedding.tolist()

def print_results(results, df, title="Результати пошуку"):
    """Форматований вивід результатів."""
    print(f"\n{'='*50}\n{title}\n{'='*50}")
    for i, match in enumerate(results['matches']):
        score = match['score']
        metadata = match['metadata']
        # Отримуємо повний abstract з локального DataFrame
        arxiv_id = metadata['arxiv_id']
        full_abstract = df[df['id'] == arxiv_id]['abstract'].values[0]
        
        print(f"[{i+1}] Score: {score:.4f} | ID: {arxiv_id}")
        print(f"Назва: {metadata['title']}")
        print(f"Категорія: {metadata['category']} | Рік: {metadata['year']}")
        print(f"Анотація: {full_abstract[:200]}...\n")

def main():
    print("Ініціалізація інфраструктури...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Використовується обчислювальний пристрій: {device.upper()}")
    
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(INDEX_NAME)
    model = SentenceTransformer(MODEL_NAME, device=device)
    df = pd.read_parquet("data/arxiv_subset.parquet")

    test_query = "teaching machines to recognize objects in pictures"
    print(f"\nЗапит: '{test_query}'")
    query_vector = encode_query(model, test_query)

    # 3. Чистий семантичний пошук
    print("\nВиконання запиту до Pinecone (Чистий семантичний пошук)...")
    pure_results = index.query(
        vector=query_vector,
        top_k=TOP_K,
        include_metadata=True
    )
    print_results(pure_results, df, "3. ЧИСТИЙ СЕМАНТИЧНИЙ ПОШУК")

    # 4. Пошук з фільтрацією метаданих
    print("Виконання запитів до Pinecone (З фільтрацією метаданих)...")
    
    # Приклад A: reinforcement learning, останні 5 років (>= 2021), категорія cs.LG
    query_a = "reinforcement learning"
    vector_a = encode_query(model, query_a)
    filter_a = {
        "category": {"$eq": "cs.LG"},
        "year": {"$gte": 2021}
    }
    results_a = index.query(vector=vector_a, top_k=TOP_K, filter=filter_a, include_metadata=True)
    print_results(results_a, df, "4A. ФІЛЬТР: RL, >=2019, cs.LG")

    # Приклад B: старі статті (< 2015), будь-яка категорія
    filter_b = {
        "year": {"$lt": 2015}
    }
    results_b = index.query(vector=vector_a, top_k=TOP_K, filter=filter_b, include_metadata=True)
    print_results(results_b, df, "4B. ФІЛЬТР: RL, <2015, Будь-яка категорія")

    # 5. Локальне порівняння метрик схожості
    print("\nЗавантаження локальних ембеддингів для аналізу метрик...")
    local_embeddings = np.load(EMBEDDINGS_PATH)
    q_vec_np = np.array(query_vector)

    # Оскільки вектори нормалізовані:
    # Dot Product
    dot_scores = np.dot(local_embeddings, q_vec_np)
    # Cosine Similarity (ідентичний dot product для одиничних векторів, але обчислюємо явно для доведення)
    norms = np.linalg.norm(local_embeddings, axis=1) * np.linalg.norm(q_vec_np)
    cosine_scores = dot_scores / norms
    # L2 Distance
    l2_distances = np.linalg.norm(local_embeddings - q_vec_np, axis=1)

    # Отримання топ-5 індексів для кожної метрики
    # argsort сортує за зростанням, тому для dot і cosine беремо з кінця [::-1]
    top_dot_idx = np.argsort(dot_scores)[::-1][:TOP_K]
    top_cos_idx = np.argsort(cosine_scores)[::-1][:TOP_K]
    # Для відстані L2 менше значення = кращий збіг (беремо з початку)
    top_l2_idx = np.argsort(l2_distances)[:TOP_K]

    print("\n" + "="*50)
    print("5. ПОРІВНЯННЯ ЛОКАЛЬНИХ МЕТРИК")
    print("="*50)
    
    print(f"Топ-5 індексів (Dot Product):   {top_dot_idx}")
    print(f"Топ-5 індексів (Cosine Sim):    {top_cos_idx}")
    print(f"Топ-5 індексів (L2 Distance):   {top_l2_idx}")
    
    # Перевірка збігу масивів
    arrays_match = np.array_equal(top_dot_idx, top_cos_idx) and np.array_equal(top_cos_idx, top_l2_idx)
    print(f"\nВисновок: Індекси топ-5 документів для всіх трьох метрик збігаються: {arrays_match}")

if __name__ == "__main__":
    main()