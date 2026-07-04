# scripts/03_load_to_pinecone.py
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"
INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
BATCH_SIZE = 200

def main():
    print("Ініціалізація підключення до Pinecone...")
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("API ключ Pinecone не знайдено. Перевірте файл .env.")
    
    pc = Pinecone(api_key=api_key)

    # 1. Створюємо індекс (якщо не існує)
    existing_indexes = pc.list_indexes().names()
    if INDEX_NAME not in existing_indexes:
        print(f"Створення індексу '{INDEX_NAME}' (це може зайняти кілька хвилин)...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=VECTOR_DIM,
            # Використовуємо dotproduct, оскільки наші вектори L2-нормалізовані
            metric="dotproduct", 
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1" # Стандартний регіон для безкоштовного тарифу Starter
            )
        )
        # Очікування готовності індексу
        while not pc.describe_index(INDEX_NAME).status['ready']:
            time.sleep(1)
        print("Індекс успішно створено.")
    else:
        print(f"Індекс '{INDEX_NAME}' вже існує. Підключення...")

    index = pc.Index(INDEX_NAME)

    # 2. Завантаження локальних даних
    print("Завантаження локальних даних з Parquet та Numpy масивів...")
    df = pd.read_parquet(INPUT_PARQUET)
    embeddings = np.load(INPUT_EMBEDDINGS)
    total_records = len(df)
    
    print(f"Дані завантажено. Підготовка {total_records} векторів до відправки...")

    # 3 & 4. Підготовка даних та батчеве завантаження
    for i in tqdm(range(0, total_records, BATCH_SIZE), desc="Завантаження батчів у Pinecone"):
        i_end = min(i + BATCH_SIZE, total_records)
        batch_df = df.iloc[i:i_end]
        batch_embeddings = embeddings[i:i_end]

        vectors_to_upsert = []
        for j, (_, row) in enumerate(batch_df.iterrows()):
            global_idx = i + j
            vector_id = f"paper_{global_idx}"

            # Формування метаданих з обмеженням довжини рядків
            metadata = {
                "arxiv_id": str(row["id"]),
                "title": str(row["title"]),
                "abstract": str(row["abstract"])[:500],
                "authors": str(row["authors"])[:200],
                "year": int(row["year"]),
                "category": str(row["category"])
            }

            # Pinecone вимагає стандартні списки Python замість numpy array
            vector_values = batch_embeddings[j].tolist()

            vectors_to_upsert.append({
                "id": vector_id,
                "values": vector_values,
                "metadata": metadata
            })

        # Відправка сформованого батчу
        index.upsert(vectors=vectors_to_upsert)

    # 5. Вивід загальної кількості векторів
    print("\n--- Результати виконання ---")
    stats = index.describe_index_stats()
    print(f"Статистика індексу: {stats}")
    print(f"Загальна кількість векторів в індексі '{INDEX_NAME}': {stats.total_vector_count}")

if __name__ == "__main__":
    main()