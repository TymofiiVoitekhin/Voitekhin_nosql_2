# scripts/05_chunking.py
import os
import re
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768
INDEX_FIXED = "arxiv-chunks-fixed"
INDEX_SEMANTIC = "arxiv-chunks-semantic"
BATCH_SIZE = 100

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def create_index_if_not_exists(pc, index_name):
    """Створює індекс, обробляючи очікування його готовності."""
    if index_name not in pc.list_indexes().names():
        print(f"Створення індексу {index_name}...")
        try:
            pc.create_index(
                name=index_name,
                dimension=VECTOR_DIM,
                metric="dotproduct",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            while not pc.describe_index(index_name).status['ready']:
                time.sleep(1)
            print(f"Індекс {index_name} готовий.")
        except Exception as e:
            print(f"\nКРИТИЧНА ПОМИЛКА: Не вдалося створити індекс {index_name}.")
            print("Ймовірно, ви досягли ліміту безкоштовного тарифу Pinecone (1 індекс).")
            print("Видаліть попередній індекс 'arxiv-papers' через веб-інтерфейс Pinecone і перезапустіть скрипт.")
            raise e
    else:
        print(f"Індекс {index_name} вже існує.")

def chunk_fixed_size(text, size=50, overlap=10):
    """Розбиття фіксованим розміром слів із перекриттям."""
    words = text.split()
    chunks = []
    step = size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + size])
        chunks.append(chunk)
        if i + size >= len(words):
            break
    return chunks

def chunk_semantic(text, max_words=50):
    """Розбиття по реченнях із лімітом слів."""
    # Розбиваємо по крапці, знаку оклику/питання, за якими йде пробіл
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence: continue
        
        words_in_sentence = len(sentence.split())
        
        if current_length + words_in_sentence > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = words_in_sentence
        else:
            current_chunk.append(sentence)
            current_length += words_in_sentence
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def process_and_upload_chunks(pc, model, df_top30, chunk_strategy, index_name):
    """Генерує чанки, векторизує та завантажує їх у Pinecone."""
    index = pc.Index(index_name)
    vectors_to_upsert = []
    
    print(f"\nГенерація чанків для індексу: {index_name}...")
    for _, row in df_top30.iterrows():
        arxiv_id = str(row['id'])
        abstract = str(row['abstract'])
        
        if chunk_strategy == 'fixed':
            chunks = chunk_fixed_size(abstract)
        else:
            chunks = chunk_semantic(abstract)
            
        for i, chunk_text in enumerate(chunks):
            vector_id = f"{arxiv_id}_chunk_{i}"
            # Кодуємо текст (нормалізовано для dotproduct)
            embedding = model.encode(chunk_text, normalize_embeddings=True).tolist()
            
            metadata = {
                "arxiv_id": arxiv_id,
                "title": str(row['title'])[:200],
                "chunk_text": chunk_text[:500], # Обмеження для метаданих
                "chunk_id": i,
                "year": int(row['year']),
                "category": str(row['category'])
            }
            
            vectors_to_upsert.append({
                "id": vector_id,
                "values": embedding,
                "metadata": metadata
            })
            
    # Завантаження батчами
    print(f"Завантаження {len(vectors_to_upsert)} векторів у Pinecone...")
    for i in tqdm(range(0, len(vectors_to_upsert), BATCH_SIZE)):
        batch = vectors_to_upsert[i:i + BATCH_SIZE]
        index.upsert(vectors=batch)
        
    print(f"Завантаження завершено. Векторів в індексі: {index.describe_index_stats().total_vector_count}")

def test_search(pc, model, index_name, query, title="Результати"):
    """Виконує тестовий пошук і виводить результати."""
    index = pc.Index(index_name)
    query_vector = model.encode(query, normalize_embeddings=True).tolist()
    
    results = index.query(vector=query_vector, top_k=3, include_metadata=True)
    
    print(f"\n{'-'*50}\n{title}\n{'-'*50}")
    for i, match in enumerate(results['matches']):
        m = match['metadata']
        print(f"[{i+1}] Score: {match['score']:.4f} | Article: {m['title']}")
        print(f"    Chunk text: {m['chunk_text']}...\n")

def main():
    print("Ініціалізація інфраструктури...")
    device = get_device()
    print(f"Використовується обчислювальний пристрій: {device.upper()}")
    
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    model = SentenceTransformer(MODEL_NAME, device=device)
    df = pd.read_parquet("data/arxiv_subset.parquet")
    
    # 1. Відбір 30 статей з найдовшими анотаціями
    df['abstract_len'] = df['abstract'].apply(lambda x: len(str(x).split()))
    df_top30 = df.sort_values(by='abstract_len', ascending=False).head(30)
    print(f"Відібрано 30 статей. Середня довжина анотації: {df_top30['abstract_len'].mean():.0f} слів.")
    
    # Створення індексів
    create_index_if_not_exists(pc, INDEX_FIXED)
    create_index_if_not_exists(pc, INDEX_SEMANTIC)
    
    # Обробка та завантаження
    process_and_upload_chunks(pc, model, df_top30, 'fixed', INDEX_FIXED)
    process_and_upload_chunks(pc, model, df_top30, 'semantic', INDEX_SEMANTIC)
    
    # 6. Тестові пошукові запити
    test_query = "quantum state optimization and entanglement"
    print(f"\nЗапит: '{test_query}'")
    
    test_search(pc, model, INDEX_FIXED, test_query, "Пошук по чанках: FIXED-SIZE")
    test_search(pc, model, INDEX_SEMANTIC, test_query, "Пошук по чанках: SEMANTIC")

if __name__ == "__main__":
    main()