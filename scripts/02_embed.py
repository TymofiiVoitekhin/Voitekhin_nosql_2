# scripts/02_embed.py
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

INPUT_PARQUET = "data/arxiv_subset.parquet"
OUTPUT_EMBEDDINGS = "embeddings/embeddings.npy"
MODEL_NAME = "allenai/specter2_base"

def main():
    # 1. Завантаження датасету
    print("Завантаження датасету...")
    df = pd.read_parquet(INPUT_PARQUET)
    
    # 2. Підготовка текстів для кодування
    # Об'єднуємо title та abstract через токен [SEP]
    # Застосовуємо fillna(""), щоб уникнути помилок типу TypeError при конкатенації
    print("Підготовка текстів для кодування...")
    texts = (df["title"].fillna("") + " [SEP] " + df["abstract"].fillna("")).tolist()
    
    # 3. Ініціалізація моделі
    print(f"Завантаження моделі {MODEL_NAME} з HuggingFace...")
    model = SentenceTransformer(MODEL_NAME)
    
    # 4. Кодування текстів в ембеддинги
    print("Початок генерації ембеддингів (обчислення можуть тривати кілька хвилин)...")
    embeddings = model.encode(
        texts,
        batch_size=64,                # Оптимізація пам'яті через батчеву обробку
        show_progress_bar=True,       # Візуалізація прогресу в терміналі
        normalize_embeddings=True     # L2-нормалізація (вектори одиничної довжини)
    )
    
    # 5. Вивід результатів у консоль
    print("\n--- Результати виконання ---")
    print(f"Загальна кількість оброблених текстів: {len(texts)}")
    print(f"Розмірність ембеддингів: {embeddings.shape}")
    
    # Перевірка норми першого вектора (L2 norm)
    # Через normalize_embeddings=True значення має бути близьким до 1.0
    first_vector_norm = np.linalg.norm(embeddings[0])
    print(f"Норма першого ембеддингу: {first_vector_norm:.4f}")
    
    # 6 & 7. Збереження отриманих ембеддингів
    print("\nІніціалізація збереження файлу на диск...")
    os.makedirs(os.path.dirname(OUTPUT_EMBEDDINGS), exist_ok=True)
    np.save(OUTPUT_EMBEDDINGS, embeddings)
    print(f"Ембеддинги успішно збережено у: {OUTPUT_EMBEDDINGS}")

if __name__ == "__main__":
    main()