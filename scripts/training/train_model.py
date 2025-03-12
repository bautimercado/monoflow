import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def train_recommendation_model(
        input_path="scripts/datasets/cleaned_data.csv",
        output_path="scripts/training/model/"
):

    print(f"Cargando datos desde {input_path}...")
    df = pd.read_csv(input_path)

    print("Procesando datos de géneros...")
    # Chequear que las columnnas generadas estén en el formato correcto
    if isinstance(df["genres"].iloc[0], str):
        df["genres"] = df["genres"].apply(eval)

    # Vectorizar géneros con MultiLabelBinarizer
    genre_columns = [col for col in df.columns if col in [
        'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
        'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
        'Thriller', 'War', 'Western'
    ]]

    # Si hay columnas de géneros codificadas, usarlas directamente
    if genre_columns:
        print(f"Usando {len(genre_columns)} géneros codificados...")
        genres_matrix = df[genre_columns].values
    else:
        # Si no, codificar desde 'genres'
        print("Codificando géneros con MultiLabelBinarizer...")
        mlb = MultiLabelBinarizer()
        genres_matrix = mlb.fit_transform(df["genres"])

    print("Calculando matriz de similitud (esto puede tomar tiempo)...")
    # Calcular similitud por lotes para mostrar progreso
    n_movies = len(df)
    batch_size = 1000
    similarity_matrix = np.zeros((n_movies, n_movies))

    for i in tqdm(
        range(0, n_movies, batch_size),
        desc="Calculando similitud"
    ):
        end = min(i + batch_size, n_movies)
        batch = genres_matrix[i:end]
        similarity_matrix[i:end] = cosine_similarity(batch, genres_matrix)

    os.makedirs(output_path, exist_ok=True)

    print("Guardando modelo y datos...")
    matrix_path = os.path.join(output_path, "similarity_matrix.joblib")
    movies_path = os.path.join(output_path, "movies_ids.csv")

    joblib.dump(similarity_matrix, matrix_path)
    df[["movieId", "title", "year"]].to_csv(movies_path, index=False)

    print(f"Modelo entrenado y guardado en '{output_path}'")

