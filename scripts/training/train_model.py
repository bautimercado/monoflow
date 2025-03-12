import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from scipy.sparse import save_npz, csr_matrix


def train_recommendation_model(
        input_path="../datasets/cleaned_movies.csv",
        output_path="model/",
        top_n=100
):
    """
    Entrena un modelo de recomendación basado en similitud de géneros.
    Almacena solo las top_n similitudes más altas para cada película.
    
    Args:
        input_path (str): Ruta al archivo CSV de películas procesadas
        output_dir (str): Directorio donde guardar el modelo y datos auxiliares
        top_n (int): Número de similitudes más altas para almacenar por película
    """
    print(f"Cargando datos desde {input_path}...")
    # Cargar datos limpios
    df = pd.read_csv(input_path)

    print("Procesando datos de géneros...")
    # Verificar que las columnas de géneros estén en formato correcto
    if isinstance(df["genres"].iloc[0], str) and df["genres"].iloc[0].startswith("["):
        # Convertir strings de lista a listas reales si es necesario
        df["genres"] = df["genres"].apply(eval)
    
    # Vectorizar géneros usando las columnas ya codificadas
    genre_columns = [col for col in df.columns if col in [
        'Action', 'Adventure', 'Animation', 'Children',
        'Comedy', 'Crime', 'Documentary', 'Drama',
        'Fantasy', 'Film-Noir', 'Horror', 'IMAX',
        'Musical', 'Mystery', 'Romance', 'Sci-Fi',
        'Thriller', 'War', 'Western'
    ]]

    if genre_columns:
        print(f"Usando {len(genre_columns)} géneros codificados...")
        genres_matrix = df[genre_columns].values.astype('float32')
    else:
        print("Codificando géneros con MultiLabelBinarizer...")
        mlb = MultiLabelBinarizer()
        genres_matrix = mlb.fit_transform(df["genres"]).astype("float32")

    print("Calculando similitudes por lotes y guardando las top-k...")
    os.makedirs(output_path, exist_ok=True)

    n_movies = len(df)
    batch_size = 1000   # Tamaño de lote

    # Crear diccionario para almacenar las top_n similitudes para cada peli
    top_similarities = {}

    for i in tqdm(range(0, n_movies, batch_size), desc="Calculando similitud"):
        end = min(i + batch_size, n_movies)
        batch = genres_matrix[i:end]

        # Calcular similitud entre el lote actual y todas las pelis
        batch_similiraties = cosine_similarity(batch, genres_matrix)

        # Para cada peli en el lote, guardar las top_n similitudes
        for j, idx in enumerate(range(i, end)):
            # Obtener índices de las top_n peliculas más similares
            sim_scores = batch_similiraties[j]
            sim_scores[idx] = -1

            # Obtener índices de las top_n peliculas más similares
            top_indexes = np.argsort(sim_scores)[-top_n:][::-1]
            top_scores = sim_scores[top_indexes]

            # Guardar solo índices con pnutuación positiva
            valid_mask = top_scores > 0
            top_similarities[idx] = {
                'indexes': top_indexes[valid_mask].tolist(),
                'scores': top_scores[valid_mask].tolist()
            }

        print("Guardando modelo y datos...")

    print("Guardando modelo y datos...")
    # Guardar similitudes en formato pickle
    similarity_path = os.path.join(output_path, "top_similarities.joblib")
    movies_path = os.path.join(output_path, "movie_ids.csv")

    joblib.dump(top_similarities, similarity_path)
    df[["movieId", "title", "year"]].to_csv(movies_path, index=False)

    # Guardar la matriz de géneros para uso futuro
    genres_path = os.path.join(output_path, "genres_matrix.npz")
    save_npz(genres_path, csr_matrix(genres_matrix))

    print(f"Modelo entrenado y guardado en {output_path}")
    return top_similarities, df


def get_recommendations(movie_id, top_n=5, model_path="model/"):
    """
    Obtiene recomendaciones para una película basadas en similitud de géneros.
    
    Args:
        movie_id (int): ID de la película para la que se buscan recomendaciones
        top_n (int): Número de recomendaciones a devolver
        model_dir (str): Directorio donde está guardado el modelo
        
    Returns:
        pandas.DataFrame: DataFrame con las películas recomendadas
    """
    similarity_path= os.path.join(model_path, "top_similarity.joblib")
    movies_path = os.path.join(model_path, "movies_ids.csv")

    top_similarities = joblib.load(similarity_path)
    movies_df = pd.read_csv(movies_path)

    # Encontrar índice de pelicula
    try:
        movie_idx = movies_df[movies_df["movieId"] == movie_id].index[0]
    except IndexError:
        return f"Película con ID {movie_id} no encontrada."

    # Verificar si hay similitudes para esa peli
    if movie_idx not in top_similarities:
        return f"No hay recomendaciones disponibles para la pelicula con ID {movie_id}"

    # Obtener índices y puntuaciones de peliculas similares
    similar_indexes = top_similarities[movie_idx]['indexes']
    similar_scores = top_similarities[movie_idx]['scores']

    # Limitar al número solicitado
    similar_indexes = similar_indexes[:top_n]
    similar_scores = similar_scores[:top_n]

    # Crear DF de resultados
    recommendations = movies_df.iloc[similar_indexes].copy()
    recommendations["similarity_score"] = similar_scores

    return recommendations


if __name__ == "__main__":
    train_recommendation_model()
