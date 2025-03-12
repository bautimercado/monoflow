import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

def clean_movies_data(input_path="datasets/movies.csv", output_path="datasets/cleaned_movies.csv"):

    df = pd.read_csv(input_path, encoding="utf-8")

    # Extraer año del título
    df["year"] = df["title"].str.extract(r"\((\d{4}(?:-\d{4})?)\)(?=[^()]*$)")

    # Eliminar año del título (solo el último par de paréntesis)
    df["title"] = df["title"].str.replace(r"\s*\(\d{4}(?:-\d{4})?\)(?=[^()]*$)", "", regex=True)
    df["title"] = df["title"].str.strip()

    # Convertir generos a lista
    df["genres"] = df["genres"].str.split("|")
    df["genres"] = df["genres"].apply(lambda x: [] if x == ["(no genres listed)"] else x)

    # Codificar generos (manteniendo la columna original)
    mlb = MultiLabelBinarizer()
    genres_encoded = pd.DataFrame(
        mlb.fit_transform(df["genres"]),
        columns=mlb.classes_,
        index=df.index
    )

    # Añadir columnas para verificación
    df["num_genres"] = df["genres_list"].apply(len)

    # Unir DataFrames
    df = pd.concat([df, genres_encoded], axis=1)

    # Manejar datos faltantes
    df["year"] = df["year"].fillna("Unknown")
    df.dropna(subset=["title"], inplace=True)

    # Verificar posibles duplicados
    potential_duplicates = df[df.duplicated(subset=["title"], keep=False)]
    if not potential_duplicates.empty:
        print(f"Se encontraron {len(potential_duplicates)} posibles peliculas duplicadas")

    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Datos limpios guardados en: {output_path}")


if __name__ == "__main__":
    clean_movies_data()
