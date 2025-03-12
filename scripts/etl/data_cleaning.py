import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

def clean_movies_data(input_path="datasets/movies.csv", output_path="datasets/cleaned_movies.csv"):

    df = pd.read_csv(input_path, encoding="utf-8")

    df["year"] = df["title"].str.extract(r"\((\d{4})\)")  # Extrae años entre paréntesis
    df["title"] = df["title"].str.replace(r"\(\d{4}\)", "", regex=True)  # Elimina año del título
    df["title"] = df["title"].str.strip()  # Elimina espacios sobrantes

    df["genres"] = df["genres"].str.split("|")  # Convertir a lista
    df["genres"] = df["genres"].apply(lambda x: [] if x == ["(no genres listed)"] else x)  # Manejar casos sin géneros

    mlb = MultiLabelBinarizer()
    genres_encoded = pd.DataFrame(mlb.fit_transform(df["genres"]), columns=mlb.classes_)
    df = pd.concat([df, genres_encoded], axis=1)

    df["year"] = df["year"].fillna("Unknown")  # Años desconocidos
    df.dropna(subset=["title"], inplace=True)  # Eliminar filas sin título

    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Datos limpios guardados en: {output_path}")


if __name__ == "__main__":
    clean_movies_data()
