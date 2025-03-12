import pandas as pd
from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure, OperationFailure
from dotenv import load_dotenv
import os
from tqdm import tqdm

def load_to_mongodb():
    try:
        load_dotenv()
        mongodb_uri = os.getenv("MONGODB_URI")

        if not mongodb_uri:
            raise ValueError("No se encontró MONGODB_URI")

        client = MongoClient(mongodb_uri)

        # Verificar conexión
        client.admin.command("ping")
        print("Conexión a MongoDB establecida")

        db = client.movie_recommender
        collection = db.movies

        print("Cargando datos...")
        df = pd.read_csv("../datasets/cleaned_movies.csv")

        if isinstance(df["genres"].iloc[0], str):
            print("Convirtiendo géneros a listas...")
            df["genres"] = df["genres"].apply(eval)
        
        print("Procesando columna year...")
        df["year"] = pd.to_numeric(df["year"], errors="ignore")

        print("Preparando documentos para MongoDB...")
        documents = df.to_dict("records")

        print("Eliminando datos existentes...")
        collection.delete_many({})

        print("Insertando documentos...")
        batch_size = 1000
        for i in tqdm(range(0, len(documents), batch_size)):
            batch = documents[i:i + batch_size]
            collection.insert_many(batch)
        
        print("Creando índice para movieId...")
        collection.create_index([("movieId", ASCENDING)], unique=True)

        count = collection.count_documents({})
        print(f"Proceso completado: {count} documentos insertados correctamente")

    except ConnectionFailure as e:
        print(f"Error de conexión a MongoDB: {e}")
    except OperationFailure as e:
        print(f"Error de operación en MongoDB: {e}")
    except ValueError as e:
        print(f"Error de valor: {e}")
    except Exception as e:
        print(f"Error inesperado: {e}")
    finally:
        if "client" in locals():
            print("Cerrando conexión con MongoDB")
            client.close()


if __name__ == "__main__":
    load_to_mongodb()
