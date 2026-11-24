from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")

def load_and_normalize_data(dataset: str) -> pd.DataFrame:
    
    if dataset == "ml-100k":
        path = DATA_DIR / dataset / "u.data"
        df = pd.read_csv(
            path,
            sep="\t",
            names=[
                "user_id",
                "item_id",
                "rating",
                "timestamp"
            ]
        )
    elif dataset in ("ml-1m", "ml-10m"):
        path = DATA_DIR / dataset / "ratings.dat"
        df = pd.read_csv(
            path,
            sep="::",
            engine="python",
            names=[
                "user_id",
                "item_id",
                "rating",
                "timestamp"
            ]
        )
    else:
        raise ValueError(
            f"Dataset no soportado: {dataset}"
        )
    df["user_id"] = df["user_id"].astype("int32")
    df["item_id"] = df["item_id"].astype("int32")
    df["rating"] = df["rating"].astype("float32")
    df["timestamp"] = df["timestamp"].astype("int64")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    # Filas incopletas
    df = df.dropna()
    # Duplicados exactos
    df = df.drop_duplicates().copy()
    # Detectar valoraciones repetidas usuario-pelicula
    duplicated = df.duplicated(
        subset=["user_id", "item_id"],
        keep=False
    )

    if duplicated.any():
        raise ValueError(
            "Se han encontrado valoraciones repetidas "
            "para un mismo usuario y pelicula."
        )    
    return df

def filter_users(df, min_interactions=5) -> pd.DataFrame:

    #user_counts = df["user_id"].value_counts()                  ## Interactions por usuario
    user_counts = df.groupby("user_id")["item_id"].nunique()  ## Contar peliculas distintas por usuario
    valid_users = user_counts[user_counts >= min_interactions].index
    return df[df["user_id"].isin(valid_users)].copy()

def show_dataset_stats(df: pd.DataFrame):
    
    num_users = df["user_id"].nunique()
    num_items = df["item_id"].nunique()
    num_interactions = len(df)
    sparsity = 1 - (num_interactions / (num_users * num_items))

    print(f"Número de usuarios: {num_users}")
    print(f"Número de películas: {num_items}")
    print(f"Número de interacciones: {num_interactions}")
    print(f"Esparsidad: {sparsity:.4f}")

"""
def split_data(df: pd.DataFrame, test_ratio: float = 0.2):
    
    df = df.sort_values(
        ["user_id", "timestamp"]
    )

    train_parts = []
    test_parts = []

    for _, group in df.groupby("user_id"):
        split_idx = int(len(group) * (1 - test_ratio))

        train_parts.append(group.iloc[:split_idx])
        test_parts.append(group.iloc[split_idx:])

    train = pd.concat(train_parts).reset_index(drop=True)
    test = pd.concat(test_parts).reset_index(drop=True)

    return train, test

def validate_split(train, test):
    
    train_users = set(train["user_id"])
    test_users = set(test["user_id"])

    if not test_users.issubset(train_users):
        raise ValueError(
            "El conjunto de prueba contiene usuarios "
            "que no están presentes en el conjunto de entrenamiento."
        )

def get_dataset_stats(df):
    
    num_users = df["user_id"].nunique()
    num_items = df["item_id"].nunique()
    num_interactions = len(df)
    sparsity = 1 - (num_interactions / (num_users * num_items))

    return {
        "num_users": num_users,
        "num_items": num_items,
        "num_interactions": num_interactions,
        "sparsity": sparsity
    }


"""
