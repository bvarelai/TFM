from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")

COLUMNS = [ "user_id", "item_id", "rating", "timestamp"]

# --------------------------------------------------------------------------
# Carga
# --------------------------------------------------------------------------
def load_and_normalize_data(dataset: str) -> pd.DataFrame:
    if dataset == "ml-100k":
        df = pd.read_csv(
            DATA_DIR / dataset / dataset / "u.data",
            sep="\t",
            names=COLUMNS
        )        
    elif dataset in ("ml-1m", "ml-10m"):
        df = pd.read_csv(
            DATA_DIR / dataset / dataset  / "ratings.dat",
            sep="::",
            engine="python",
            names=COLUMNS
        )
    else:
        raise ValueError(f"Dataset no soportado: {dataset}")

    return df.astype(
        {
            "user_id": "int32",
            "item_id": "int32",
            "rating": "float32",
            "timestamp": "int64"
        }
    )


# --------------------------------------------------------------------------
# Limpieza y filtrado
# --------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    n_inicial = len(df)

    # Filas incompletas
    df = df.dropna()
    # Duplicados exactos
    df = df.drop_duplicates().copy()
    
    # Detectar valoraciones repetidas usuario-item
    duplicated = df.duplicated(subset=["user_id", "item_id"], keep=False)
    if duplicated.any():
        raise ValueError(
            "Se han encontrado valoraciones repetidas "
            "para un mismo usuario y item."
        )    
    print(f"[clean] filas eliminadas: {n_inicial - len(df)}")
    return df


def filter_min_interactions(df, min_user: int, min_item: int) -> pd.DataFrame:
    """
    Filtra usuarios e items con pocas interacciones.
    Se repite hasta que ambos criterios se cumplan a la vez, porque
    quitar items puede dejar a un usuario por debajo del minimo.
    """
    n_inicial = len(df)

    while True:
      n_antes = len(df)
      
      ## Contar items distintos por usuario
      user_counts = df.groupby("user_id")["item_id"].nunique()  
      valid_users = user_counts[user_counts >= min_user].index
      df = df[df["user_id"].isin(valid_users)]
    
      ## Contar usuarios distintos por item
      item_counts = df.groupby("item_id")["user_id"].nunique()
      valid_items = item_counts[item_counts >= min_item].index
      df = df[df["item_id"].isin(valid_items)]

      if len(df) == n_antes:
          break
 
    print(f"[filter] interacciones eliminadas: {n_inicial - len(df)}")
    return df.copy()

# --------------------------------------------------------------------------
# Estadisticas
# --------------------------------------------------------------------------
def get_dataset_stats(df: pd.DataFrame) -> dict:
    """Devuelve las estadisticas como dict serializable a JSON."""
    num_users = int(df["user_id"].nunique())
    num_items = int(df["item_id"].nunique())
    num_interactions = int(len(df))

    per_user = df.groupby("user_id").size()
    per_item = df.groupby("item_id").size()

    return {
        "num_users": num_users,
        "num_items": num_items,
        "num_interactions": num_interactions,
        "sparsity": float(1 - num_interactions / (num_users * num_items)),
        "rating_mean": float(df["rating"].mean()),
        "rating_std": float(df["rating"].std()),
        "rating_distribution": {
            str(k): int(v)
            for k, v in df["rating"].value_counts().sort_index().items()
        },
        "interactions_per_user": {
            "min": int(per_user.min()),
            "median": float(per_user.median()),
            "mean": float(per_user.mean()),
            "max": int(per_user.max()),
        },
        "interactions_per_item": {
            "min": int(per_item.min()),
            "median": float(per_item.median()),
            "mean": float(per_item.mean()),
            "max": int(per_item.max()),
        },
        "timestamp_min": int(df["timestamp"].min()),
        "timestamp_max": int(df["timestamp"].max()),
        "memory_mb": float(df.memory_usage(deep=True).sum() / 1e6),
    }


def show_dataset_stats(df: pd.DataFrame) -> None:
    stats = get_dataset_stats(df)
    print(f"Numero de usuarios: {stats['num_users']}")
    print(f"Numero de peliculas: {stats['num_items']}")
    print(f"Numero de interacciones: {stats['num_interactions']}")
    print(f"Esparsidad: {stats['sparsity']:.4f}")
    print(f"Rating medio: {stats['rating_mean']:.3f} (std {stats['rating_std']:.3f})")
    print(f"Memoria: {stats['memory_mb']:.1f} MB")
