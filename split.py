import numpy as np
import pandas as pd

def split_data_temporal(df: pd.DataFrame, test_ratio: float = 0.2, val_ratio: float = 0.1):
    """
    Split temporal por usuario (vectorizado, sin bucle groupby).

    Para cada usuario, ordenado por timestamp:
      - las ultimas  test_ratio interacciones -> test
      - las siguientes val_ratio anteriores    -> val (para entrenar el ranker)
      - el resto                               -> train (para entrenar ALS)

    Cada usuario aporta como minimo 1 interaccion a test (y a val si
    val_ratio > 0). Es determinista: el desempate por item_id evita que
    los timestamps repetidos de MovieLens cambien el orden entre ejecuciones.
        

    Usuario con 10 valoraciones, ordenadas de más antigua a más reciente:

        [1][2][3][4][5][6][7] [8] [9][10]
        └──────  train ─────┘ └val┘└test┘
                (70 %)        (10%) (20%)

    """
    df = df.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)

    grp = df.groupby("user_id")["item_id"]
    pos = grp.cumcount()
    n = grp.transform("size")

    n_test = np.maximum(1, np.floor(n * test_ratio)).astype(int)
    if val_ratio > 0:
        n_val = np.maximum(1, np.floor(n * val_ratio)).astype(int)
    else:
        n_val = pd.Series(0, index=df.index)

    is_test = pos >= (n - n_test)
    is_val = (pos >= (n - n_test - n_val)) & ~is_test

    train = df[~is_test & ~is_val].reset_index(drop=True)
    val = df[is_val].reset_index(drop=True)
    test = df[is_test].reset_index(drop=True)
    return train, val, test


def drop_items_not_in_train(train: pd.DataFrame, *others: pd.DataFrame):
    """
    Quita de val/test las interacciones con items que no estan en train
    (ALS no puede predecirlos). Misma politica para todas las implementaciones.
    """
    known_items = set(train["item_id"])
    result = []
    for part in others:
        mask = part["item_id"].isin(known_items)
        print(f"[split] interacciones cold-item descartadas: {(~mask).sum()}")
        result.append(part[mask].reset_index(drop=True))
    return result


def validate_split(train, val, test):
    """Comprueba que el split es valido; lanza ValueError si no."""
    train_users = set(train["user_id"])

    for name, part in (("val", val), ("test", test)):
        if len(part) == 0:
            continue

        # Todo usuario de val/test debe existir en train
        if not set(part["user_id"]).issubset(train_users):
            raise ValueError(f"{name} contiene usuarios que no estan en train.")

        # No puede haber pares (usuario, item) repetidos entre particiones
        overlap = train.merge(part, on=["user_id", "item_id"], how="inner")
        if len(overlap) > 0:
            raise ValueError(f"Hay pares usuario-item compartidos entre train y {name}.")

    if len(val) > 0 and len(test) > 0:
        if len(val.merge(test, on=["user_id", "item_id"], how="inner")) > 0:
            raise ValueError("Hay pares usuario-item compartidos entre val y test.")

    # Sin fuga temporal: por usuario, train <= val <= test
    last_train = train.groupby("user_id")["timestamp"].max()
    first_test = test.groupby("user_id")["timestamp"].min()
    if (last_train.reindex(first_test.index) > first_test).any():
        raise ValueError("Fuga temporal: hay interacciones de train posteriores a test.")