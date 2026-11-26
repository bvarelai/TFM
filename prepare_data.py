import json
import sys
import time
from pathlib import Path

from data import (
    clean_data,
    filter_min_interactions,
    get_dataset_stats,
    load_and_normalize_data,
)
from split import drop_items_not_in_train, split_data_temporal, validate_split

PROCESSED_DIR = Path("data/processed")

MIN_USER_INTERACTIONS = 5
MIN_ITEM_INTERACTIONS = 1
TEST_RATIO = 0.2
VAL_RATIO = 0.1

def prepare(dataset: str):
    print(f"\n=== {dataset} ===")
    timings = {}

    t0 = time.perf_counter()
    df = load_and_normalize_data(dataset)
    print("load:", type(df))

    timings["load_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    df = clean_data(df)
    print("load:", type(df))

    df = filter_min_interactions(df, MIN_USER_INTERACTIONS, MIN_ITEM_INTERACTIONS)
    print("load:", type(df))

    timings["clean_filter_s"] = time.perf_counter() - t0

    stats = get_dataset_stats(df)

    t0 = time.perf_counter()
    train, val, test = split_data_temporal(df, TEST_RATIO, VAL_RATIO)
    val, test = drop_items_not_in_train(train, val, test)
    validate_split(train, val, test)
    timings["split_s"] = time.perf_counter() - t0

    out_dir = PROCESSED_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    train.to_parquet(out_dir / "train.parquet", index=False)
    val.to_parquet(out_dir / "val.parquet", index=False)
    test.to_parquet(out_dir / "test.parquet", index=False)
    timings["save_s"] = time.perf_counter() - t0

    stats["split_sizes"] = {"train": len(train), "val": len(val), "test": len(test)}
    stats["params"] = {
        "min_user": MIN_USER_INTERACTIONS,
        "min_item": MIN_ITEM_INTERACTIONS,
        "test_ratio": TEST_RATIO,
        "val_ratio": VAL_RATIO,
    }
    stats["prep_timings"] = timings

    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"train={len(train)} val={len(val)} test={len(test)}")
    print(f"tiempos: { {k: round(v, 2) for k, v in timings.items()} }") 

if __name__ == "__main__":
    datasets = sys.argv[1:] or ["ml-100k", "ml-1m", "ml-10m"]
    for ds in datasets:
        prepare(ds)