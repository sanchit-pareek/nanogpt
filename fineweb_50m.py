"""Build a bounded 50M training / 1M validation token FineWeb-Edu dataset.

Streams HuggingFaceFW/fineweb-edu sample-10BT; does not download the full sample.
Writes to a NEW directory, edu_fineweb_50M, without touching edu_fineweb10B.
Requires: pip install datasets tiktoken numpy tqdm
"""
from pathlib import Path
import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "edu_fineweb_50M"
STAGING = ROOT / "edu_fineweb_50M_building"
VAL_TOKENS = 1_000_000
TRAIN_SHARD_TOKENS = 5_000_000
NUM_TRAIN_SHARDS = 10


def main():
    if DEST.exists() or STAGING.exists():
        raise FileExistsError(
            f"Refusing to overwrite {DEST} or {STAGING}. "
            "If a prior build failed, inspect the staging folder before deleting it."
        )
    STAGING.mkdir(parents=True)
    try:
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )
        encoder = tiktoken.get_encoding("gpt2")
        eot = encoder._special_tokens["<|endoftext|>"]
        documents = iter(dataset)
        pending = np.empty(0, dtype=np.uint16)
        pending_at = 0

        outputs = [("edufineweb_val_000000.npy", VAL_TOKENS)]
        outputs.extend(
            (f"edufineweb_train_{i:06d}.npy", TRAIN_SHARD_TOKENS)
            for i in range(1, NUM_TRAIN_SHARDS + 1)
        )
        with tqdm(total=VAL_TOKENS + TRAIN_SHARD_TOKENS * NUM_TRAIN_SHARDS,
                  desc="FineWeb-Edu", unit="tok", unit_scale=True) as bar:
            for filename, count in outputs:
                shard = np.empty(count, dtype=np.uint16)
                filled = 0
                while filled < count:
                    if pending_at >= len(pending):
                        try:
                            doc = next(documents)
                        except StopIteration as exc:
                            raise RuntimeError("Dataset exhausted before requested token count") from exc
                        pending = np.asarray(
                            [eot] + encoder.encode_ordinary(doc["text"]),
                            dtype=np.uint16,
                        )
                        pending_at = 0
                    take = min(count - filled, len(pending) - pending_at)
                    shard[filled:filled + take] = pending[pending_at:pending_at + take]
                    filled += take
                    pending_at += take
                    bar.update(take)
                file_path = STAGING / filename
                with file_path.open("wb") as f:
                    np.save(f, shard)
                print(f"Saved {filename}: {count:,} tokens", flush=True)

        os.rename(STAGING, DEST)
        print(f"DONE: {DEST} (1M val + 50M train tokens)")
    except Exception:
        print(f"Build incomplete. Inspect: {STAGING}")
        raise


if __name__ == "__main__":
    main()
