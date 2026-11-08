"""Prepare a bounded FineWeb-Edu sample for local nanoGPT smoke tests.

Run from the nanoGPT project directory:
    python fineweb_small.py

Outputs one validation and one training .npy shard in edu_fineweb10B/.
Use only for short experimental runs; this is NOT the full 10B-token corpus.
"""
from pathlib import Path
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent / "edu_fineweb10B"
VAL_TOKENS = 50_000
TRAIN_TOKENS = 500_000


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        (DATA_DIR / "edufineweb_val_000000.npy", VAL_TOKENS),
        (DATA_DIR / "edufineweb_train_000001.npy", TRAIN_TOKENS),
    ]
    existing = [str(p) for p, _ in outputs if p.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing shard(s): " + ", ".join(existing)
        )

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True
    )
    encoder = tiktoken.get_encoding("gpt2")
    eot = encoder._special_tokens["<|endoftext|>"]
    iterator = iter(dataset)

    for filename, target in outputs:
        tokens = np.empty(target, dtype=np.uint16)
        filled = 0
        with tqdm(total=target, desc=filename.stem, unit="tokens") as pbar:
            while filled < target:
                try:
                    document = next(iterator)
                except StopIteration as exc:
                    raise RuntimeError("Dataset exhausted before target token counts") from exc
                doc_tokens = [eot] + encoder.encode_ordinary(document["text"])
                take = min(len(doc_tokens), target - filled)
                tokens[filled:filled + take] = np.asarray(doc_tokens[:take], dtype=np.uint16)
                filled += take
                pbar.update(take)
        np.save(filename, tokens)
        print(f"Saved {filename} ({filled:,} tokens)")

    print("Done. Training loader can now find train/val shards.")


if __name__ == "__main__":
    main()
