# GPT-2 (124M) Pretraining from Scratch

A PyTorch implementation of a GPT-2–style autoregressive language model, built to explore the mechanics of transformer pretraining at scale—from dataset preparation and distributed optimization to validation, language modeling benchmarks, and text generation.

This project implements the model architecture and training workflow in a compact, readable codebase, with support for multi-GPU training using PyTorch DistributedDataParallel (DDP).

## Overview

| Component | Details |
| --- | --- |
| Model | GPT-2–style decoder-only Transformer (~124M parameters) |
| Architecture | 12 transformer blocks, 12 attention heads, 768-dimensional embeddings |
| Context length | Up to 1,024 tokens; training configured for 256-token sequences |
| Dataset | FineWeb-Edu `sample-10BT` |
| Tokenization | GPT-2 tokenizer via `tiktoken` |
| Optimizer | AdamW with warmup and cosine learning-rate decay |
| Training | Gradient accumulation, gradient clipping, DDP, bf16 mixed precision |
| Evaluation | Validation loss and HellaSwag multiple-choice accuracy |
| Artifacts | Training logs, evaluation logs, checkpoints, generated text samples |

## Architecture

The model follows the GPT-2 small configuration, with:

- Learned token and position embeddings
- Causal multi-head self-attention
- Feed-forward MLP blocks with GELU activations
- Residual connections and layer normalization
- Tied input-embedding and language-model-head weights

The implementation uses PyTorch's scaled dot-product attention to take advantage of optimized attention kernels where supported.

## Training pipeline

### 1. Dataset preparation

`fineweb.py` downloads and tokenizes the FineWeb-Edu `sample-10BT` dataset subset, then stores token IDs in NumPy shards under `edu_fineweb10B/` for efficient streaming during training.

The data loader constructs batches of contiguous token sequences and next-token targets. The training setup uses separate training and validation shards.

### 2. Optimization

`train_gpt2.py` implements a full next-token prediction training loop with:

- AdamW parameter optimization
- Learning-rate warmup followed by cosine decay
- Gradient accumulation for a larger effective batch size
- Gradient clipping for stability
- bf16 autocast on compatible GPUs
- PyTorch DistributedDataParallel for multi-GPU training

The checked-in script configures a microbatch size of `32`, a sequence length of `256`, an effective batch size of `524,288` tokens, `19,073` maximum steps, and a peak learning rate of `6e-4`. These settings can be changed directly in the script to match available hardware.

### 3. Evaluation

The training loop evaluates language modeling performance using held-out validation cross-entropy and measures commonsense sentence-completion performance using **HellaSwag**.

For HellaSwag, the model scores candidate completions using average token-level negative log-likelihood and selects the lowest-loss option. Evaluation is run periodically during training and at the final step.

### 4. Checkpoints and sampling

The pipeline saves model checkpoints and emits text samples to track training behavior. Checkpoints contain model weights, configuration, training step, and validation loss.

## Repository structure

```text
.
├── train_gpt2.py     # Transformer model, training, evaluation, and sampling
├── fineweb.py        # FineWeb-Edu download, tokenization, and shard creation
├── hellaswag.py      # HellaSwag dataset utilities and evaluation helpers
├── requirements.txt  # Python environment dependencies
├── test.py           # Development and test code
├── test_hf.py        # Hugging Face comparison and test code
└── play.ipynb        # Experiments and exploration
```

## Getting started

### Requirements

- Python and PyTorch
- CUDA-capable GPU recommended for training
- Multiple CUDA GPUs for DDP training
- Sufficient disk space for the FineWeb-Edu token shards

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Ensure your PyTorch installation matches your CUDA environment. See the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/) if a different build is needed.

### Prepare the dataset

```bash
python fineweb.py
```

The script creates token shards in `edu_fineweb10B/`. Downloading and processing this dataset requires substantial storage and network bandwidth.

### Train on one GPU

```bash
python train_gpt2.py
```

### Train on multiple GPUs

For example, to launch eight distributed processes on a single node with eight GPUs:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt2.py
```

Adjust `--nproc_per_node` to match your GPU count. Training hyperparameters are configured inside `train_gpt2.py`.

## Logs and evaluation outputs

Training writes console output and structured records to `log/log.txt`, including training loss, validation loss, and HellaSwag accuracy:

```text
<step> train <loss>
<step> val <loss>
<step> hella <accuracy>
```

The lines above illustrate the logging format rather than benchmark scores. Model checkpoints are stored as `log/model_XXXXX.pt` at configured intervals.

**Note:** The published repository focuses on the model and training implementation; run-specific checkpoint files and final benchmark scores are not bundled with the source code. The `sample-10BT` dataset designation describes the dataset subset used for preprocessing, not a claim about the number of tokens consumed by a specific training run.

The training script currently opens `log/log.txt` in write mode at startup. Preserve any existing logs and checkpoints before starting another run.

`hellaswag.py` also supports evaluating reference pretrained GPT-2 models. Results from those pretrained models should be distinguished from evaluations performed within the custom training loop.

## Implementation notes

- Multi-GPU execution uses PyTorch DDP with the NCCL backend on CUDA.
- Checkpoints store model state and metadata; they are not full optimizer-and-RNG-state snapshots for exact training resumption.
- The script uses in-file hyperparameter configuration rather than command-line training flags.
- GPU memory, runtime, and throughput depend heavily on the hardware and chosen batch/sequence settings.

## References

- [OpenAI — Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Hugging Face — FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [HellaSwag benchmark](https://rowanzellers.com/hellaswag/)

## Author

**Sanchit Pareek** · [GitHub](https://github.com/sanchit-pareek)
