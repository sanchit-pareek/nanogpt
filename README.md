GPT-2 (124M) Pretraining from Scratch
A PyTorch implementation of a GPT-2-style 124M-parameter decoder-only Transformer, covering model architecture, FineWeb-Edu preprocessing, gradient-based pretraining, checkpointing, text generation, and language-model evaluation.

The repository includes a general GPT-2 training implementation with PyTorch DistributedDataParallel (DDP) support, alongside a single-GPU, resource-constrained experiment conducted on an NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM).
Highlights
- Built GPT-2-style transformer blocks with learned token/position embeddings, causal multi-head attention, GELU MLPs, residual connections, layer normalization, and tied input/output embeddings.
- Implemented next-token language-model training using AdamW, learning-rate warmup and cosine decay, gradient accumulation, gradient clipping, and BF16 mixed precision.
- Added structured metrics, checkpoint-based resumption, HellaSwag evaluation, and generated-text sampling.
- Implemented a multi-GPU DDP training path in the general training script; the experiment reported below ran on one GPU.
- Measured and plotted training/validation loss for a 49.15M-token experiment on consumer hardware.
Model architecture
| Component | Configuration |
| --- | --- |
| Architecture | GPT-2-style causal decoder-only Transformer |
| Parameters | Approximately 124M |
| Transformer blocks | 12 |
| Attention heads | 12 |
| Embedding dimension | 768 |
| Maximum context length | 1,024 tokens |
| Tokenizer | GPT-2 BPE (tiktoken) |
| Attention | PyTorch scaled dot-product causal attention |
| Objective | Next-token cross-entropy |


The measured local experiment used 512-token training sequences; the model architecture supports sequences up to 1,024 tokens.
Experiment: 50M-token FineWeb-Edu pretraining
The following results are from one local run trained from random initialization, using tokenized FineWeb-Edu data. They are not results from the pretrained OpenAI GPT-2 model.
Configuration
| Parameter | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU (6GB) |
| Training implementation | PyTorch, CUDA, BF16 autocast |
| Dataset | FineWeb-Edu sample-10BT source, bounded streaming subset |
| Unique training tokens prepared | 50,000,000 |
| Validation tokens prepared | 1,000,000 |
| Optimizer updates | 3,000 |
| Tokens per optimizer update | 16,384 |
| Total tokens processed | 49,152,000 |
| Micro-batch size | 1 |
| Sequence length | 512 |
| Gradient accumulation | 32 |
| Optimizer | AdamW |
| Peak learning rate | 6e-4 |
| Learning-rate schedule | Warmup followed by cosine decay |
| Checkpoint interval | Every 250 optimizer steps |
| Validation interval | Every 250 optimizer steps |


Measured results
| Metric | Result |
| --- | --- |
| First logged training loss | 10.9461 |
| Final logged training loss | 4.6301 |
| First logged validation loss (step 250) | 6.5567 |
| Last logged validation loss (step 2999) | 4.9956 |
| HellaSwag subset accuracy (independent checkpoint evaluation) | 28.91% (37 / 128) |


Validation loss decreased at all 12 logged evaluation points. The final validation measurement was performed before the last optimizer update, whereas the saved final checkpoint represents the completed 3,000 updates.

HellaSwag accuracy was measured independently from the final checkpoint on 128 validation examples, not the full HellaSwag benchmark. The score is near the 25% random-choice baseline and is not a claim of strong commonsense reasoning ability.

The original run's metrics.csv contains a known invalid zero HellaSwag entry caused by comparing string labels with integer predictions in the training-time evaluator. The standalone eval_hellaswag.py evaluator converts labels correctly and produced the independently recorded 37/128 result in hellaswag_eval_128.json. The training-time comparison is corrected in the published overnight script.
Training curves

![Training and validation loss curves](results/rtx4050_50m/loss_curve.png)

![Validation loss curve](results/rtx4050_50m/validation_curve.png)

Source artifacts: [`metrics.csv`](results/rtx4050_50m/metrics.csv) and [`hellaswag_eval_128.json`](results/rtx4050_50m/hellaswag_eval_128.json).
Interpretation
The run demonstrates functioning optimization and improved next-token prediction, with validation loss continuing to decrease throughout the experiment. The 49.15M-token budget is much smaller than full GPT-2 pretraining, so the result should be understood as a practical training-systems and learning experiment, not a reproduction of GPT-2's pretrained quality.
Repository guide
| File | Purpose |
| --- | --- |
| train_gpt2.py | General model/training implementation; includes DDP support |
| train_gpt2_overnight.py | Configurable, single-GPU-friendly experiment, run logging and resumable checkpoints |
| fineweb.py | Larger FineWeb-Edu dataset preprocessing |
| fineweb_50m.py | Bounded streaming preparation for the 50M-token experiment |
| fineweb_small.py | Small quick-test dataset preparation |
| hellaswag.py | HellaSwag dataset parsing/rendering utilities |
| eval_hellaswag.py | Independent checkpoint-based HellaSwag subset evaluation |
| plot_training.py | Training/validation curve generation from metrics.csv |
| results/rtx4050_50m/ | Published plots, recorded metrics and benchmark summary |


Large tokenized datasets, raw logs, and model checkpoints are excluded from ordinary Git history.
Getting started
Requirements
- Python and a compatible CUDA-enabled PyTorch installation for GPU training
- numpy, tiktoken, datasets, tqdm and other packages required by the scripts
- Enough disk space for tokenized dataset shards and optimizer checkpoints
Install the project's dependencies in a virtual environment:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
For Windows/NVIDIA GPU training, verify that you have a CUDA-enabled PyTorch build using the official installer selector. An NVIDIA driver alone is insufficient if your installed PyTorch build is CPU-only.
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
Prepare the bounded experiment dataset
python -u .\fineweb_50m.py
This prepares tokenized training/validation shards in edu_fineweb_50M/. Dataset access requires an internet connection and may incur downloading/caching overhead. Before running training, verify the prepared shard counts and token totals.
Run the RTX 4050 experiment
These settings reproduce the configuration of the documented run; they do not guarantee identical floating-point results on another machine or with different data shards.
Remove-Item Env:NANOGPT_RESUME -ErrorAction SilentlyContinue
Remove-Item Env:NANOGPT_RUN_ID -ErrorAction SilentlyContinue
$env:PYTHONIOENCODING = "utf-8"
$env:NANOGPT_DATA_DIR = "edu_fineweb_50M"
$env:NANOGPT_BATCH = "1"
$env:NANOGPT_SEQ_LEN = "512"
$env:NANOGPT_TOKENS_PER_STEP = "16384"
$env:NANOGPT_MAX_STEPS = "3000"
$env:NANOGPT_VAL_INTERVAL = "250"
$env:NANOGPT_CKPT_INTERVAL = "250"
$env:NANOGPT_HELLA_INTERVAL = "3000"
$env:NANOGPT_HELLA_EXAMPLES = "128"
python -u .\train_gpt2_overnight.py
Results are written to a timestamped directory under log/ with metrics.csv, log.txt, and a resumable latest.pt checkpoint. Checkpoint files should be backed up outside the Git repository.
Resume an interrupted run
With the same dataset and environment settings as the original run:
$env:NANOGPT_RESUME = "log\<run_id>\latest.pt"
python -u .\train_gpt2_overnight.py
The checkpoint includes model and optimizer state, step information and data-loader position. Load only checkpoints you trust. Changing batch size, context, training horizon, or dataset across a resume is not supported by the run's safety checks.
Evaluate a saved checkpoint
python -u .\eval_hellaswag.py --checkpoint "log\<run_id>\latest.pt" --examples 128
Use --examples to control the evaluated subset. Always report the subset size along with its accuracy. The independent evaluator does not modify model weights.
Generate plots
python -m pip install matplotlib
python .\plot_training.py --metrics "log\<run_id>\metrics.csv"

Acknowledgments and references

- OpenAI — Language Models are Unsupervised Multitask Learners
- Hugging Face — FineWeb-Edu
- HellaSwag benchmark
Author
Sanchit Pareek · GitHub
