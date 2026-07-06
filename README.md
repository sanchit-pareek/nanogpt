# nanoGPT

This repo holds a from-scratch reproduction of [nanoGPT](https://github.com/karpathy/nanoGPT). The git commits are kept step by step and clean so that one can easily walk through the git commit history to see it built incrementally. There is an accompanying [video lecture on YouTube](https://youtu.be/l8pRSuU81PU) where each commit is introduced and explained.

We start from an empty file and work our way to a reproduction of the [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (124M) model. With more patience or compute, the code can also reproduce [GPT-3](https://arxiv.org/pdf/2005.14165) models. Reproducing GPT-2 (124M) today takes about ~1hr and ~$10 on a cloud GPU.

Note that GPT-2 and GPT-3 are simple language models trained on internet documents — they "dream" internet documents. This repo does not cover Chat finetuning.

## install

```
pip install torch numpy transformers datasets tiktoken wandb tqdm requests
```

## usage

### Data preparation

Download and tokenize the FineWeb-Edu dataset:

```sh
python fineweb.py
```

This downloads 10B tokens from FineWeb-Edu and saves shards to `edu_fineweb10B/`.

### Training

Train a GPT-2 (124M) model from scratch:

```sh
python train_gpt2.py
```

For distributed training across multiple GPUs:

```sh
torchrun --standalone --nproc_per_node=8 train_gpt2.py
```

### Evaluation

Evaluate on HellaSwag:

```sh
python hellaswag.py
```
## License

MIT
