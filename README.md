# Overview

This repository contains experimental code and results that underpin my thesis for a MSc in CSML @ UCL (to be released upon marking).

In it, we study the potential benefits (sample efficiency and generalisation) of reasoning over syntactic representations of language instructions in the [`BabyAI`](https://github.com/mila-iqia/babyai) environment. We introduce a `BabyGIE` agent that uses a graph neural network to encode instruction representations from a syntax parser.

# Technical Introduction

## Installation

Most easily accomplished using the included `Dockerfile`.

If you choose to install without Docker, a few important notes to sequence correctly:

- In a fresh virtual env, install `torch` and `torchvision`
- Then follow the [torch-geometric setup instructions](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) (note the need to specify Torch and CUDA versions to set up correctly)
- `pip install -r requirements.txt` to install key remaining dependencies
- Finish by locally installing `gym-minigrid` (included in this repo) rather than the version hosted on pip -- there are a couple of breaking changes in the pypi version

## Code modifications

Our agent `BabyGIE` is built on top of the [`babyai`](https://github.com/mila-iqia/babyai) and [`gym-minigrid`](https://github.com/maximecb/gym-minigrid) environments with some key modifications:

- `babyai/gie`: contains code for our syntactic dependency parser, `BabyGIE`-specific levels we've developed, and code to generate level train-test splits
- `babyai/model.py`: adds definitions of our `GAT` and `GCN` models for graph-based language encoding. The main `ACModel` class is updated to incorporate these models as well as pre-trained word embedding initialisation (via HuggingFace's [Transformers](https://huggingface.co/transformers/) package).
- `train_rl.py`: integrates [wandb](https://wandb.ai/), our instruction pre-processor, and some level initialisation code to support our new testing methodology
- `babyai/utils/format.py`: introduces a new observation pre-processor (`GieInstructionsPreprocessor`) that returns the syntactically parsed instructions and returns the mission tokens (enabling use of re-trained word embeddings)
- `babyai/arguments.py`: adds GIE-specific arguments

# Running

## Key arguments

As documented and described in `arguments.py` (with some limited additional arguments in `scripts/train_rl.py`). Some key args to consider:

- `instr-arch`: defines the instruction processor to use, such as `gru`, `attgru`, `gie`, `gie_gat`
- `gie-aggr-method`: controls how instruction representations are aggregated — either `root` or `mean`
- `gie-pretrained-emb`: either `random` (randomly initialised embedding layer, no pre-training) or `fast_bert`
- `gie-message-rounds`: number of times to pass messages in the GNN
- `gie-freeze-emb`: whether to freeze embedding layer or allow backpro to adjust gradients

## Example launch

### Docker
```bash
$ cd babygie
$ sudo docker build . -t babygie-image
$ sudo docker run --gpus all -dit -P --name babygie -v <LOCAL_MODEL_DIR>:/models babygie-image
$ docker attach <PID>
$ python3 -m scripts.train_rl --env BabyAI-GoToObj-v0 --arch film_endpool_res --instr-arch gie --log-interval 2 --save-interval 20 --frames 300_000 --seed 1
```

### Locally
```bash
$ cd babygie
$ python -m scripts.train_rl --env BabyAI-GoToObj-v0 --arch film_endpool_res --instr-arch gie --log-interval 2 --save-interval 20 --frames 300_000 --seed 1
```

# Experimental results

See XXX for an analysis of agent trajectories and behaviours.

## Experiment 1
- `env` `BabyAI-GoToObj-v0`
- `instr-arch` repeated over [`gru`, `attgru`, `gie`, `gie_gat`]
- `seed` repeated over [1, 40, ...]

TK

## Experiment 2
TK