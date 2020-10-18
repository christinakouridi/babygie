# Overview

This repository contains experimental code and results that underpin my thesis for a MSc in CSML @ UCL (to be released upon marking).

In it, we study the potential benefits (sample efficiency and generalisation) of reasoning over syntactic representations of language instructions in the [`BabyAI`](https://github.com/mila-iqia/babyai) environment. We introduce a `BabyGIE` agent that uses a graph neural network (`GCN` or `GAT`) to encode instruction representations from a natural language dependency parser.

# Technical Introduction

## Installation
There are few important notes to sequence environment setup correctly:

- In a fresh virtual env, install Python 3.8
- Install `torch==1.5.1` and `torchvision==0.6.1`
- Then follow the [torch-geometric setup instructions](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) (note the need to specify Torch and CUDA versions to set up correctly, e.g. `CUDA=cu102 and TORCH=1.5.0`)
- `pip install -r requirements.txt` to install key remaining dependencies
- Finish by locally installing `gym-minigrid` (included in this repo) rather than the version hosted on pip -- there are a couple of breaking changes in the pypi version

## Code modifications

Our agent `BabyGIE` is built on top of the [`babyai`](https://github.com/mila-iqia/babyai) and [`gym-minigrid`](https://github.com/maximecb/gym-minigrid) environments with some key modifications:

- `babyai/gie`: contains code for our syntactic dependency parser, `BabyGIE`-specific levels we've developed, and code to generate level train-test splits
- `babyai/model.py`: adds definitions of our `GAT` and `GCN` models for graph-based language encoding. The main `ACModel` class is updated to incorporate these models as well as pre-trained word embedding initialisation (via HuggingFace's [Transformers](https://huggingface.co/transformers/) package).
- `scripts/train_rl.py`: integrates [wandb](https://wandb.ai/), our instruction pre-processor, and some level initialisation code to support our new testing methodology
- `babyai/utils/format.py`: introduces a new observation pre-processor (`GieInstructionsPreprocessor`) that returns the syntactically parsed instructions and returns the mission tokens (enabling use of re-trained word embeddings)
- `scripts/arguments.py`: adds GIE-specific arguments
- `babyai/rl/algos/ppo.py`: adds a clipping parameter for the value loss; previously the clipping parameter for the surrogate loss function was used
- `scripts/gie_evaluate.py`: enables visualisation of trained agents tested on typical BabyAI instructions, a custom instruction set to assess language understanding, and paraphrased instructions (note the latter should only be used with models trained with BERT (fixed or fine-tuned), otherwise the learnt embedding layer will assign a random embedding to out-of-vocabulary words in our paraphrased instructions, which can in turn result in an adversarial attack on the policy)
- `babyai/levels/verifier.py`: adds custom sets of instructions to assess the language understanding of trained agents, and how they can generalise to syntactically-diverse, paraphrased instructions
- `gym-minigrid/gym_minigrid/roomgrid.py`: adds function to populate grid with instruction objects and distractors from our custom compositional train-test splits

# Running

## Key arguments

As documented and described in `arguments.py` (with some limited additional arguments in `scripts/train_rl.py`). Some key args to consider:

General training flags:

- `env`: name of the environment to train on, such as `BabyAI-GoToObj-v0` (sample efficiency), `BabyAI-GoToObj_c-v0` (compositional generalisation), `BabyAI-PutNextLocal_d0_e` (sample efficiency), `BabyAI-PutNextLocal_d0_c` (compositional generalisation)
- `instr-arch`: defines the instruction processor to use, such as `gru`, `attgru`, `gie_gcn`, `gie_gat` [default:`gru`]
- `arch`: defines the image enbedding architecture, such as `film_endpool_res` (BabyAI 1.1), `film` (BabyAI 1.0), `cnn` [default:`film_endpool_res`]
- `lr`: step size of Adam optimiser [default: `1e-4`]
- `max-grad-norm`: maximum norm of the global loss function's gradient [default: `0.5`]
- `frames`: maximum number of training steps across actors [default: `100_000_000`]
- `seed`: seed controlling randomness in the training procedure (including train/test set split for the compositional generalisation experiment) [default: `1`]
- `test-seed`: seed for environment used for validation; this is incremented for each instruction / episode in the test batch [default: `1e9`]
- `test-episodes`: number of episodes to test on [default: `500`]
- `log-interval`: how often to output training metrics [default: `2`]
- `save-interval`: number of updates between saving and evaluating the trained model on test instructions. Note for `envs` ending in `_c`, compositional generalisation will be automatically evaluated; for normal BabyAI levels and envs ending in `_e`, sample efficiency will be evaluated [default: 10]
- `clip-eps-value`: clipping coefficient for value loss [default: `0.2`]

GIE-specific flags:
- `gie-aggr-method`: controls how instruction representations are aggregated — either `root`, `mean` or `max` [default: `root`]
- `gie-pretrained-emb`: either `random` (randomly initialised embedding layer, no pre-training), `tiny_bert` (128-dimensional embeddings), `distil_bert` & `bert` (768-dimensional embeddings but further projected to 128-dimensional representations via a linear layer) [default: `random`]
- `gie-message-rounds`: number of times to pass messages in the GNN [default: `2`]
- `gie-freeze-emb`: whether to freeze embedding layer or allow backpropagation to adjust gradients. It is recommended to enable this flag with bert [default: `False`]

## Example launch

```bash
$ cd babygie
$ python -m scripts.train_rl --env BabyAI-GoToObj-v0 --instr-arch gie_gcn --frames 300_000
```

# Experimental results

See [our Notion page](https://www.notion.so/Agent-Analysis-678a4693229542868f2d526e132df4cd) for a brief analysis of agent trajectories and behaviours.

## Experiment 1
Baseline agent with gru or attention-gru instruction encoder:

- `env` repeated over [`BabyAI-GoToObj-v0`, `BabyAI-GoToLocal-v0`, `BabyAI-PutNextLocal_d0_e-v0`, `BabyAI-PutNextLocal_d2_e-v0`, `BabyAI-PutNextLocal_d4_e-v0`, `BabyAI-PutNextLocal-v0`]
- `frames` repeated over for each env respectively [300_000, 25_000_000, 7_000_000, 30_000_000, 80_000_000, 100_000_000] 
- `instr-arch` repeated over [`gru`, `attgru`]
- `batch-size` `1280`
- `seed` repeated over for each env [`1`, `40`, `365`, `961`]

babyGIE agent with GCN or GAT instruction encoder:
- `env` repeated over [`BabyAI-GoToObj-v0`, `BabyAI-GoToLocal-v0`, `BabyAI-PutNextLocal_d0_e-v0`, `BabyAI-PutNextLocal_d2_e-v0`, `BabyAI-PutNextLocal_d4_e-v0`, `BabyAI-PutNextLocal-v0`]
- `frames` repeated over for each env respectively [300_000, 25_000_000, 7_000_000, 30_000_000, 80_000_000, 100_000_000] 
- `instr-arch` repeated over [`gie_gcn`, `gie_gat`]
- `seed` repeated over [`1`, `40`, `365`, `961`, ...]
- `lr` `5e-5`
- `clip-eps-value` `0.0`

## Experiment 2
Baseline agent with gru or attention-gru instruction encoder:

- `env` repeated over [`BabyAI-GoToObj_c-v0`, `BabyAI-GoToLocal_c-v0`, `BabyAI-PutNextLocal_d0_c-v0`, `BabyAI-PutNextLocal_d2_c-v0`, `BabyAI-PutNextLocal_d4_c-v0`, `BabyAI-PutNextLocal_c-v0`]
- `frames` repeated over for each env respectively [300_000, 25_000_000, 7_000_000, 30_000_000, 80_000_000, 100_000_000] 
- `instr-arch` repeated over [`gru`, `attgru`]
- `batch-size` `1280`
- `seed` repeated over for each env [`1`, `40`, `365`, `961`]

babyGIE agent with GCN or GAT instruction encoder:
- `env` repeated over [`BabyAI-GoToObj_c-v0`, `BabyAI-GoToLocal_c-v0`, `BabyAI-PutNextLocal_d0_c-v0`, `BabyAI-PutNextLocal_d2_c-v0`, `BabyAI-PutNextLocal_d4_c-v0`, `BabyAI-PutNextLocal_c-v0`]
- `frames` repeated over for each env respectively [300_000, 25_000_000, 7_000_000, 30_000_000, 80_000_000, 100_000_000] 
- `instr-arch` repeated over [`gie_gcn`, `gie_gat`]
- `seed` repeated over [`1`, `40`, `365`, `961`, ...]
- `lr` `5e-5`
- `clip-eps-value` `0.0`

## Experiment 3
Baseline agent with gru or attention-gru instruction encoder:

- `env` repeated over [`BabyAI-GoToObj-v0`, `BabyAI-GoToLocal-v0`, `BabyAI-PutNextLocal_d0_e-v0`, `BabyAI-PutNextLocal_d2_e-v0`, `BabyAI-PutNextLocal_d4_e-v0`, `BabyAI-PutNextLocal-v0`]
- `frames` repeated over for each env respectively [300_000, 25_000_000, 7_000_000, 30_000_000, 80_000_000, 100_000_000] 
- `instr-arch` repeated over [`gru_bert`, `attgru_bert`]
- `batch-size` `1280`
- `seed` repeated over for each env [`1`, `40`, `365`, `961`]

babyGIE agent with GCN or GAT instruction encoder:
- `env` repeated over [`BabyAI-GoToObj-v0`, `BabyAI-GoToLocal-v0`, `BabyAI-PutNextLocal_d0_e-v0`, `BabyAI-PutNextLocal_d2_e-v0`, `BabyAI-PutNextLocal_d4_e-v0`, `BabyAI-PutNextLocal-v0`]
- `frames` repeated over for each env respectively [300_000, 25_000_000, 7_000_000, 30_000_000, 80_000_000, 100_000_000] 
- `instr-arch` repeated over [`gie_gcn`, `gie_gat`]
- `gie-pretrained-emb` `tiny_bert`
- `gie-freeze-emb`
- `seed` repeated over [`1`, `40`, `365`, `961`, ...]
- `lr` `5e-5`
- `clip-eps-value` `0.0`


## Visualise agent trajectories
Original instructions (support for any level):
- `model` e.g. `BabyAI-PutNextLocal_d2_e-v0_ppo_film_endpool_res_gie_gcn_mem_seed40_20-09-19-23-12-27_best`
- `env`  e.g. `BabyAI-PutNextLocal_d2_e-v0`
- `instr-arch` should be the same as the trained model! i.e. `gie_gcn`
- `test-episodes` `10`
- `log-every` `1`
- `seed` e.g. `40`
- `exp` `original`

Language understanding instruction sets (only support for `PutNextLocal` tasks):
- `model` e.g. `BabyAI-PutNextLocal_d2_c-v0_ppo_film_endpool_res_gie_gcn_mem_seed40_20-09-19-23-12-27_best`
- `env`  e.g. `BabyAI-PutNextLocal_n-v0`
- `instr-arch` should be the same as the trained model! i.e. `gie_gcn`
- `exp` `lang_understanding`
- `num-dists` number of distractors e.g. `2`
- `test-episodes` `10`
- `log-every` `1`
- `seed` e.g. `40`
- `test-comp-set` in order to test on the compositional set, created using the seed above

Paraphrased instructions (only for `BERT` models and `PutNextLocal` tasks):
- `model` e.g. `BabyAI-PutNextLocal_d2_e-v0_ppo_film_endpool_res_gru_bert_mem_seed40_20-09-19-23-12-27_best`
- `env`  e.g. `BabyAI-PutNextLocal_n-v0`
- `instr-arch` should be the same as the trained model!  i.e.`gru_bert`
- `exp` `paraphrases`
- `num-dists` number of distractors e.g. `2`
- `test-episodes` `10`
- `log-every` `1`
- `seed` e.g. `40`
