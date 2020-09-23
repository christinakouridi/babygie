"""
Common arguments for BabyAI training scripts
"""

import os
import argparse
import numpy as np


class ArgumentParser(argparse.ArgumentParser):

    def __init__(self):
        super().__init__()

        # Base arguments
        self.add_argument("--env", default=None,
                            help="name of the environment to train on (REQUIRED)")
        self.add_argument("--model", default=None,
                            help="name of the model (default: ENV_ALGO_TIME)")
        self.add_argument("--pretrained-model", default=None,
                            help='If you\'re using a pre-trained model and want the fine-tuned one to have a new name')
        self.add_argument("--seed", type=int, default=1,
                            help="random seed; if 0, a random random seed will be used  (default: 1)")
        self.add_argument("--task-id-seed", action='store_true',
                            help="use the task id within a Slurm job array as the seed")  # TODO: delete if not used
        self.add_argument("--procs", type=int, default=64,
                            help="number of processes (default: 64)")
        self.add_argument("--wandb", action='store_true',
                          help="log metrics to weights and biases")
        self.add_argument("--project-name", type=str, default='new_proj',
                          help="wandb project name")
        self.add_argument("--job-id", type=int, default=0,
                          help="job id on NLP computer cluster")  # TODO: delete, temp for cluster training
        self.add_argument("--write-csv", action="store_true", default=False,
                          help="write model output to a csv file in a tabular format")
        self.add_argument("--savelog-missions", action="store_true", default=False,
                          help="whether to save and log seen missions strings (default: False)")

        # Training arguments
        self.add_argument("--log-interval", type=int, default=2,
                            help="number of updates between two logs (default: 10)")
        self.add_argument("--frames", type=int, default=int(150_000_000),
                            help="number of frames of training (default: 9e10)")
        self.add_argument("--patience", type=int, default=100,
                            help="patience for early stopping (default: 100)")
        self.add_argument("--epochs", type=int, default=3_000_000,
                            help="maximum number of epochs")
        self.add_argument("--epoch-length", type=int, default=0,
                            help="number of examples per epoch; the whole dataset is used by if 0")
        self.add_argument("--frames-per-proc", type=int, default=40,
                            help="number of frames per process before update (default: 40)")
        self.add_argument("--lr", type=float, default=1e-4,
                            help="learning rate (default: 1e-4)")
        self.add_argument("--beta1", type=float, default=0.9,
                            help="beta1 for Adam (default: 0.9)")
        self.add_argument("--beta2", type=float, default=0.999,
                            help="beta2 for Adam (default: 0.999)")
        self.add_argument("--recurrence", type=int, default=20,
                            help="number of timesteps gradient is backpropagated (default: 20)")
        self.add_argument("--optim-eps", type=float, default=1e-5,
                            help="Adam and RMSprop optimizer epsilon (default: 1e-5)")
        self.add_argument("--optim-alpha", type=float, default=0.99,
                            help="RMSprop optimizer apha (default: 0.99)")
        self.add_argument("--batch-size", type=int, default=1280,
                                help="batch size for PPO (default: 1280)")
        self.add_argument("--entropy-coef", type=float, default=0.01,
                            help="entropy term coefficient (default: 0.01)")

        # Model parameters
        self.add_argument("--image-dim", type=int, default=128,
                            help="dimensionality of the image embedding.  Defaults to 128 in residual architectures")
        self.add_argument("--memory-dim", type=int, default=128,
                            help="dimensionality of the memory LSTM")
        self.add_argument("--instr-dim", type=int, default=128,
                            help="dimensionality of the memory LSTM")
        self.add_argument("--no-instr", action="store_true", default=False,
                            help="don't use instructions in the model")
        self.add_argument("--instr-arch", default="gru",
                            help="arch to encode instructions, possible values: gru, bigru, conv, bow (default: gru)")
        self.add_argument("--no-mem", action="store_true", default=False,
                            help="don't use memory in the model")
        self.add_argument("--arch", default='bow_endpool_res',
                            help="image embedding architecture")

        # GIE parameters
        self.add_argument("--gie-aggr-method", default='root',
                          help="gie word embedding aggregation method, possible values: root, mean (default: mean)")
        self.add_argument("--gie-pretrained-emb", default='random',
                          help="whether to init. gie with pre-trained embeddings, possible values: random, fast_bert")
        self.add_argument("--gie-message-rounds", type=int, default=2,
                          help="number of message passing rounds. If 0 the maximum depth will be used (default:1)")
        self.add_argument("--gie-two-layers", action="store_true", default=False,
                          help="whether to use two deep layers in the graph (default: False)")
        self.add_argument("--gie-heads", type=int, default=1,
                          help="number of attention heads for gie gat (default:1)")
        self.add_argument("--gie-freeze-emb", action="store_true", default=False,
                          help="whether to freeze embeddings (default: False)")
        self.add_argument("--no-clip-value-loss", action="store_true", default=False,
                          help="don't clip the value loss (default: False)")

        # Validation / Test parameters
        self.add_argument("--test-env", default=None,
                          help="name of the test environment. If none, the train environment is used for validation")
        self.add_argument("--test-seed", type=int, default=int(1e9),
                            help="seed for environment used for validation (default: 1e9)")
        self.add_argument("--test-interval", type=int, default=1,
                            help="number of epochs between two validation checks (default: 1)")
        self.add_argument("--test-episodes", type=int, default=500,
                            help="number of episodes used to evaluate the agent, and to evaluate validation accuracy")
        self.add_argument("--monitor-gym", action="store_true", default=False,
                          help="whether to log videos of episodes on wandb")

    def parse_args(self):
        """
        Parse the arguments and perform some basic validation
        """

        args = super().parse_args()

        # Set seed for all randomness sources
        if args.seed == 0:
            args.seed = np.random.randint(10000)
        if args.task_id_seed:
            args.seed = int(os.environ['SLURM_ARRAY_TASK_ID'])
            print('set seed to {}'.format(args.seed))

        return args
