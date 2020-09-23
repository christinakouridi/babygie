#!/usr/bin/env python3

"""
Script to train the agent through reinforcment learning.
"""

import os
import logging
import csv
import json
import gym
import time
import datetime
import torch
import numpy as np
import subprocess

import babyai
import babyai.utils as utils
import babyai.rl
from babyai.arguments import ArgumentParser
from babyai.model import ACModel
from babyai.evaluate import batch_evaluate
from babyai.utils.agent import ModelAgent
from gym_minigrid.wrappers import RGBImgPartialObsWrapper

from collections import deque
import wandb


# Parse arguments
parser = ArgumentParser()
parser.add_argument("--algo", default='ppo',
                    help="algorithm to use (default: ppo)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--reward-scale", type=float, default=20.,
                    help="Reward scale multiplier")
parser.add_argument("--gae-lambda", type=float, default=0.99,
                    help="lambda coefficient in GAE formula (default: 0.99, 1 means no gae)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--clip-eps-policy", type=float, default=0.2,
                    help="clipping epsilon for policy loss in PPO (default: 0.2)")
parser.add_argument("--clip-eps-value", type=float, default=0.2,
                    help="clipping epsilon for value loss in  PPO (default: 0.2)")
parser.add_argument("--ppo-epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--save-interval", type=int, default=50,
                    help="number of updates between two saves (default: 50, 0 means no saving)")
args = parser.parse_args()

utils.seed(args.seed)

# Define model name
suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
instr = args.instr_arch if args.instr_arch else "noinstr"
mem = "mem" if not args.no_mem else "nomem"

model_name_parts = {'env': args.env, 'algo': args.algo, 'arch': args.arch, 'instr': instr, 'mem': mem,
                    'seed': args.seed, 'info': '', 'coef': '', 'suffix': suffix}
default_model_name = "{env}_{algo}_{arch}_{instr}_{mem}_seed{seed}{info}{coef}_{suffix}".format(**model_name_parts)

if args.pretrained_model:
    default_model_name = args.pretrained_model + '_pretrained_' + default_model_name
args.model = args.model.format(**model_name_parts) if args.model else default_model_name

utils.configure_logging(args.model)
logger = logging.getLogger(__name__)

# Generate train-test set
if '_c' in args.env:
    model_path = utils.get_model_dir(default_model_name)
    pairs_dict = utils.train_test_splits(seed=args.seed)
    utils.save_json(file=pairs_dict, path=model_path, filename='all_pairs')
else:
    model_path = None
    pairs_dict = None

# Generate environments
envs = []
use_pixel = 'pixel' in args.arch
for i in range(args.procs):
    if '_c' in args.env:
        env = gym.make(args.env, pairs_dict=pairs_dict)
    else:
        env = gym.make(args.env)
    if use_pixel:
        env = RGBImgPartialObsWrapper(env)
    env.seed(100 * args.seed + i)
    envs.append(env)

ReturnsBuffer = deque(maxlen=10_000)

# Define obss preprocessor
if 'emb' in args.arch:
    obss_preprocessor = utils.IntObssPreprocessor(args.model, envs[0].observation_space, args.pretrained_model)
else:
    obss_preprocessor = utils.ObssPreprocessor(instr_arch=args.instr_arch, model_name=args.model,
                                               obs_space=envs[0].observation_space,
                                               load_vocab_from=args.pretrained_model)

# Define actor-critic model
acmodel = utils.load_model(args.model, raise_not_found=False)
if acmodel is None:
    if args.pretrained_model:
        acmodel = utils.load_model(args.pretrained_model, raise_not_found=True)
    elif 'gie' in args.instr_arch:
        acmodel = ACModel(obss_preprocessor.obs_space, envs[0].action_space,
                          args.image_dim, args.memory_dim, args.instr_dim,
                          not args.no_instr, args.instr_arch, not args.no_mem, args.arch,
                          gie_pretrained_emb=args.gie_pretrained_emb, gie_freeze_emb=args.gie_freeze_emb,
                          gie_aggr_method=args.gie_aggr_method, gie_message_rounds=args.gie_message_rounds,
                          gie_two_layers=args.gie_two_layers, gie_heads=args.gie_heads)
    else:
        acmodel = ACModel(obss_preprocessor.obs_space, envs[0].action_space,
                          args.image_dim, args.memory_dim, args.instr_dim,
                          not args.no_instr, args.instr_arch, not args.no_mem, args.arch)

obss_preprocessor.vocab.save()
utils.save_model(acmodel, args.model)

if torch.cuda.is_available():
    acmodel.cuda()

# Define actor-critic algo
reshape_reward = lambda _0, _1, reward, _2: args.reward_scale * reward

if args.algo == "ppo":
    algo = babyai.rl.PPOAlgo(envs, acmodel, args.frames_per_proc, args.discount, args.lr, args.beta1, args.beta2,
                             args.gae_lambda,
                             args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                             args.optim_eps, args.clip_eps_policy, args.clip_eps_value, args.ppo_epochs,
                             args.batch_size, obss_preprocessor, reshape_reward, savelog_missions=args.savelog_missions,
                             no_clip_value_loss=args.no_clip_value_loss)
else:
    raise ValueError("Incorrect algorithm name: {}".format(args.algo))

# When using extra binary information, more tensors (model params) are initialized compared to when we don't use that.
# Thus, there starts to be a difference in the random state. If we want to avoid it, in order to make sure that
# the results of supervised-loss-coef=0. and extra-binary-info=0 match, we need to reseed here.

utils.seed(args.seed)

# Restore training status

status_path = os.path.join(utils.get_log_dir(args.model), 'status.json')
if os.path.exists(status_path):
    with open(status_path, 'r') as src:
        status = json.load(src)
else:
    status = {'i': 0,
              'num_episodes': 0,
              'num_frames': 0}

# Define logger and Tensorboard writer and CSV writer

header = (["update", "episodes", "frames", "FPS", "duration"]
          + ["return_" + stat for stat in ['mean', 'std', 'min', 'max']]
          + ["success_rate"]
          + ["num_frames_" + stat for stat in ['mean', 'std', 'min', 'max']]
          + ["entropy", "value", "policy_loss", "value_loss", "loss", "grad_norm"])


if args.write_csv:
    csv_path = os.path.join(utils.get_log_dir(args.model), 'log.csv')
    first_created = not os.path.exists(csv_path)
    # we don't buffer data going in the csv log, cause we assume
    # that one update will take much longer that one write to the log
    csv_writer = csv.writer(open(csv_path, 'a', 1))
    if first_created:
        csv_writer.writerow(header)

# Log code state, command, availability of CUDA and model

babyai_code = list(babyai.__path__)[0]
# try:
#     last_commit = subprocess.check_output(
#         'cd {}; git log -n1'.format(babyai_code), shell=True).decode('utf-8')
#     logger.info('LAST COMMIT INFO:')
#     logger.info(last_commit)
# except subprocess.CalledProcessError:
#     logger.info('Could not figure out the last commit')
# try:
#     diff = subprocess.check_output(
#         'cd {}; git diff'.format(babyai_code), shell=True).decode('utf-8')
#     if diff:
#         logger.info('GIT DIFF:')
#         logger.info(diff)
# except subprocess.CalledProcessError:
#     logger.info('Could not figure out the last commit')
logger.info('COMMAND LINE ARGS:')
logger.info(args)
logger.info("CUDA available: {}".format(torch.cuda.is_available()))
logger.info(acmodel)

# wandb group name
if "gie" in args.instr_arch:
    group = args.instr_arch + '_' + \
            args.arch + '_' + \
            args.gie_pretrained_emb + '_' + \
            str(args.lr) + '_' + \
            str(args.max_grad_norm) + '_' + \
            str(args.batch_size) + '_' + \
            ('F' if args.gie_freeze_emb else 'NF') + '_' + \
            ('' if 'att' in args.instr_arch else args.gie_aggr_method + '_') + \
            str(args.gie_message_rounds) + 'M' + '_' + \
            ('' if args.no_clip_value_loss else str(args.clip_eps_value) + '_') + \
            ('' if args.gie_two_layers == False else '2L' + '_') + \
            ('' if args.gie_heads == 1 else str(args.gie_heads) + 'H' + '_')
else:
    group = args.instr_arch + '_' + args.arch


if args.wandb:
    wandb.init(project=args.project_name,
               name=default_model_name,
               reinit=True,
               monitor_gym=args.monitor_gym,
               config={'group': group})
    wandb.config.update(args)
    wandb.watch(acmodel)

# Train model

total_start_time = time.time()
best_success_rate = 0
best_mean_return = 0
test_env_name = args.test_env or args.env

while status['num_frames'] < args.frames:
    # Update parameters

    update_start_time = time.time()
    logs = algo.update_parameters
    update_end_time = time.time()

    if args.savelog_missions:
        utils.save_txt(file=logs['seen_missions'], path=utils.get_model_dir(default_model_name),
                       filename=f"train_missions")
        logger.info('Training missions: %s', logs["seen_missions"])

    status['num_frames'] += logs["num_frames"]
    status['num_episodes'] += logs['episodes_done']
    status['i'] += 1

    # Print logs

    if status['i'] % args.log_interval == 0:
        total_ellapsed_time = int(time.time() - total_start_time)
        fps = logs["num_frames"] / (update_end_time - update_start_time)
        duration = datetime.timedelta(seconds=total_ellapsed_time)
        return_per_episode = utils.synthesize(logs["return_per_episode"])
        success_per_episode = utils.synthesize(
            [1 if r > 0 else 0 for r in logs["return_per_episode"]])
        num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

        ReturnsBuffer.extend(logs["return_per_episode"])
        rolling_episode_success_rate_mean = (np.stack(ReturnsBuffer) > 0.0).mean()
        rolling_episode_success_rate_std = (np.stack(ReturnsBuffer) > 0.0).std()

        data = [status['i'], status['num_episodes'], status['num_frames'],
                fps, total_ellapsed_time,
                *return_per_episode.values(),
                *success_per_episode.values(),
                rolling_episode_success_rate_mean,
                rolling_episode_success_rate_std,
                *num_frames_per_episode.values(),
                logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"],
                logs["loss"], logs["grad_norm"]]

        format_str = ("U {} | E {} | F {:06} | FPS {:04.0f} | D {} | R:xsmM {: .2f} {: .2f} {: .2f} {: .2f} | "
                      "S:xsmM {: .3f} {: .3f} {: .3f} {: .3f} | RS:xs {: .3f} {: .3f} |  F:xsmM {:.1f} {:.1f} {} {} | "
                      "H {:.3f} | V {:.3f} | pL {: .3f} | vL {:.3f} | L {:.3f} | gN {:.3f} | ")

        logger.info(format_str.format(*data))
        if args.wandb:
            wandb.log({'train_mean_episode_return': return_per_episode['mean'],
                       'episode_count': ['num_episodes']}, step=status['num_frames'])
            wandb.log({'train_mean_success_rate': success_per_episode['mean'],
                       'episode_count': ['num_episodes']}, step=status['num_frames'])
            wandb.log({'rolling_episode_success_rate': rolling_episode_success_rate_mean,
                       'episode_count': ['num_episodes']}, step=status['num_frames']),
            wandb.log({'total_loss': logs["loss"],
                       'episode_count': status['num_episodes']}, step=status['num_frames'])
            wandb.log({'grad_norm': logs["grad_norm"],
                       'episode_count': status['num_episodes']}, step=status['num_frames'])

        if args.write_csv:
            csv_writer.writerow(data)

    # Save obss preprocessor vocabulary and model

    if args.save_interval > 0 and status['i'] % args.save_interval == 0:
        obss_preprocessor.vocab.save()

        with open(status_path, 'w') as dst:
            json.dump(status, dst)
            utils.save_model(acmodel, args.model)

        #################################################################
        # Testing the model before saving
        agent = ModelAgent(args.model, obss_preprocessor, argmax=True)
        agent.model = acmodel
        agent.model.eval()

        monitor_gym_this_time = args.monitor_gym and status['i'] > 0 and status['i'] % (args.save_interval * 5) == 0
        logs = batch_evaluate(agent, test_env_name, args.test_seed, args.test_episodes, pixel=use_pixel,
                              monitor_gym=monitor_gym_this_time, pairs_dict=pairs_dict, model_path=model_path)

        # save and log test mission strings
        if args.savelog_missions:
            logger.info('Testing missions: %s', logs["seen_missions"])
            if status['i'] % 1 == 0:
                utils.save_txt(file=logs['seen_missions'],
                               path=utils.get_model_dir(default_model_name),
                               filename=f"test_missions_{status['i']}")

        agent.model.train()

        # metrics
        mean_return = np.mean(logs["return_per_episode"])
        std_return = np.std(logs["return_per_episode"])
        mean_success_rate = np.mean([1 if r > 0 else 0 for r in logs['return_per_episode']])
        std_success_rate = np.std([1 if r > 0 else 0 for r in logs['return_per_episode']])

        if mean_success_rate > best_success_rate:
            best_success_rate = mean_success_rate
            save_model = True
        elif (mean_success_rate == best_success_rate) and (mean_return > best_mean_return):
            best_mean_return = mean_return
            save_model = True
        else:
            save_model = False

        if save_model:
            utils.save_model(acmodel, args.model + '_best')
            obss_preprocessor.vocab.save(utils.get_vocab_path(args.model + '_best'))

            # TODO: save embedding layer but link to vocab
            # utils.save_embeddings(acmodel, args.model + '_best')

            logger.info("Return {: .2f}; best model is saved".format(mean_return))
            logger.info("Return Std {: .2f}; best model is saved".format(std_return))
        else:
            logger.info("Return {: .2f}; not the best model; not saved".format(mean_return))
            logger.info("Return Std {: .2f}; not the best model is saved".format(std_return))

        logger.info("SR mean {: .2f}".format(mean_success_rate))
        logger.info("SR std {: .8f}".format(std_success_rate))
        logger.info("BSR {: .2f}".format(best_success_rate))
        logger.info("episodes {: .8f}".format(status['num_episodes']))
        logger.info("frames {: .8f}".format(status['num_frames']))

        if args.wandb:
            wandb.log({'test_mean_return': mean_return,
                       'episode_count': status['num_episodes']}, step=status['num_frames'])
            wandb.log({'test_mean_success_rate': mean_success_rate,
                       'episode_count': status['num_episodes']}, step=status['num_frames'])