#!/usr/bin/env python3

"""
Visualize the performance of a model on a given environment.
"""
import argparse
import gym
import time
import numpy as np
import os

import babyai.utils as utils

from gym.wrappers import Monitor

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the trained model (REQUIRED or --demos-origin or --demos REQUIRED)")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 0 if model agent, 1 if demo agent)")
parser.add_argument("--argmax", action="store_true", default=True,
                    help="action with highest probability is selected for model agent")
parser.add_argument("--pause", type=float, default=0.1,
                    help="the pause between two consequent actions of an agent")
parser.add_argument("--instr-arch", type=str, default=None,
                    help="Instruction architecture used")
parser.add_argument("--test-episodes", type=int, default=5,
                    help="Number of episodes to evaluate over")
parser.add_argument("--test-instr-mode", type=str, default='all',
                    help="Type of instruction to test on. If `all` will run against all modes")
parser.add_argument("--test-comp-set", action="store_true", default=False,
                    help="Test on compositional held out set")
parser.add_argument("--num-dists", type=int, default=0,
                    help="Number of distractor objects")
parser.add_argument("--log-every", type=int, default=1,
                    help="How often to show agent performance on a test episode")
parser.add_argument("--exp", type=str, default="lang_understanding",
                    help="Which experiment to run: lang_understanding or paraphrases")

args = parser.parse_args()


def _check_log_this(i):
    return not i % args.log_every


assert args.model is not None, "--model must be specified."
if args.seed is None:
    args.seed = 0 if args.model is not None else 1

if args.test_instr_mode == 'all':
    if args.exp == 'lang_understanding':
        test_modes = ['original',
                      'false_permuted',
                      'false',
                      'text_inverted'
                      ]
    elif args.exp == "paraphrases":
        test_modes = ['original',
                      'paraphrased',
                      'paraphrased_permuted',
                      'semi_random_grammatical',
                      'random_grammatical',
                      'semi_random_agrammatical',
                      'random_agrammatical',
                      ]
    else:
        test_modes = ['original']
else:
    test_modes = [args.test_instr_mode]

test_mode_results = {test_mode: [] for test_mode in test_modes}
# Set seed for all randomness sources
utils.seed(args.seed)

if args.test_comp_set:
    pairs_dict = utils.train_test_splits(seed=args.seed)
else:
    pairs_dict = None

model_path = utils.get_model_dir(args.model)

for test_mode in test_modes:

    # Generate environment
    env = gym.make(args.env, pairs_dict=pairs_dict, test_instr_mode=test_mode, num_dists=args.num_dists)


    demo_path = os.path.join(model_path, test_mode)
    env = Monitor(env, demo_path, _check_log_this, force=True)
    env.seed(args.seed)

    # Define agent
    agent = utils.load_agent(env=env, model_name=args.model, argmax=args.argmax,
                             env_name=args.env, instr_arch=args.instr_arch)
    utils.seed(args.seed)

    print('\n')
    print(f'=== EVALUATING MODE: {test_mode} ===')

    # Run the agent
    done = False
    action = None
    obs = env.reset()

    step = 0
    episode_num = 0

    while episode_num < args.test_episodes:

        log_this = _check_log_this(episode_num)

        result = agent.act(obs)
        obs, reward, done, _ = env.step(result['action'])
        agent.analyze_feedback(reward, done)

        success_rate = reward > 0

        if log_this:
            time.sleep(args.pause)
            renderer = env.render("human")

        if done:
            if log_this:
                print(f'Mode: {test_mode} | Episode: {episode_num:,.0f} | Reward: {reward:.2f} | Success Rate: '
                      f'{success_rate:.2f} | Mission: {obs["mission"]}')

            test_mode_results[test_mode].append(reward)
            episode_num += 1
            env.seed(args.seed + episode_num)
            obs = env.reset()
            agent.on_reset()
            step = 0
        else:
            step += 1

    env.close()

print('\n')
print('=== Evaluation complete! ===')

utils.save_json(file=test_mode_results,
               path=model_path,
               filename="logs")

print('Test mode results:')
for test_mode, results in test_mode_results.items():
    success_rates = [1 if _ > 0 else 0 for _ in results]

    print(f'{test_mode} | Mean return: {np.mean(results):.3f} | Std return: {np.std(results):.3f} | Mean success rate: '
          f'{np.mean(success_rates):.3f} | Std success rate: {np.std(success_rates):.3f} |')
