import warnings
warnings.filterwarnings('ignore')
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
import torch
import numpy as np
import gym

gym.logger.set_level(40)
import time
import random
from pathlib import Path
from cfg import parse_cfg
from env import make_env
from algorithm.tdmpc import TDMPC
from algorithm.helper import Episode, ReplayBuffer, get_dataset_dict
import logger
from copy import deepcopy
import gc

torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__ = 'cfgs', 'logs'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(env, agent, num_episodes, step, env_step, video):
    """Evaluate a trained agent and optionally save a video."""
    episode_rewards = []
    episode_successes = []
    episode_lengths = []
    for i in range(num_episodes):
        obs, done, ep_reward, t = env.reset(), False, 0, 0
        ep_success = False
        if video: video.init(env, enabled=(i == 0))
        while not done:
            action = agent.act(obs, t0=t == 0, eval_mode=True, step=step)
            obs, reward, done, info = env.step(action.cpu().numpy())
            ep_reward += reward
            if 'success' in info and info['success']:
                ep_success = True
            if video: video.record(env)
            t += 1
        episode_rewards.append(float(ep_reward))
        episode_successes.append(float(ep_success))
        episode_lengths.append(t)
        if video: video.save(env_step)
    return {
        'episode_reward': np.nanmean(episode_rewards),
        'episode_success': np.nanmean(episode_successes),
        'episode_length': np.nanmean(episode_lengths)
    }


def train(cfg):
    """Training script for TD-MPC. Requires a CUDA-enabled device."""
    assert torch.cuda.is_available()
    set_seed(cfg.seed)
    work_dir = Path().cwd() / __LOGS__ / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
    print("Making env...")
    env = make_env(cfg)
    print("Instantiating agent...")
    agent = TDMPC(cfg)
    if cfg.pretrained_model_path:
        print(f"Loading pretrained model from `{cfg.pretrained_model_path}`...")
        agent.load(cfg.pretrained_model_path)
    print("Loading dataset...")
    dataset, reward_normalizer = get_dataset_dict(cfg, env, return_reward_normalizer=True)
    offline_buffer = ReplayBuffer(cfg, dataset=dataset)
    del dataset
    gc.collect()
    if cfg.balanced_sampling:
        buffer = ReplayBuffer(deepcopy(cfg))
    else:
        buffer = offline_buffer

    # Run training
    L = logger.Logger(work_dir, cfg)
    episode_idx, start_time = 0, time.time()

    step = 0
    last_log_step, last_save_step = 0, 0
    print("Training starts!")
    while step < cfg.train_steps:

        is_offline = True
        num_updates = cfg.episode_length
        _step = step + num_updates
        rollout_metrics = {}

        if step >= cfg.offline_steps:
            is_offline = False

            # Collect trajectory
            obs = env.reset()
            episode = Episode(cfg, obs)
            success = False
            while not episode.done:
                action = agent.act(obs, step=step, t0=episode.first)
                obs, reward, done, info = env.step(action.cpu().numpy())
                reward = reward_normalizer(reward)
                mask = 1.0 if (not done or "TimeLimit.truncated" in info) else 0.0
                success = info.get('success', False)
                episode += (obs, action, reward, done, mask, success)
            assert len(episode) <= cfg.episode_length
            buffer += episode
            episode_idx += 1
            rollout_metrics = {
                'episode_reward': episode.cumulative_reward,
                'episode_success': float(success),
                'episode_length': len(episode)
            }
            num_updates = len(episode) * cfg.utd
            _step = min(step + len(episode), cfg.train_steps)

        # Update model
        train_metrics = {}
        if is_offline:
            for i in range(num_updates):
                train_metrics.update(agent.update(offline_buffer, step + i))
        else:
            for i in range(num_updates):
                train_metrics.update(
                    agent.update(buffer, step + i // cfg.utd,
                                 demo_buffer=offline_buffer if cfg.balanced_sampling else None)
                )

        # Log training metrics
        env_step = int(_step * cfg.action_repeat)
        common_metrics = {
            'episode': episode_idx,
            'step': _step,
            'env_step': env_step,
            'total_time': time.time() - start_time,
            'is_offline': float(is_offline)
        }
        train_metrics.update(common_metrics)
        train_metrics.update(rollout_metrics)
        L.log(train_metrics, category='train')

        # Evaluate agent periodically
        if step == 0 or env_step - last_log_step >= cfg.eval_freq:
            eval_metrics = evaluate(env, agent, cfg.eval_episodes, step, env_step, L.video)
            if hasattr(env, "get_normalized_score"):
                eval_metrics['normalized_score'] = env.get_normalized_score(eval_metrics["episode_reward"]) * 100.0
            common_metrics.update(eval_metrics)
            L.log(common_metrics, category='eval')
            last_log_step = env_step - env_step % cfg.eval_freq

        # Save model periodically
        if cfg.save_model and env_step - last_save_step >= cfg.save_freq:
            L.save_model(agent, identifier=env_step)
            print(f"Model has been checkpointed at step {env_step}")
            last_save_step = env_step - env_step % cfg.save_freq

        if cfg.save_model and is_offline and _step >= cfg.offline_steps:
            # save the model after offline training
            L.save_model(agent, identifier="offline")

        step = _step

    L.finish(agent, buffer)
    print('Training completed successfully')


if __name__ == '__main__':
    train(parse_cfg(Path().cwd() / __CONFIG__))
