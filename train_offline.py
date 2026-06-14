import datetime
import os
import pickle
from typing import Tuple

import gym
import jax
import numpy as np
from tqdm import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter

import wrappers
from dataset_utils import D4RLDataset, reward_from_preference, reward_from_preference_transformer, split_into_trajectories
from evaluation import evaluate
from learner import Learner

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.40'

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'halfcheetah-expert-v2', 'Environment name.')
flags.DEFINE_string('save_dir', './logs/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('use_reward_model', False, 'Use reward model for relabeling reward.')
flags.DEFINE_string('model_type', 'MLP', 'type of reward model.')
flags.DEFINE_string('ckpt_dir',
                    './logs/pref_reward',
                    'ckpt path for reward model.')
flags.DEFINE_string('comment',
                    'base',
                    'comment for distinguishing experiments.')
flags.DEFINE_integer('seq_len', 25, 'sequence length for relabeling reward in Transformer.')
flags.DEFINE_bool('use_diff', False, 'boolean whether use difference in sequence for reward relabeling.')
flags.DEFINE_string('label_mode', 'last', 'mode for relabeling reward with tranformer.')
flags.DEFINE_boolean('save_policy_ckpt', False,
                     'Save IQL parameter checkpoints for later inference/fine-tuning.')
flags.DEFINE_string('policy_ckpt_dir', '',
                    'Directory where IQL actor checkpoints are saved.')
flags.DEFINE_integer('policy_ckpt_interval', 0,
                     'Save a policy checkpoint every N training steps. 0 means only final/latest saves.')

config_flags.DEFINE_config_file(
    'config',
    'default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def normalize(dataset, env_name, max_episode_steps=1000):
    trajs = split_into_trajectories(dataset.observations, dataset.actions,
                                    dataset.rewards, dataset.masks,
                                    dataset.dones_float,
                                    dataset.next_observations)
    trj_mapper = []
    for trj_idx, traj in tqdm(enumerate(trajs), total=len(trajs), desc="chunk trajectories"):
        traj_len = len(traj)

        for _ in range(traj_len):
            trj_mapper.append((trj_idx, traj_len))

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    sorted_trajs = sorted(trajs, key=compute_returns)
    min_return, max_return = compute_returns(sorted_trajs[0]), compute_returns(sorted_trajs[-1])

    normalized_rewards = []
    for i in range(dataset.size):
        _reward = dataset.rewards[i]
        if 'antmaze' in env_name:
            _, len_trj = trj_mapper[i]
            _reward -= min_return / len_trj
        _reward /= max_return - min_return
        # if ('halfcheetah' in env_name or 'walker2d' in env_name or 'hopper' in env_name):
        _reward *= max_episode_steps
        normalized_rewards.append(_reward)

    dataset.rewards = np.array(normalized_rewards)


def make_env_and_dataset(env_name: str,
                         seed: int) -> Tuple[gym.Env, D4RLDataset]:
    env = gym.make(env_name)

    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    dataset = D4RLDataset(env)

    if FLAGS.use_reward_model:
        reward_model = initialize_model()
        if FLAGS.model_type == "MR":
            dataset = reward_from_preference(FLAGS.env_name, dataset, reward_model, batch_size=FLAGS.batch_size)
        else:
            dataset = reward_from_preference_transformer(
                FLAGS.env_name,
                dataset,
                reward_model,
                batch_size=FLAGS.batch_size,
                seq_len=FLAGS.seq_len,
                use_diff=FLAGS.use_diff,
                label_mode=FLAGS.label_mode
            )
        del reward_model

    if FLAGS.use_reward_model:
        normalize(dataset, FLAGS.env_name, max_episode_steps=env.env.env._max_episode_steps)
        if 'antmaze' in FLAGS.env_name:
            dataset.rewards -= 1.0
        if ('halfcheetah' in FLAGS.env_name or 'walker2d' in FLAGS.env_name or 'hopper' in FLAGS.env_name):
            dataset.rewards += 0.5
    else:
        if 'antmaze' in FLAGS.env_name:
            dataset.rewards -= 1.0
            # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
            # but I found no difference between (x - 0.5) * 4 and x - 1.0
        elif ('halfcheetah' in FLAGS.env_name or 'walker2d' in FLAGS.env_name or 'hopper' in FLAGS.env_name):
            normalize(dataset, FLAGS.env_name, max_episode_steps=env.env.env._max_episode_steps)

    return env, dataset


def initialize_model():
    if os.path.exists(os.path.join(FLAGS.ckpt_dir, "best_model.pkl")):
        model_path = os.path.join(FLAGS.ckpt_dir, "best_model.pkl")
    else:
        model_path = os.path.join(FLAGS.ckpt_dir, "model.pkl")

    with open(model_path, "rb") as f:
        ckpt = pickle.load(f)
    reward_model = ckpt['reward_model']
    return reward_model


def save_policy_checkpoint(agent: Learner, step: int, tag: str = None):
    if not FLAGS.save_policy_ckpt:
        return
    if not FLAGS.policy_ckpt_dir:
        raise ValueError('--policy_ckpt_dir must be set when --save_policy_ckpt=True')

    os.makedirs(FLAGS.policy_ckpt_dir, exist_ok=True)
    name = tag if tag is not None else f'step_{step}'
    param_paths = {
        'actor': os.path.join(FLAGS.policy_ckpt_dir, f'actor_{name}.params'),
        'critic': os.path.join(FLAGS.policy_ckpt_dir, f'critic_{name}.params'),
        'value': os.path.join(FLAGS.policy_ckpt_dir, f'value_{name}.params'),
        'target_critic': os.path.join(FLAGS.policy_ckpt_dir, f'target_critic_{name}.params'),
    }
    latest_param_paths = {
        'actor': os.path.join(FLAGS.policy_ckpt_dir, 'actor_latest.params'),
        'critic': os.path.join(FLAGS.policy_ckpt_dir, 'critic_latest.params'),
        'value': os.path.join(FLAGS.policy_ckpt_dir, 'value_latest.params'),
        'target_critic': os.path.join(FLAGS.policy_ckpt_dir, 'target_critic_latest.params'),
    }
    full_ckpt_path = os.path.join(FLAGS.policy_ckpt_dir, f'iql_checkpoint_{name}.pkl')
    latest_full_ckpt_path = os.path.join(FLAGS.policy_ckpt_dir, 'iql_checkpoint_latest.pkl')

    metadata = {
        'step': int(step),
        'env_name': FLAGS.env_name,
        'seed': FLAGS.seed,
        'config': dict(FLAGS.config),
        'comment': FLAGS.comment,
        'use_reward_model': FLAGS.use_reward_model,
        'model_type': FLAGS.model_type,
        'reward_ckpt_dir': FLAGS.ckpt_dir,
        'param_paths': param_paths,
        'full_ckpt_path': full_ckpt_path,
    }

    agent.actor.save(param_paths['actor'])
    agent.actor.save(latest_param_paths['actor'])
    agent.critic.save(param_paths['critic'])
    agent.critic.save(latest_param_paths['critic'])
    agent.value.save(param_paths['value'])
    agent.value.save(latest_param_paths['value'])
    agent.target_critic.save(param_paths['target_critic'])
    agent.target_critic.save(latest_param_paths['target_critic'])

    full_checkpoint = {
        'metadata': metadata,
        'params': {
            'actor': jax.device_get(agent.actor.params),
            'critic': jax.device_get(agent.critic.params),
            'value': jax.device_get(agent.value.params),
            'target_critic': jax.device_get(agent.target_critic.params),
        },
        'opt_state': {
            'actor': jax.device_get(agent.actor.opt_state),
            'critic': jax.device_get(agent.critic.opt_state),
            'value': jax.device_get(agent.value.opt_state),
        },
        'rng': jax.device_get(agent.rng),
    }

    with open(full_ckpt_path, 'wb') as f:
        pickle.dump(full_checkpoint, f)
    with open(latest_full_ckpt_path, 'wb') as f:
        pickle.dump(full_checkpoint, f)
    with open(os.path.join(FLAGS.policy_ckpt_dir, f'metadata_{name}.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    with open(os.path.join(FLAGS.policy_ckpt_dir, 'metadata_latest.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    print(f'Saved IQL params checkpoint: {full_ckpt_path}', flush=True)


def main(_):
    save_dir = os.path.join(FLAGS.save_dir, 'tb',
                        FLAGS.env_name,
                            f"reward_{FLAGS.use_reward_model}_{FLAGS.model_type}" if FLAGS.use_reward_model else "original",
                            f"{FLAGS.comment}",
                            str(FLAGS.seed),
                            f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    summary_writer = SummaryWriter(save_dir,
                                   write_to_disk=True)
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    env, dataset = make_env_and_dataset(FLAGS.env_name, FLAGS.seed)

    kwargs = dict(FLAGS.config)
    agent = Learner(FLAGS.seed,
                    env.observation_space.sample()[np.newaxis],
                    env.action_space.sample()[np.newaxis],
                    max_steps=FLAGS.max_steps,
                    **kwargs)

    eval_returns = []
    for i in tqdm(range(1, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm):
        batch = dataset.sample(FLAGS.batch_size)
        update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                if v.ndim == 0:
                    summary_writer.add_scalar(f'training/{k}', v, i)
                else:
                    summary_writer.add_histogram(f'training/{k}', v, i)
            summary_writer.flush()

        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(agent, env, FLAGS.eval_episodes)

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f'evaluation/average_{k}s', v, i)
            summary_writer.flush()

            eval_returns.append((i, eval_stats['return']))
            np.savetxt(os.path.join(save_dir, 'progress.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])

            if FLAGS.policy_ckpt_interval > 0 and i % FLAGS.policy_ckpt_interval == 0:
                save_policy_checkpoint(agent, i)

    save_policy_checkpoint(agent, FLAGS.max_steps, tag='final')


if __name__ == '__main__':
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    app.run(main)
