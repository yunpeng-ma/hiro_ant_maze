import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import yaml
import sys
import torch
import random
import numpy as np
import gym
import datetime
import copy
from hiro.utils import listdirs, record_experience_to_csv, _is_update, Logger
from hiro.agent import HiroAgent


def run_evaluation(config, env, agent, fg):
    agent.load(episode=-1)

    rewards, success_rate = agent.evaluate_policy(env, fg, config["eval_episodes"], config["render"],
                                                  config["save_video"], config["sleep"])

    print('mean:{mean:.2f}, \
            std:{std:.2f}, \
            median:{median:.2f}, \
            success:{success:.2f}'.format(
        mean=np.mean(rewards),
        std=np.std(rewards),
        median=np.median(rewards),
        success=success_rate))


class Trainer():
    def __init__(self, config, fg, env, agent, experiment_name):
        self.config = config
        self.env = env
        self.agent = agent
        log_path = os.path.join(config["log_path"], experiment_name)
        self.logger = Logger(log_path=log_path)
        self.fg_eval = fg

    def train(self):
        global_step = 0

        for e in np.arange(self.config["num_episode"])+1:
            obs = self.env.reset()
            fg = np.random.uniform(env.observation_space.low, env.observation_space.high)[:-1] # sample a final goal
            s = obs
            done = False

            step = 0
            episode_reward = 0

            self.agent.set_final_goal(fg)

            while not done:
                # self.env.render()
                a, r, n_s, done = self.agent.step(s, self.env, step, global_step, explore=True)
                self.agent.append(step, s, a, n_s, r, done)
                losses, params = self.agent.train(global_step)
                self.log(global_step, [losses, params])
                # Updates
                s = n_s
                episode_reward += r
                step += 1
                global_step += 1
                self.agent.end_step()

            self.agent.end_episode(e, self.logger)
            self.logger.write('reward/Reward', episode_reward, e)
            self.evaluate(e)

    def log(self, global_step, data):
        losses, params = data[0], data[1]

        # Logs
        if global_step >= self.config["start_training_steps"] and _is_update(global_step, self.config["write_freq"]):
            for k, v in losses.items():
                self.logger.write('loss/%s'%(k), v, global_step)

            for k, v in params.items():
                self.logger.write('params/%s'%(k), v, global_step)

    def evaluate(self, e):
        # Print
        if _is_update(e, config["print_freq"]):
            # agent = copy.deepcopy(self.agent)
            # rewards, success_rate = agent.evaluate_policy(self.env)
            rewards, success_rate = self.agent.evaluate_policy(self.env, self.fg_eval, eval_episodes=1, render=True)
            self.logger.write('Success Rate', success_rate, e)

            print('episode:{episode:05d}, mean:{mean:.2f}, std:{std:.2f}, median:{median:.2f}, success:{success:.2f}'.format(
                episode=e,
                mean=np.mean(rewards),
                std=np.std(rewards),
                median=np.median(rewards),
                success=success_rate))


if __name__ == "__main__":
    config = yaml.safe_load(open(sys.argv[1]))
    # Select or Generate a name for this experiment
    if config["exp_name"]:
        experiment_name = config["exp_name"]
    else:
        if config["eval"]:
            # choose most updated experiment for evaluation
            dirs_str = listdirs(config["model_path"])
            dirs = np.array(list(map(int, dirs_str)))
            experiment_name = dirs_str[np.argmax(dirs)]
        else:
            experiment_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    print(experiment_name)

    env = gym.make(config["env_name"])
    # environment setting
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    env.seed(config['seed'])
    env.action_space.np_random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    low_action_space = env.action_space
    goal_dim = 120
    subgoal_dim = 120

    agent = HiroAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        goal_dim=goal_dim,
        subgoal_dim=subgoal_dim,
        hidden_dim=config["hidden_size"],
        low_action_space=low_action_space,
        model_path=config["model_path"],
        auto_alpha=config["automatic_entropy_tuning"],
        actor_lr=config["actor_lr"],
        critic_lr=config["critic_lr"],
        start_training_steps=config["start_training_steps"],
        model_save_freq=config["model_save_freq"],
        buffer_size=config["replay_size"],
        batch_size=config["batch_size"],
        buffer_freq=config["buffer_freq"],
        train_freq=config["train_freq"])

    fg_eval = np.r_[[1.265]*60, [1.0] * 60]

    # Run training or evaluation
    if not config["eval"]:
        # Record this experiment with arguments to a CSV file
        record_experience_to_csv(config, experiment_name)
        # Start training
        trainer = Trainer(config, fg_eval, env, agent, experiment_name)
        trainer.train()
    if config["eval"]:
        run_evaluation(config, env, agent, fg_eval)
