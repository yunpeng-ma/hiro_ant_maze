import numpy as np
import gym
import time
from .controller import LowerController, HigherController
from .replay_buffer import LowReplayBuffer, HighReplayBuffer
from .utils import _is_update, Subgoal


class Agent():
    def __init__(self):
        pass

    def set_final_goal(self, fg):
        self.fg = fg

    def step(self, s, env, step, global_step=0, explore=False):
        raise NotImplementedError

    def append(self, step, s, a, n_s, r, d):
        raise NotImplementedError

    def train(self, global_step):
        raise NotImplementedError

    def end_step(self):
        raise NotImplementedError

    def end_episode(self, episode, logger=None):
        raise NotImplementedError

    def evaluate_policy(self, env, eval_episodes=10, render=False, save_video=False, sleep=-1):
        if save_video:
            # from OpenGL import GL
            env = gym.wrappers.Monitor(env, directory='video',
                                       write_upon_reset=True, force=True, resume=True, mode='evaluation')
            render = False

        success = 0
        rewards = []
        env.evaluate = True
        for e in range(eval_episodes):
            obs = env.reset()
            fg = obs['desired_goal']
            s = obs['observation']
            done = False
            reward_episode_sum = 0
            step = 0

            self.set_final_goal(fg)

            while not done:
                if render:
                    env.render()
                if sleep>0:
                    time.sleep(sleep)

                a, r, n_s, done = self.step(s, env, step)
                reward_episode_sum += r

                s = n_s
                step += 1
                self.end_step()
            else:
                error = np.sqrt(np.sum(np.square(fg-s[:2])))
                print('Goal, Curr: (%02.2f, %02.2f, %02.2f, %02.2f)     Error:%.2f'%(fg[0], fg[1], s[0], s[1], error))
                rewards.append(reward_episode_sum)
                success += 1 if error <=5 else 0
                self.end_episode(e)

        env.evaluate = False
        return np.array(rewards), success/eval_episodes


class HiroAgent(Agent):
    def __init__(
            self,
            state_dim,
            action_dim,
            goal_dim,
            subgoal_dim,
            hidden_dim,
            low_action_space,
            model_path,
            auto_alpha,
            actor_lr,
            critic_lr,
            start_training_steps,
            model_save_freq,
            buffer_size,
            batch_size,
            buffer_freq,
            train_freq):
        super().__init__()
        self.subgoal = Subgoal(subgoal_dim)
        high_action_space = self.subgoal.action_space

        self.model_save_freq = model_save_freq

        self.high_con = HigherController(
            state_dim=state_dim,
            goal_dim=goal_dim,
            action_dim=subgoal_dim,
            hidden_dim=hidden_dim,
            action_space=high_action_space,
            auto_alpha=auto_alpha,
            model_path=model_path,
            actor_lr=actor_lr,
            critic_lr=critic_lr)

        self.low_con = LowerController(
            state_dim=state_dim,
            goal_dim=subgoal_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            action_space=low_action_space,
            auto_alpha=auto_alpha,
            model_path=model_path,
            actor_lr=actor_lr,
            critic_lr=critic_lr
        )

        self.replay_buffer_low = LowReplayBuffer(
            state_dim=state_dim,
            goal_dim=subgoal_dim,
            action_dim=action_dim,
            buffer_size=buffer_size,
            batch_size=batch_size
        )

        self.replay_buffer_high = HighReplayBuffer(
            state_dim=state_dim,
            goal_dim=goal_dim,
            subgoal_dim=subgoal_dim,
            action_dim=action_dim,
            buffer_size=buffer_size,
            batch_size=batch_size,
            freq=buffer_freq
        )

        self.buffer_freq = buffer_freq
        self.train_freq = train_freq
        self.reward_denominator = 1.0 / buffer_freq
        self.episode_subreward = 0
        self.sr = 0

        self.buf = [None, None, None, 0, None, None, [], []]
        self.fg = np.array([0,0])
        self.sg = self.subgoal.action_space.sample()

        self.start_training_steps = start_training_steps

    def set_final_goal(self, fg):
        self.fg = fg

    def step(self, s, env, step, global_step=0, explore=False):
        ## Lower Level Controller
        if explore:
            # Take random action for start_training_steps
            if global_step < self.start_training_steps:
                a = env.action_space.sample()
            else:
                a = self._choose_action_with_noise(s, self.sg)
        else:
            a = self._choose_action(s, self.sg)

        # Take action
        obs, r, done, _ = env.step(a)
        n_s = obs['observation']

        ## Higher Level Controller
        # Take random action for start_training steps
        if explore:
            if global_step < self.start_training_steps:
                n_sg = self.subgoal.action_space.sample()
            else:
                n_sg = self._choose_subgoal_with_noise(step, s, self.sg, n_s)
        else:
            n_sg = self._choose_subgoal(step, s, self.sg, n_s)

        self.n_sg = n_sg

        return a, r, n_s, done

    def append(self, step, s, a, n_s, r, d):
        self.sr = self.low_reward(s, self.sg, n_s)

        # Low Replay Buffer
        self.replay_buffer_low.append(
            s, self.sg, a, n_s, self.n_sg, self.sr, float(d))

        # High Replay Buffer
        if _is_update(step, self.buffer_freq, rem=1):
            if len(self.buf[6]) == self.buffer_freq:
                self.buf[4] = s
                self.buf[5] = float(d)
                self.replay_buffer_high.append(
                    state=self.buf[0],
                    goal=self.buf[1],
                    action=self.buf[2],
                    n_state=self.buf[4],
                    reward=self.buf[3],
                    done=self.buf[5],
                    state_arr=np.array(self.buf[6]),
                    action_arr=np.array(self.buf[7])
                )
            self.buf = [s, self.fg, self.sg, 0, None, None, [], []]

        self.buf[3] += self.reward_denominator * r
        self.buf[6].append(s)
        self.buf[7].append(a)

    def train(self, global_step):
        losses = {}
        params = {}

        # print("The final goal is:", self.fg)

        if global_step >= self.start_training_steps:
            loss, param = self.low_con.train(self.replay_buffer_low)
            losses.update(loss)
            params.update(param)

            if global_step % self.train_freq == 0:
                loss, param = self.high_con.train(self.replay_buffer_high, self.low_con)
                losses.update(loss)
                params.update(param)

        return losses, params

    def evaluate_policy(self, env, eval_episodes=10, render=False, save_video=False, sleep=-1):
        if save_video:
            # from OpenGL import GL
            env = gym.wrappers.Monitor(env, directory='video',
                                       write_upon_reset=True, force=True, resume=True, mode='evaluation')
            render = False

        success = 0
        rewards = []
        env.evaluate = True
        for e in range(eval_episodes):
            obs = env.reset()
            fg = obs['desired_goal']
            s = obs['observation']
            done = False
            reward_episode_sum = 0
            step = 0

            self.set_final_goal(fg)

            while not done:
                if render:
                    env.render()
                if sleep > 0:
                    time.sleep(sleep)

                a, r, n_s, done = self.step(s, env, step)
                reward_episode_sum += r

                s = n_s
                step += 1
                self.end_step()
            else:
                error = np.sqrt(np.sum(np.square(fg-s[:2])))
                print('Goal, Curr: (%02.2f, %02.2f, %02.2f, %02.2f)     Error:%.2f'%(fg[0], fg[1], s[0], s[1], error))
                rewards.append(reward_episode_sum)
                success += 1 if error <=5 else 0
                self.end_episode(e)

        env.evaluate = False
        return np.array(rewards), success/eval_episodes

    def _choose_action_with_noise(self, s, sg):
        return self.low_con.policy(s, sg, explore=True)

    def _choose_subgoal_with_noise(self, step, s, sg, n_s):
        if step % self.buffer_freq == 0: # Should be zero
            sg = self.high_con.policy(s, self.fg, explore=True)
        else:
            sg = self.subgoal_transition(s, sg, n_s)
            # print("s shape %s, sg shape %s, n_s shape:%s" % (np.shape(s), np.shape(sg), np.shape(n_s)))

        return sg

    def _choose_action(self, s, sg):
        return self.low_con.policy(s, sg, explore=False)

    def _choose_subgoal(self, step, s, sg, n_s):
        if step % self.buffer_freq == 0:
            sg = self.high_con.policy(s, self.fg, explore=False)
        else:
            sg = self.subgoal_transition(s, sg, n_s)

        return sg

    @staticmethod
    def subgoal_transition(s, sg, n_s):
        return s[:sg.shape[0]] + sg - n_s[:sg.shape[0]]

    @staticmethod
    def low_reward(s, sg, n_s):
        # print('s shape: %s, sg shape:%s' %(np.shape(s), np.shape(sg)))
        abs_s = s[:sg.shape[0]] + sg
        # print('abs_s shape:', abs_s)
        return -np.sqrt(np.sum((abs_s - n_s[:sg.shape[0]])**2))

    def end_step(self):
        self.episode_subreward += self.sr
        self.sg = self.n_sg

    def end_episode(self, episode, logger=None):
        if logger:
            # log
            logger.write('reward/Intrinsic Reward', self.episode_subreward, episode)

            # Save Model
            if _is_update(episode, self.model_save_freq):
                self.save(episode=episode)

        self.episode_subreward = 0
        self.sr = 0
        self.buf = [None, None, None, 0, None, None, [], []]

    def save(self, episode):
        self.low_con.save_model(episode)
        self.high_con.save_model(episode)

    def load(self, episode):
        self.low_con.load_model(episode)
        self.high_con.load_model(episode)
