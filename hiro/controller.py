import os
import torch
from torch.optim import Adam
import torch.nn.functional as F
from hiro.network import SacCritic, SacActor
from hiro.utils import get_tensor
import numpy as np


class SacController:
    def __init__(
            self,
            state_dim,
            goal_dim,
            action_dim,
            hidden_dim,
            action_space,
            auto_alpha,
            model_path,
            actor_lr,
            critic_lr,
            reward_scale=1,
            alpha=0.2,
            gamma=0.99,
            tau=0.005,
            target_update_interval=1):

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
            'cpu')
        self.name = 'sac'
        self.model_path = model_path

        self.auto_alpha = auto_alpha
        self.gamma = gamma
        self.tau = tau
        self.reward_scale = reward_scale
        self.alpha = alpha
        self.target_update_interval = target_update_interval
        self.action_space = action_space

        self.critic = SacCritic(state_dim, goal_dim, action_dim, hidden_dim).to(device=self.device)
        self.critic_target = SacCritic(state_dim, goal_dim, action_dim, hidden_dim).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr)

        self.actor = SacActor(state_dim, goal_dim, action_dim, hidden_dim, action_space).to(device=self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr=actor_lr)

        if self.auto_alpha:
            self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=actor_lr)

        self._initialized = False
        self._initialize_target_networks()

        self.updates = 0

    def _initialize_target_networks(self):
        self._update_target_network(self.critic_target, self.critic, 1.0)
        self.initialized = True

    @staticmethod
    def _update_target_network(target, origin, tau):
        for target_param, origin_param in zip(target.parameters(), origin.parameters()):
            target_param.data.copy_(tau * origin_param.data + (1.0 - tau) * target_param.data)

    def _train(self, s, g, a, n_s, n_g, r, not_d):

        self.updates += 1
        with torch.no_grad():
            n_a, n_log_pi, _ = self.actor.choose_action_log_prob(n_s, n_g)
            next_Q1_target, next_Q2_target = self.critic_target(n_s, n_g, n_a)
            min_next_Q_target = torch.min(next_Q1_target, next_Q2_target) - self.alpha * n_log_pi
            next_Q = r * self.reward_scale + not_d * self.gamma * min_next_Q_target

        Q1, Q2 = self.critic(s, g, a)  # Two Q-functions to mitigate positive bias in the policy improvement step
        Q1_loss = F.mse_loss(Q1, next_Q)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        Q2_loss = F.mse_loss(Q2, next_Q)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        pi, log_pi, _ = self.actor.choose_action_log_prob(s, g)

        Q1_pi, Q2_pi = self.critic(s, g, pi)
        min_Q_pi = torch.min(Q1_pi, Q2_pi)

        policy_loss = (
                (self.alpha * log_pi) - min_Q_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.critic_optim.zero_grad()
        Q1_loss.backward()
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        Q2_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        if self.updates % self.target_update_interval == 0:
            self._update_target_network(self.critic_target, self.critic, self.tau)

        return {"Q1_loss_"+self.name: Q1_loss.item(), "Q2_loss_"+self.name: Q2_loss.item(),
                "Actor_loss_"+self.name: policy_loss.item(), "Alpha_loss_"+self.name: alpha_loss.item()}, \
               {"Alpha_"+self.name: alpha_tlogs.item(), "Q1_batch_mean_"+self.name: torch.mean(Q1).item(),
                "Q2_batch_mean_"+self.name:torch.mean(Q2).item(), "next_Q_batch_mean_"+self.name:torch.mean(next_Q).item()}

    def train(self, replay_buffer):
        s, g, a, n_s, r, not_d = replay_buffer.sample()
        return self._train(s, g, a, n_s, g, r, not_d)

    def policy(self, s, g, explore=True, to_numpy=True):
        s = get_tensor(s)
        g = get_tensor(g)
        if explore:
            a, _, _ = self.actor.choose_action_log_prob(s, g)
        else:
            _, _, a = self.actor.choose_action_log_prob(s, g)

        if to_numpy:
            return a.detach().cpu().numpy().squeeze()
        return a.detach().cpu().squeeze()

        # Save model parameters

    def save_model(self, episode):
        print('\n ======================    save model     ====================')
        model_path = os.path.join(self.model_path, str(episode))
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.actor.state_dict(),
                   os.path.join(model_path, self.name+"_actor.h5"))
        torch.save(self.critic.state_dict(),
                   os.path.join(model_path, self.name+'/critic.h5'))
        print('\n =============    model successfully be saved     ============')

    def load_model(self, episode):
        print('\n ======================    load model     ====================')
        # episode is -1, then read most updated
        if episode < 0:
            episode_list = map(int, os.listdir(self.model_path))
            episode = max(episode_list)

        model_path = os.path.join(self.model_path, str(episode))
        self.actor.load_state_dict(torch.load(
            os.path.join(model_path, self.name+'/actor.h5')))
        self.critic.load_state_dict(torch.load(
            os.path.join(model_path, self.name+'/critic.h5')))
        self.actor.eval()
        self.critic.eval()
        print('\n =============    model successfully be loaded     ============')


class LowerController(SacController):
    def __init__(
            self,
            state_dim,
            goal_dim,
            action_dim,
            hidden_dim,
            action_space,
            auto_alpha,
            model_path,
            actor_lr,
            critic_lr,
            reward_scale=1,
            alpha=0.2,
            gamma=0.99,
            tau=0.005,
            target_update_interval=1):
        super(LowerController, self).__init__(
            state_dim, goal_dim, action_dim, hidden_dim, action_space,
            auto_alpha, model_path, actor_lr, critic_lr, reward_scale, alpha,
            gamma, tau, target_update_interval)
        self.name = 'low'

    def train(self, replay_buffer):
        if not self._initialized:
            self._initialize_target_networks()

        s, sg, a, n_s, n_sg, r, not_d = replay_buffer.sample()
        return self._train(s, sg, a, n_s, n_sg, r, not_d)


class HigherController(SacController):
    def __init__(
            self,
            state_dim,
            goal_dim,
            action_dim,
            hidden_dim,
            action_space,
            auto_alpha,
            model_path,
            actor_lr,
            critic_lr,
            reward_scale=1.0,
            alpha=0.2,
            gamma=0.99,
            tau=0.005,
            target_update_interval=1):
        super(HigherController, self).__init__(
            state_dim, goal_dim, action_dim, hidden_dim, action_space,
            auto_alpha, model_path, actor_lr, critic_lr, reward_scale, alpha,
            gamma, tau, target_update_interval)
        self.action_dim = action_dim
        self.name = 'high'

    def off_policy_corrections(self, low_con, batch_size, sgoals, states, actions, candidate_goals=8):
        first_s = [s[0] for s in states] # First x
        last_s = [s[-1] for s in states] # Last x
        # Shape: (batch_size, 1, subgoal_dim)
        # diff = 1
        diff_goal = (np.array(last_s) -
                     np.array(first_s))[:, np.newaxis, :self.action_dim]

        # Shape: (batch_size, 1, subgoal_dim)
        # original = 1
        # random = candidate_goals
        original_goal = np.array(sgoals)[:, np.newaxis, :]
        random_goals = np.random.normal(loc=diff_goal, scale=0.5 * 0.5 * (self.action_space.high -self.action_space.low)[None, None, :],
                                        size=(batch_size, candidate_goals, original_goal.shape[-1]))
        random_goals = random_goals.clip(self.action_space.low, self.action_space.high)

        # Shape: (batch_size, 10, subgoal_dim)
        candidates = np.concatenate([original_goal, diff_goal, random_goals], axis=1)
        #states = np.array(states)[:, :-1, :]
        actions = np.array(actions)
        seq_len = len(states[0])

        # For ease
        new_batch_sz = seq_len * batch_size
        action_dim = actions[0][0].shape
        obs_dim = states[0][0].shape
        ncands = candidates.shape[1]

        true_actions = actions.reshape((new_batch_sz,) + action_dim)
        observations = states.reshape((new_batch_sz,) + obs_dim)
        goal_shape = (new_batch_sz, self.action_dim)
        # observations = get_obs_tensor(observations, sg_corrections=True)

        # batched_candidates = np.tile(candidates, [seq_len, 1, 1])
        # batched_candidates = batched_candidates.transpose(1, 0, 2)

        policy_actions = np.zeros((ncands, new_batch_sz) + action_dim)

        for c in range(ncands):
            subgoal = candidates[:,c]
            candidate = (subgoal + states[:, 0, :self.action_dim])[:, None] - states[:, :, :self.action_dim]
            candidate = candidate.reshape(*goal_shape)
            policy_actions[c] = low_con.policy(observations, candidate)

        difference = (policy_actions - true_actions)
        difference = np.where(difference != -np.inf, difference, 0)
        difference = difference.reshape((ncands, batch_size, seq_len) + action_dim).transpose(1, 0, 2, 3)

        logprob = -0.5*np.sum(np.linalg.norm(difference, axis=-1)**2, axis=-1)
        max_indices = np.argmax(logprob, axis=-1)

        return candidates[np.arange(batch_size), max_indices]

    def train(self, replay_buffer, low_con):
        if not self._initialized:
            self._initialize_target_networks()

        states, goals, actions, n_states, rewards, not_done, states_arr, actions_arr = replay_buffer.sample()

        actions = self.off_policy_corrections(
            low_con,
            replay_buffer.batch_size,
            actions.cpu().data.numpy(),
            states_arr.cpu().data.numpy(),
            actions_arr.cpu().data.numpy())

        actions = get_tensor(actions)
        return self._train(states, goals, actions, n_states, goals, rewards, not_done)
