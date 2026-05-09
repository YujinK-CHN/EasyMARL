import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# ==========================================================
# Utility Functions
# ==========================================================


def build_mlp(input_dim, hidden_sizes, output_dim, activation='relu'):
    activations = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'elu': nn.ELU,
    }

    act_fn = activations[activation]

    layers = []
    prev_dim = input_dim

    for h in hidden_sizes:
        layers.append(nn.Linear(prev_dim, h))
        layers.append(act_fn())
        prev_dim = h

    layers.append(nn.Linear(prev_dim, output_dim))

    return nn.Sequential(*layers)


# ==========================================================
# Actor-Critic Network
# ==========================================================


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        actor_hidden_sizes,
        critic_hidden_sizes,
        activation='relu'
    ):
        super().__init__()

        self.actor = build_mlp(
            obs_dim,
            actor_hidden_sizes,
            action_dim,
            activation
        )

        self.critic = build_mlp(
            obs_dim,
            critic_hidden_sizes,
            1,
            activation
        )

    def forward(self, obs):
        logits = self.actor(obs)
        value = self.critic(obs)
        return logits, value

    def act(self, obs):
        logits, value = self.forward(obs)

        dist = Categorical(logits=logits)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob, value

    def evaluate_actions(self, obs, actions):
        logits, values = self.forward(obs)

        dist = Categorical(logits=logits)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, entropy, values.squeeze(-1)


# ==========================================================
# Rollout Buffer
# ==========================================================


class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []


# ==========================================================
# PPO Agent
# ==========================================================


class PPO:
    def __init__(self, config, env):
        self.config = config
        self.env = env

        self.device = self._build_device()

        self.gamma = config['training']['gamma']
        self.gae_lambda = config['training']['gae_lambda']
        self.clip_range = config['training']['clip_range']
        self.ppo_epochs = config['training']['ppo_epochs']
        self.batch_size = config['training']['batch_size']
        self.mini_batch_size = config['training']['mini_batch_size']
        self.entropy_coef = config['training']['entropy_coef']
        self.value_loss_coef = config['training']['value_loss_coef']
        self.max_grad_norm = config['training']['max_grad_norm']

        self.shared_policy = config['algorithm']['shared_policy']

        self.agents = env.possible_agents

        obs_space = env.observation_space(self.agents[0])
        act_space = env.action_space(self.agents[0])

        self.obs_dim = obs_space.shape[0]
        self.action_dim = act_space.n

        self.policy = ActorCritic(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            actor_hidden_sizes=config['model']['actor_hidden_sizes'],
            critic_hidden_sizes=config['model']['critic_hidden_sizes'],
            activation=config['model']['activation']
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=config['training']['learning_rate']
        )

        self.buffer = RolloutBuffer()

    # ======================================================
    # Device
    # ======================================================

    def _build_device(self):
        use_cuda = self.config['device']['use_cuda']

        if use_cuda and torch.cuda.is_available():
            gpu_id = self.config['device']['gpu_id']
            return torch.device(f'cuda:{gpu_id}')

        return torch.device('cpu')

    # ======================================================
    # Action Selection
    # ======================================================

    def select_actions(self, observations):
        actions = {}
        log_probs = {}
        values = {}

        for agent, obs in observations.items():
            obs_tensor = torch.FloatTensor(obs).to(self.device)

            action, log_prob, value = self.policy.act(obs_tensor)

            actions[agent] = action.item()
            log_probs[agent] = log_prob.detach()
            values[agent] = value.detach()

        return actions, log_probs, values

    # ======================================================
    # GAE Computation
    # ======================================================

    def compute_returns_and_advantages(
        self,
        rewards,
        dones,
        values,
        next_value
    ):
        advantages = []

        gae = 0

        values = values + [next_value]

        for step in reversed(range(len(rewards))):
            delta = (
                rewards[step]
                + self.gamma * values[step + 1] * (1 - dones[step])
                - values[step]
            )

            gae = (
                delta
                + self.gamma
                * self.gae_lambda
                * (1 - dones[step])
                * gae
            )

            advantages.insert(0, gae)

        returns = [a + v for a, v in zip(advantages, values[:-1])]

        return returns, advantages

    # ======================================================
    # PPO Update
    # ======================================================

    def update(self):
        obs = torch.FloatTensor(np.array(self.buffer.obs)).to(self.device)
        actions = torch.LongTensor(np.array(self.buffer.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(
            np.array(self.buffer.log_probs)
        ).to(self.device)

        returns = torch.FloatTensor(np.array(self.buffer.returns)).to(self.device)
        advantages = torch.FloatTensor(
            np.array(self.buffer.advantages)
        ).to(self.device)

        advantages = (
            (advantages - advantages.mean())
            / (advantages.std() + 1e-8)
        )

        dataset_size = obs.shape[0]

        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(dataset_size)

            for start in range(0, dataset_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_idx = indices[start:end]

                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]

                new_log_probs, entropy, values = \
                    self.policy.evaluate_actions(
                        mb_obs,
                        mb_actions
                    )

                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_range,
                    1.0 + self.clip_range
                ) * mb_advantages

                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = (mb_returns - values).pow(2).mean()

                entropy_loss = entropy.mean()

                loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.max_grad_norm
                )

                self.optimizer.step()

        self.buffer.clear()

    # ======================================================
    # Training Loop
    # ======================================================

    def train(self):
        total_timesteps = self.config['training']['total_timesteps']

        timestep = 0

        while timestep < total_timesteps:
            observations, infos = self.env.reset()

            done = False

            episode_reward = 0

            self.buffer.clear()

            while not done:
                actions, log_probs, values = \
                    self.select_actions(observations)

                next_obs, rewards, terminations, truncations, infos = \
                    self.env.step(actions)

                done_dict = {
                    agent: (
                        terminations[agent]
                        or truncations[agent]
                    )
                    for agent in self.agents
                }

                done = all(done_dict.values())

                for agent in self.agents:
                    self.buffer.obs.append(observations[agent])
                    self.buffer.actions.append(actions[agent])
                    self.buffer.log_probs.append(
                        log_probs[agent].cpu().numpy()
                    )
                    self.buffer.rewards.append(rewards[agent])
                    self.buffer.dones.append(done_dict[agent])
                    self.buffer.values.append(
                        values[agent].cpu().numpy().item()
                    )

                    episode_reward += rewards[agent]

                observations = next_obs

                timestep += 1

            next_value = 0

            returns, advantages = \
                self.compute_returns_and_advantages(
                    self.buffer.rewards,
                    self.buffer.dones,
                    self.buffer.values,
                    next_value
                )

            self.buffer.returns = returns
            self.buffer.advantages = advantages

            self.update()

            if timestep % self.config['logging']['log_interval'] == 0:
                print(
                    f'[PPO] timestep={timestep} '
                    f'episode_reward={episode_reward:.2f}'
                )

    # ======================================================
    # Evaluation
    # ======================================================

    def evaluate(self, episodes=5):
        rewards = []

        for _ in range(episodes):
            observations, infos = self.env.reset()

            done = False
            episode_reward = 0

            while not done:
                actions = {}

                with torch.no_grad():
                    for agent, obs in observations.items():
                        obs_tensor = torch.FloatTensor(obs).to(self.device)

                        logits, _ = self.policy(obs_tensor)

                        action = torch.argmax(logits).item()

                        actions[agent] = action

                observations, reward, terminations, truncations, infos = \
                    self.env.step(actions)

                episode_reward += sum(reward.values())

                done = all([
                    terminations[a] or truncations[a]
                    for a in self.agents
                ])

            rewards.append(episode_reward)

        avg_reward = np.mean(rewards)

        print(f'[Evaluation] Average Reward: {avg_reward:.2f}')

        return avg_reward
