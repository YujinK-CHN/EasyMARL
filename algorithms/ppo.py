import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import defaultdict

class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        self.__init__()

# ==========================================================
# Encoders
# ==========================================================

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_sizes=[128, 128], activation='relu'):
        super().__init__()

        act = nn.ReLU if activation == 'relu' else nn.Tanh

        layers = []
        prev = input_dim

        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), act()]
            prev = h

        self.net = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, x):
        return self.net(x)


class CNNEncoder(nn.Module):
    def __init__(self, input_shape, activation='relu'):
        super().__init__()

        # input_shape is expected as (H, W, C) from PettingZoo
        h, w, c = input_shape
        #print(f'Input shape (HWC): {input_shape}')

        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            # FIX: convert HWC → CHW
            dummy = torch.zeros(1, h, w, c).permute(0, 3, 1, 2)

            #print(f'Dummy input shape (CHW): {dummy.shape}')

            n_flat = self.cnn(dummy).shape[1]
            # print(f'CNN output shape: {n_flat}')

        self.fc = nn.Sequential(
            nn.Linear(n_flat, 256),
            nn.ReLU()
        )

        self.out_dim = 256

    def forward(self, x):
        if x.ndim == 3:
            x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        elif x.ndim == 4:
            x = x.permute(0, 3, 1, 2)
        #print(f'CNN input shape: {x.shape}')
        x = self.cnn(x)
        return self.fc(x)

# ==========================================================
# Actor-Critic
# ==========================================================

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, encoder):
        super().__init__()

        self.encoder = encoder

        self.actor = nn.Sequential(
            nn.Linear(encoder.out_dim, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim)
        )

        self.critic = nn.Sequential(
            nn.Linear(encoder.out_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs):
        z = self.encoder(obs)
        return self.actor(z), self.critic(z)

    def act(self, obs):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate(self, obs, actions):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, value.squeeze(-1)

# ==========================================================
# PPO
# ==========================================================

class PPO:
    def __init__(self, config, env):
        self.config = config
        self.env = env
        self.device = self._device()
        self.buffer = RolloutBuffer()

        self.agents = env.possible_agents
        self.shared = config['algorithm']['shared_policy']
        self.gamma = config['training']['gamma']
        self.gae_lambda = config['training']['gae_lambda']
        self.ppo_epochs = config['training']['ppo_epochs']
        self.mini_batch_size = config['training']['mini_batch_size']
        self.clip_range = config['training']['clip_range']
        self.value_loss_coef = config['training']['value_loss_coef']
        self.entropy_coef = config['training']['entropy_coef']
        self.max_grad_norm = config['training']['max_grad_norm']

        obs_space = env.observation_space(self.agents[0])
        act_space = env.action_space(self.agents[0])

        self.act_dim = act_space.n

        obs_shape = obs_space.shape

        if len(obs_shape) == 1:
            encoder = lambda: MLPEncoder(obs_shape[0])
        else:
            encoder = lambda: CNNEncoder(obs_shape)

        self.policies = {}
        self.optimizers = {}

        lr = config['training']['learning_rate']

        if self.shared:
            net = ActorCritic(obs_shape, self.act_dim, encoder()).to(self.device)

            for a in self.agents:
                self.policies[a] = net

            self.optimizer = optim.Adam(net.parameters(), lr=lr)

        else:
            for a in self.agents:
                net = ActorCritic(obs_shape, self.act_dim, encoder()).to(self.device)
                self.policies[a] = net
                self.optimizers[a] = optim.Adam(net.parameters(), lr=lr)

    # ------------------------------------------------------

    def _device(self):
        if self.config['device']['use_cuda'] and torch.cuda.is_available():
            return torch.device(f"cuda:{self.config['device']['gpu_id']}")
        return torch.device("cpu")

    # ------------------------------------------------------

    def select_actions(self, obs):
        actions, logps, values = {}, {}, {}

        for a, o in obs.items():
            o = torch.tensor(o, dtype=torch.float32, device=self.device).unsqueeze(0)

            net = self.policies[a]

            act, logp, val = net.act(o)

            actions[a] = act.item()
            logps[a] = logp
            values[a] = val

        return actions, logps, values

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
        agents = rewards[0].keys()

        # initialize per-agent storage
        returns_dict = {a: [] for a in agents}
        advantages_dict = {a: [] for a in agents}

        gae = {a: 0 for a in agents}

        # append next value per agent
        values = values + [next_value]

        for t in reversed(range(len(rewards))):
            for a in agents:

                reward = rewards[t][a]
                done = dones[t][a]
                value = values[t][a]
                next_v = values[t + 1][a]

                mask = 1.0 - float(done)

                delta = reward + self.gamma * next_v * mask - value

                gae[a] = delta + self.gamma * self.gae_lambda * mask * gae[a]

                advantages_dict[a].insert(0, gae[a])
                returns_dict[a].insert(0, gae[a] + value)

        return returns_dict, advantages_dict

    # ======================================================
    # PPO Update
    # ======================================================

    def update(self):
        if self.shared:
            agents = self.agents

            obs = []
            actions = []
            old_log_probs = []
            returns = []
            advantages = []

            # flatten all agents into one dataset
            for t in range(len(self.buffer.obs)):
                for a in agents:
                    obs.append(self.buffer.obs[t][a])
                    actions.append(self.buffer.actions[t][a])
                    old_log_probs.append(self.buffer.log_probs[t][a])
                    returns.append(self.buffer.returns[a][t])
                    advantages.append(self.buffer.advantages[a][t])

            obs = torch.FloatTensor(np.array(obs)).to(self.device)
            actions = torch.LongTensor(np.array(actions)).to(self.device)
            old_log_probs = torch.FloatTensor(np.array(old_log_probs)).to(self.device)
            returns = torch.FloatTensor(np.array(returns)).to(self.device)
            advantages = torch.FloatTensor(np.array(advantages)).to(self.device)

            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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
                        self.policy.evaluate_actions(mb_obs, mb_actions)

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

                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

                    self.optimizer.step()

            self.buffer.clear()
        else:
            for a in self.agents:

                obs = torch.FloatTensor(
                    np.array([t[a] for t in self.buffer.obs])
                ).to(self.device)

                actions = torch.LongTensor(
                    np.array([t[a] for t in self.buffer.actions])
                ).to(self.device)

                old_log_probs = torch.FloatTensor(
                    np.array([t[a] for t in self.buffer.log_probs])
                ).to(self.device)

                returns = torch.FloatTensor(
                    np.array(self.buffer.returns[a])
                ).to(self.device)

                advantages = torch.FloatTensor(
                    np.array(self.buffer.advantages[a])
                ).to(self.device)

                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                policy = self.policies[a]
                optimizer = self.optimizers[a]

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
                            policy.evaluate_actions(mb_obs, mb_actions)

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

                        optimizer.zero_grad()
                        loss.backward()

                        nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)

                        optimizer.step()

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
                #print(f'Actions: {actions}')

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

                self.buffer.obs.append(observations)
                self.buffer.actions.append(actions)
                self.buffer.log_probs.append({
                    a: log_probs[a].detach().cpu().item()
                    for a in self.agents
                })
                self.buffer.rewards.append(rewards)
                self.buffer.dones.append(done_dict)
                self.buffer.values.append({
                    a: values[a].detach().cpu().item()
                    for a in self.agents
                })

                episode_reward += np.mean(list(rewards.values()))

                observations = next_obs

                timestep += 1

            next_value = {a: 0.0 for a in self.agents}

            #print(self.buffer.rewards)
            #print(self.buffer.values)

            returns, advantages = \
                self.compute_returns_and_advantages(
                    self.buffer.rewards,
                    self.buffer.dones,
                    self.buffer.values,
                    next_value
                )
            
            #print(f'Returns: {len(returns["first_0"])}')
            #print(f'Advantages: {len(advantages["first_0"])}')

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
