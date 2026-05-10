import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
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

        # single observation
        if x.ndim == 3:
            x = x.unsqueeze(0)

        # NHWC -> NCHW
        x = x.permute(0, 3, 1, 2)

        x = x.float() / 255.0

        x = self.cnn(x)

        return self.fc(x)

# ==========================================================
# Actor-Critic
# ==========================================================

class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim,
        act_dim,
        encoder,
        action_space='discrete',   # "discrete" or "continuous"
        log_std_init=-0.5
    ):
        super().__init__()

        self.encoder = encoder
        self.action_space = action_space
        self.act_dim = act_dim

        # --------------------------------------------------
        # Actor
        # --------------------------------------------------

        if self.action_space == 'discrete':

            self.actor = nn.Sequential(
                nn.Linear(encoder.out_dim, 256),
                nn.ReLU(),
                nn.Linear(256, act_dim)
            )

        elif self.action_space == 'continuous':

            # Mean network
            self.actor_mean = nn.Sequential(
                nn.Linear(encoder.out_dim, 256),
                nn.ReLU(),
                nn.Linear(256, act_dim)
            )

            # Learnable log standard deviation
            self.actor_log_std = nn.Parameter(
                torch.ones(act_dim) * log_std_init
            )

        else:
            raise ValueError(
                f'Unsupported action_space: {action_space}'
            )

        # --------------------------------------------------
        # Critic
        # --------------------------------------------------

        self.critic = nn.Sequential(
            nn.Linear(encoder.out_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    # ======================================================
    # Forward
    # ======================================================

    def forward(self, obs):
        z = self.encoder(obs)

        if self.action_space == 'discrete':
            actor_out = self.actor(z)

        else:
            actor_out = self.actor_mean(z)
            actor_out = torch.clamp(actor_out, -10, 10)

        value = self.critic(z)

        return actor_out, value

    # ======================================================
    # Distribution helper
    # ======================================================

    def get_dist(self, obs):
        actor_out, value = self.forward(obs)

        if self.action_space == 'discrete':

            dist = Categorical(logits=actor_out)

        else:
            mean = torch.nan_to_num(actor_out, nan=0.0, posinf=1.0, neginf=-1.0)

            log_std = torch.clamp(
                self.actor_log_std,
                -20,
                2
            )

            std = torch.exp(log_std).clamp(1e-3, 1.0)
            std = std.expand_as(mean)
            if torch.isnan(mean).any():
                print("NaN in mean detected!")
                print(mean)
                raise ValueError("Policy exploded")
            dist = Normal(mean, std)

        return dist, value

    # ======================================================
    # Action sampling
    # ======================================================

    def act(self, obs):
        dist, value = self.get_dist(obs)

        raw_action = dist.sample()

        if self.action_space == 'continuous':
            action = torch.tanh(raw_action)
        else:
            action = raw_action

        if self.action_space == 'discrete':
            log_prob = dist.log_prob(action)

        else:
            # sum across action dimensions
            log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob, value

    # ======================================================
    # PPO evaluation
    # ======================================================

    def evaluate(self, obs, actions):
        dist, value = self.get_dist(obs)

        if self.action_space == 'discrete':

            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

        else:

            log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

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
        self.batch_size = config['training']['batch_size']
        self.mini_batch_size = config['training']['mini_batch_size']
        self.clip_range = config['training']['clip_range']
        self.value_loss_coef = config['training']['value_loss_coef']
        self.entropy_coef = config['training']['entropy_coef']
        self.max_grad_norm = config['training']['max_grad_norm']

        # ======================================================
        # Evaluation
        # ======================================================

        self.eval_enabled = config['evaluation']['enabled']
        self.eval_interval = config['evaluation']['eval_interval']
        self.eval_episodes = config['evaluation']['eval_episodes']

        # ======================================================
        # Logging
        # ======================================================

        self.log_interval = config['logging']['log_interval']
        self.save_model_interval = config['logging']['save_model_interval']

        obs_space = env.observation_space(self.agents[0])
        act_space = env.action_space(self.agents[0])

        obs_shape = obs_space.shape

        if len(obs_shape) == 1:
            encoder = lambda: MLPEncoder(obs_shape[0])
        else:
            encoder = lambda: CNNEncoder(obs_shape)

        

        lr = config['training']['learning_rate']

        act_space = env.action_space(self.agents[0])

        if hasattr(act_space, "n"):
            action_space_type = "discrete"
            self.act_dim = act_space.n

        else:
            action_space_type = "continuous"
            self.act_dim = act_space.shape[0]

        print(f'Action space type: {action_space_type}, act_dim: {self.act_dim}')

        if self.shared:
            net = ActorCritic(obs_shape, self.act_dim, encoder(), action_space=action_space_type).to(self.device)

            #for a in self.agents:
            self.policy = net

            self.optimizer = optim.Adam(net.parameters(), lr=lr)

        else:
            self.policies = {}
            self.optimizers = {}
            for a in self.agents:
                net = ActorCritic(obs_shape, self.act_dim, encoder(), action_space=action_space_type).to(self.device)
                self.policies[a] = net
                self.optimizers[a] = optim.Adam(net.parameters(), lr=lr)

    # ------------------------------------------------------

    def _device(self):
        if self.config['device']['use_cuda'] and torch.cuda.is_available():
            return torch.device(f"cuda:{self.config['device']['gpu_id']}")
        return torch.device("cpu")

    # ------------------------------------------------------

    def select_actions(self, obs):
        if self.shared:
            actions, logps, values = {}, {}, {}

            for a, o in obs.items():
                o = (
                    torch.tensor(
                        o,
                        dtype=torch.float32,
                        device=self.device
                    ) / 255.0
                ).unsqueeze(0)

                act, logp, val = self.policy.act(o)

                if self.policy.action_space == 'discrete':
                    actions[a] = act.item()
                else:
                    actions[a] = (
                        act.squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                logps[a] = logp
                values[a] = val

            return actions, logps, values
        
        else:
            actions, logps, values = {}, {}, {}

            for a, o in obs.items():
                o = torch.tensor(o, dtype=torch.float32, device=self.device).unsqueeze(0)

                net = self.policies[a]

                act, logp, val = net.act(o)

                if net.action_space == 'discrete':
                    actions[a] = act.item()
                else:
                    actions[a] = (
                        act.squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
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

            obs = torch.FloatTensor(
                np.array(obs)
            ).to(self.device) / 255.0
            actions = torch.LongTensor(np.array(actions)).to(self.device)
            old_log_probs = torch.FloatTensor(np.array(old_log_probs)).to(self.device)
            returns = torch.FloatTensor(np.array(returns)).to(self.device)
            advantages = torch.FloatTensor(np.array(advantages)).to(self.device)

            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages = torch.clamp(advantages, -5, 5)
            returns = torch.clamp(returns, -10, 10)

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
                        self.policy.evaluate(mb_obs, mb_actions)

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

        else:
            for a in self.agents:

                obs = torch.FloatTensor(
                    np.array([t[a] for t in self.buffer.obs])
                ).to(self.device) / 255.0

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
                            policy.evaluate(mb_obs, mb_actions)

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


    # ======================================================
    # Training Loop
    # ======================================================

    def train(self):
        total_timesteps = self.config['training']['total_timesteps']

        timestep = 0

        episode_reward = 0.0

        observations, infos = self.env.reset(seed=self.config['env']['seed'])

        self.buffer.clear()

        while timestep < total_timesteps:
            # ---------- Evaluation ----------
            if (self.eval_enabled and timestep % self.eval_interval == 0):

                avg_reward = self.evaluate(
                    episodes=self.eval_episodes
                )

                print(
                    f'[Evaluation] timestep={timestep} '
                    f'avg_reward={avg_reward:.2f}'
                )
                observations, infos = self.env.reset(seed=self.config['env']['seed'])
            # --------------------------------

            # -------- Checkpointing ---------
            if timestep % self.save_model_interval == 0 and timestep > 0:

                self.save(
                    f'checkpoints/checkpoint_{timestep}.pt'
                )

                print(f'[Checkpoint] saved at timestep={timestep}')
            # --------------------------------

            actions, log_probs, values = \
                    self.select_actions(observations)
                # print(f'Actions: {actions}')

            next_obs, rewards, terminations, truncations, infos = \
                self.env.step(actions)
            # print(f'Rewards: {rewards}')

            if timestep % self.log_interval == 0:
                print(
                    f'[PPO] timestep={timestep} '
                    f'episode_reward={episode_reward:.2f}'
                )
                episode_reward = 0.0

            done_dict = {
                agent: (
                    terminations.get(agent, True)
                    or truncations.get(agent, True)
                )
                for agent in observations.keys()
            }

            done = all(done_dict.values())
            if done:
                observations, infos = self.env.reset(seed=self.config['env']['seed'])

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

            # -------- Compute returns and advantages, then update policy ---------
            if timestep % self.batch_size == 0 and timestep > 0:
                next_value = {a: 0.0 for a in self.agents}

                #print(len(self.buffer.rewards))
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
                self.buffer.clear()

            timestep += 1

    # ======================================================
    # Evaluation
    # ======================================================

    def evaluate(self, episodes=5):
        rewards = []

        for i in range(episodes):
            observations, infos = self.env.reset()

            done = False
            episode_reward = 0.0

            while not done:
                actions = {}

                with torch.no_grad():

                    for agent, obs in observations.items():

                        obs_tensor = (
                            torch.FloatTensor(obs)
                            .unsqueeze(0)
                            .to(self.device)
                        )

                        # -----------------------------
                        # choose correct policy
                        # -----------------------------
                        if self.shared:
                            policy = self.policy
                        else:
                            policy = self.policies[agent]

                        dist, _ = policy.get_dist(obs_tensor)

                        # -----------------------------
                        # action selection
                        # -----------------------------
                        if policy.action_space == 'discrete':
                            action = torch.argmax(
                                dist.logits,
                                dim=-1
                            ).item()

                        else:
                            action = dist.mean[0].cpu().numpy()
                            action = np.clip(action, -1.0, 1.0)

                        actions[agent] = action

                observations, reward, terminations, truncations, infos = \
                    self.env.step(actions)

                episode_reward += sum(reward.values())

                done = all(
                    terminations[a] or truncations[a]
                    for a in observations.keys()
                )

            rewards.append(episode_reward)

        return np.mean(rewards)
    
    def save(self, path):

        if self.shared:

            torch.save({
                'model': self.policy.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }, path)

        else:

            torch.save({
                'models': {
                    a: self.policies[a].state_dict()
                    for a in self.agents
                },
                'optimizers': {
                    a: self.optimizers[a].state_dict()
                    for a in self.agents
                }
            }, path)

    def load(self, path):

        checkpoint = torch.load(path, map_location=self.device)

        if self.shared:

            self.policy.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        else:

            for a in self.agents:
                self.policies[a].load_state_dict(
                    checkpoint['models'][a]
                )

                self.optimizers[a].load_state_dict(
                    checkpoint['optimizers'][a]
                )
