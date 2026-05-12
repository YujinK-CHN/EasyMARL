import numpy as np
import os
import random
import string
import csv
from tqdm import trange
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal

class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.global_obs = []
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
        #print("CNN input shape:", x.shape)
        #print("CNN input dtype:", x.dtype)
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
        log_std_init=-0.5,
        centralized_critic=False,
        global_obs_dim=None
    ):
        super().__init__()

        self.centralized_critic = centralized_critic
        if self.centralized_critic:
            self.actor_encoder = encoder
        else:
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
        if self.centralized_critic:
            self.critic = nn.Sequential(
                nn.Linear(global_obs_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
        else:
            self.critic = nn.Sequential(
                nn.Linear(encoder.out_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )

    # ======================================================
    # Forward
    # ======================================================

    def forward(self, obs, global_obs=None):
        if self.centralized_critic and global_obs is not None:
            actor_z = self.actor_encoder(obs)

            if self.action_space == 'discrete':
                actor_out = self.actor(actor_z)

            else:
                actor_out = self.actor_mean(actor_z)
                actor_out = torch.clamp(actor_out, -10, 10)

            value = self.critic(global_obs)

            return actor_out, value
        else:
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

    def get_dist(self, obs, global_obs=None):
        if self.centralized_critic and global_obs is not None:
            actor_out, value = self.forward(obs, global_obs=global_obs)
        else:
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

    def act(self, obs, global_obs=None):
        dist, value = self.get_dist(obs, global_obs=global_obs)

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

    def evaluate(self, obs, actions, global_obs=None):
        if self.centralized_critic:
            dist, value = self.get_dist(obs, global_obs=global_obs)
        else:
            dist, value = self.get_dist(obs)

        if self.action_space == 'discrete':

            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

        else:

            log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

        return log_probs, entropy, value.squeeze(-1)
    
    def encode(self, obs):
        return self.actor_encoder(obs)

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
        self.centralized_critic = config['algorithm'].get('centralized_critic', False)
        self.total_timesteps = self.config['training']['total_timesteps']

        # ======================================================
        # Evaluation
        # ======================================================

        self.eval_enabled = config['evaluation']['enabled']
        self.eval_interval = config['evaluation']['eval_interval']
        self.eval_episodes = config['evaluation']['eval_episodes']

        # ======================================================
        # Logging
        # ======================================================

        self.log_type = config['logging']['log_type']
        self.log_interval = config['logging']['log_interval']
        self.save_model_interval = config['logging']['save_model_interval']

        obs_space = env.observation_space(self.agents[0])
        act_space = env.action_space(self.agents[0])

        obs_shape = obs_space.shape

        if len(obs_shape) == 1:
            encoder = lambda: MLPEncoder(obs_shape[0])
        else:
            encoder = lambda: CNNEncoder(obs_shape)

        if self.centralized_critic:
            self.global_obs_dim = 256 * len(self.agents)

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
            if self.centralized_critic:
                net = ActorCritic(
                    obs_shape,
                    self.act_dim,
                    encoder(),
                    action_space=action_space_type,
                    centralized_critic=True,
                    global_obs_dim=self.global_obs_dim
                ).to(self.device)
            else:
                net = ActorCritic(obs_shape, self.act_dim, encoder(), action_space=action_space_type).to(self.device)

            #for a in self.agents:
            self.policy = net

            self.optimizer = optim.Adam(net.parameters(), lr=lr)

        else:
            self.policies = {}
            self.optimizers = {}
            for a in self.agents:
                if self.centralized_critic:
                    net = ActorCritic(
                        obs_shape,
                        self.act_dim,
                        encoder(),
                        action_space=action_space_type,
                        centralized_critic=True,
                        global_obs_dim=self.global_obs_dim
                    ).to(self.device)
                else:
                    net = ActorCritic(obs_shape, self.act_dim, encoder(), action_space=action_space_type).to(self.device)
                self.policies[a] = net
                self.optimizers[a] = optim.Adam(net.parameters(), lr=lr)

    # ------------------------------------------------------

    def _device(self):
        if self.config['device']['use_cuda'] and torch.cuda.is_available():
            return torch.device(f"cuda:{self.config['device']['gpu_id']}")
        return torch.device("cpu")

    # ------------------------------------------------------

    def select_actions(self, obs, global_obs=None):
        if self.shared:
            actions, logps, values = {}, {}, {}

            for a, o in obs.items():
                o = (
                    torch.tensor(
                        o,
                        dtype=torch.float32,
                        device=self.device
                    )
                ).unsqueeze(0)

                act, logp, val = self.policy.act(o, global_obs)

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

                act, logp, val = net.act(o, global_obs)

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
            ).to(self.device)

            if self.centralized_critic:
                global_obs = []
                for t in range(len(self.buffer.global_obs)):
                    for _ in self.agents:
                        global_obs.append(self.buffer.global_obs[t])

                global_obs = torch.stack(global_obs).to(self.device)

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

                    if self.centralized_critic:
                        mb_global_obs = global_obs[mb_idx]
                        new_log_probs, entropy, values = \
                            self.policy.evaluate(
                                mb_obs,
                                mb_actions,
                                mb_global_obs
                            )
                    else:
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

                if self.centralized_critic:
                    global_obs = []
                    for t in range(len(self.buffer.global_obs)):
                        for _ in self.agents:
                            global_obs.append(self.buffer.global_obs[t])

                    global_obs = torch.stack(global_obs).to(self.device)

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

                        if self.centralized_critic:
                            mb_global_obs = global_obs[mb_idx]
                            new_log_probs, entropy, values = \
                                policy.evaluate(
                                    mb_obs,
                                    mb_actions,
                                    mb_global_obs
                                )
                        else:
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

                        if loss.grad_fn is None:
                            print(f"[PPO WARNING] {a}: No gradient graph detected. Skipping update.")
                            return

                        optimizer.zero_grad()
                        loss.backward()

                        nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)

                        optimizer.step()


    # ======================================================
    # Training Loop
    # ======================================================

    def train(self, callback=None):

        timestep = 0

        episode_reward = 0.0

        ind_episode_reward = {a: 0.0 for a in self.agents}

        observations, infos = self.env.reset(seed=self.config['env']['seed'])

        self.buffer.clear()

        ###################################################
        if self.config['logging']['enabled']:
            # ensure folder exists 
            os.makedirs("results/PPO", exist_ok=True)
            # generate unique filename
            while True:
                rand_code = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
                filename = f"results/PPO/ppo_{self.config['env']['name']}_{rand_code}.csv"
                
                if not os.path.exists(filename):
                    break  # found unused name
                
            # logger
            log_file = open(filename, "w", newline="")
            writer = csv.writer(log_file)

            if self.log_type == "independent":
                header = ["timestep"] + [f"reward_{a}" for a in self.agents]
            elif self.log_type in ["mean", "max"]:
                header = ["timestep", "episode_reward"]
            writer.writerow(header)
        ###################################################

        iterator = trange(self.total_timesteps)
        for timestep in iterator:
            # ---------- Evaluation ----------
            '''
            if (self.eval_enabled and timestep % self.eval_interval == 0):

                avg_reward = self.evaluate(
                    episodes=self.eval_episodes
                )

                print(
                    f'[Evaluation] timestep={timestep} '
                    f'avg_reward={avg_reward:.2f}'
                )
                observations, infos = self.env.reset(seed=self.config['env']['seed'])
            '''
            # --------------------------------

            # -------- Checkpointing ---------
            '''
            if timestep % self.save_model_interval == 0 and timestep > 0:

                self.save(
                    f'checkpoints/checkpoint_{timestep}.pt'
                )

                print(f'[Checkpoint] saved at timestep={timestep}')
            '''
            # --------------------------------

            if self.centralized_critic:
                global_obs = torch.tensor(
                    self.build_global_obs(observations),
                    dtype=torch.float32,
                    device=self.device
                ).unsqueeze(0)
                actions, log_probs, values = \
                    self.select_actions(observations, global_obs)
            else:
                actions, log_probs, values = \
                    self.select_actions(observations)
                # print(f'Actions: {actions}')

            next_obs, rewards, terminations, truncations, infos = \
                self.env.step(actions)
            # print(f'Rewards: {rewards}')

            done_dict = {
                agent: (
                    terminations.get(agent, True)
                    or truncations.get(agent, True)
                )
                for agent in observations.keys()
            }

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

            if self.centralized_critic:
                global_obs = self.build_global_obs(observations)
                self.buffer.global_obs.append(global_obs)

            if all(done_dict.values()):
                observations, _ = self.env.reset(seed=self.config['env']['seed'])
            else:
                observations = next_obs

            if self.log_type == "mean":
                episode_reward += np.mean(list(rewards.values()))
            elif self.log_type == "max":
                episode_reward += np.max(list(rewards.values()))
            elif self.log_type == "independent":
                for a, r in rewards.items():
                    ind_episode_reward[a] += r

            if timestep % self.log_interval == 0:
                if self.log_type == "independent":
                    print(
                        f'[PPO] timestep={timestep} '
                        f'episode_reward={ind_episode_reward}'
                    )
                    if callback:
                        results = {
                            "timestep": timestep,
                            "episode_reward": {
                                a: ind_episode_reward[a]
                                for a in self.agents
                            }
                        }
                        callback(results)
                    if self.config['logging']['enabled']:
                        writer.writerow([timestep] + [ind_episode_reward[a] for a in self.agents])
                    ind_episode_reward = {a: 0.0 for a in self.agents}
                    
                else:
                    print(
                        f'[PPO] timestep={timestep} '
                        f'episode_reward={episode_reward:.2f}'
                    )
                    if callback:
                        results = {
                            "timestep": timestep,
                            "episode_reward": episode_reward
                        }
                        callback(results)
                    if self.config['logging']['enabled']:
                        writer.writerow([timestep, episode_reward])
                    episode_reward = 0.0

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

            if self.log_type == "independent":
                desc = 'Timestep:{} Return:{}'.format(timestep, ' '.join('{:0.6f}' for reward in ind_episode_reward.values()).format(*ind_episode_reward.values()))
            else:
                desc = 'Timestep:{} Return:{:0.6f}'.format(timestep, episode_reward)
            iterator.set_description(desc)

    # ======================================================
    # Evaluation
    # ======================================================

    def evaluate(self, episodes=5):
        rewards = []
        mean_rewards = {}

        for _ in range(episodes):
            observations, infos = self.env.reset()

            done = False
            ind_episode_reward = {a: 0.0 for a in self.agents}

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

                #print(actions)
                observations, reward, terminations, truncations, infos = \
                    self.env.step(actions)

                for a, r in reward.items():
                    ind_episode_reward[a] += r

                done = all(
                    terminations[a] or truncations[a]
                    for a in observations.keys()
                )

            rewards.append(ind_episode_reward)

        for a in self.agents:
            score = np.mean([ep[a] for ep in rewards])
            mean_rewards[a] = score
        print(f'[Evaluation] mean_rewards={mean_rewards}')
        return mean_rewards
    
    def build_global_obs(self, obs_dict):

        features = []

        with torch.no_grad():

            for a in self.agents:

                obs = torch.tensor(
                    obs_dict[a],
                    dtype=torch.float32,
                    device=self.device
                ).unsqueeze(0)

                if self.shared:
                    z = self.policy.encode(obs)
                else:
                    z = self.policies[a].encode(obs)

                features.append(z)

        global_obs = torch.cat(features, dim=-1)

        return global_obs.squeeze(0)
    
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
