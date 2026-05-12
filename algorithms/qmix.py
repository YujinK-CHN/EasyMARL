import os
import string
import csv
import numpy as np
from tqdm import trange
import torch
import torch.nn as nn
import torch.optim as optim
import random
import torch.nn.functional as F


# ==========================================================
# Replay Buffer (Episode-based for QMIX)
# ==========================================================

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
# RNN-based Agent Network
# ==========================================================

class RNNAgent(nn.Module):
    def __init__(self, obs_shape, n_actions, hidden_size):
        super().__init__()

        h, w, c = obs_shape

        # -------------------------
        # CNN encoder (Atari-style)
        # -------------------------
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # infer conv output size safely
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out_size = self.conv(dummy).shape[-1]

        # fully connected projection before GRU
        self.fc = nn.Linear(conv_out_size, hidden_size)

        # recurrent layer
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)

        self.fc_q = nn.Linear(hidden_size, n_actions)

    def forward(self, obs, hidden_state):

        b, t = obs.shape[:2]

        # merge batch/time
        x = obs.reshape(b * t, *obs.shape[2:])

        # HWC -> CHW
        x = x.permute(0, 3, 1, 2)

        x = self.conv(x)

        x = self.fc(x)
        x = torch.relu(x)

        x = x.view(b, t, -1)

        x, h = self.rnn(x, hidden_state)

        q = self.fc_q(x)

        return q, h

    def init_hidden(self, batch_size):
        return torch.zeros(
            1, batch_size, self.rnn.hidden_size,
            device=next(self.parameters()).device
        )


# ==========================================================
# QMIX Mixing Network
# ==========================================================

class QMixer(nn.Module):
    def __init__(self, n_agents, state_dim, mixing_hidden_dim, hypernet_hidden_dim):
        super().__init__()

        self.n_agents = n_agents
        self.state_dim = state_dim

        # Hypernet for weights
        self.hyper_w1 = nn.Linear(state_dim, hypernet_hidden_dim * n_agents)
        self.hyper_w2 = nn.Linear(state_dim, hypernet_hidden_dim)

        self.hyper_b1 = nn.Linear(state_dim, hypernet_hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(hypernet_hidden_dim, 1)
        )

        self.V = nn.Sequential(
            nn.Linear(state_dim, mixing_hidden_dim),
            nn.ReLU(),
            nn.Linear(mixing_hidden_dim, 1)
        )

    def forward(self, q_values, state):
        """
        q_values: (batch, n_agents)
        state: (batch, state_dim)
        """

        bs = q_values.size(0)

        w1 = torch.abs(self.hyper_w1(state)).view(bs, self.n_agents, -1)
        b1 = self.hyper_b1(state).view(bs, 1, -1)

        w2 = torch.abs(self.hyper_w2(state)).view(bs, -1, 1)
        b2 = self.hyper_b2(state).view(bs, 1, 1)

        hidden = torch.relu(torch.bmm(q_values.unsqueeze(1), w1).squeeze(1) + b1.squeeze(1))
        y = torch.bmm(hidden.unsqueeze(1), w2).squeeze(1) + b2.squeeze(1)

        return y + self.V(state)


# ==========================================================
# QMIX Trainer
# ==========================================================

class QMIX:
    def __init__(self, config, env):
        self.config = config
        self.env = env

        self.device = torch.device(
            f"cuda:{config['device']['gpu_id']}"
            if config['device']['use_cuda'] and torch.cuda.is_available()
            else "cpu"
        )

        self.agents = env.possible_agents
        self.n_agents = len(self.agents)

        sample_obs = env.reset()[0][self.agents[0]]
        obs_shape = np.array(sample_obs).shape   # (C,H,W)
        self.total_timesteps = self.config['training']['total_timesteps']
        self.train_step_count = 0

        model_cfg = config["model"]
        mix_cfg = config["mixer"]
        train_cfg = config["training"]
        exp_cfg = config["exploration"]

        act_space = env.action_space(self.agents[0])

        if hasattr(act_space, "n"):
            self.action_space_type = "discrete"
            n_actions = act_space.n
        else:
            action_space_type = "continuous"
            n_actions = act_space.shape[0]

        print(f'Action space type: {self.action_space_type}, act_dim: {n_actions}')

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

        # --------------------------------------------------
        # Agent networks
        # --------------------------------------------------
        self.policies = nn.ModuleDict({
            a: RNNAgent(
                obs_shape,
                n_actions,
                model_cfg["rnn_hidden_size"]
            ).to(self.device)
            for a in self.agents
        })

        self.target_policies = nn.ModuleDict({
            a: RNNAgent(
                obs_shape,
                n_actions,
                model_cfg["rnn_hidden_size"]
            ).to(self.device)
            for a in self.agents
        })

        # copy weights
        for a in self.agents:
            self.target_policies[a].load_state_dict(self.policies[a].state_dict())

        # --------------------------------------------------
        # Mixer
        # --------------------------------------------------
        self.mixer = QMixer(
            self.n_agents,
            state_dim=np.prod(obs_shape) * self.n_agents,
            mixing_hidden_dim=mix_cfg["mixing_hidden_dim"],
            hypernet_hidden_dim=mix_cfg["hypernet_hidden_dim"]
        ).to(self.device)

        self.target_mixer = QMixer(
            self.n_agents,
            state_dim=np.prod(obs_shape) * self.n_agents,
            mixing_hidden_dim=mix_cfg["mixing_hidden_dim"],
            hypernet_hidden_dim=mix_cfg["hypernet_hidden_dim"]
        ).to(self.device)

        self.target_mixer.load_state_dict(self.mixer.state_dict())

        # --------------------------------------------------
        # Optimizer
        # --------------------------------------------------
        self.params = list(self.mixer.parameters())
        for net in self.policies.values():
            self.params += list(net.parameters())

        self.optimizer = optim.Adam(self.params, lr=train_cfg["learning_rate"])

        # --------------------------------------------------
        # Buffer
        # --------------------------------------------------
        self.buffer = RolloutBuffer()

        # --------------------------------------------------
        # Training config
        # --------------------------------------------------
        self.gamma = train_cfg["gamma"]
        self.batch_size = config["buffer"]["batch_size"]
        self.target_update_interval = train_cfg["target_update_interval"]

        # --------------------------------------------------
        # Exploration
        # --------------------------------------------------
        self.eps_start = exp_cfg["epsilon_start"]
        self.eps_end = exp_cfg["epsilon_end"]
        self.eps_decay = exp_cfg["epsilon_decay_steps"]

        self.step_count = 0

    # ======================================================
    # Epsilon schedule
    # ======================================================

    def epsilon(self):
        t = min(self.step_count / self.eps_decay, 1.0)
        return self.eps_start + t * (self.eps_end - self.eps_start)

    # ======================================================
    # Action selection
    # ======================================================

    def select_actions(self, obs, hidden_states):
        actions = {}
        new_hidden = {}

        eps = self.epsilon()

        for i, a in enumerate(self.agents):

            # ------------------------------------------------
            # DEBUG: inspect actual observation structure
            # ------------------------------------------------
            obs_np = np.array(obs[a])
            '''
            print("================================")
            print("Agent:", a)
            print("Observation type:", type(obs[a]))
            print("Observation shape:", obs_np.shape)
            print("Observation dtype:", obs_np.dtype)

            # optional:
            print("Min:", obs_np.min())
            print("Max:", obs_np.max())
            '''
            # ------------------------------------------------
            # Convert to tensor
            # ------------------------------------------------
            obs_t = torch.tensor(
                obs_np,
                dtype=torch.float32,
                device=self.device
            ) / 255.0

            # add batch and sequence dimensions
            obs_t = obs_t.unsqueeze(0).unsqueeze(0)

            q, h = self.policies[a](obs_t, hidden_states[a])

            q = q.squeeze(0).squeeze(0)

            if random.random() < eps:
                act = random.randint(0, q.shape[-1] - 1)
            else:
                act = torch.argmax(q).item()

            actions[a] = act
            new_hidden[a] = h.detach()

        return actions, new_hidden

    # ======================================================
    # Build global state
    # ======================================================

    def build_state(self, obs):
        state = np.concatenate(
            [obs[a] for a in self.agents],
            axis=-1
        )

        return state.flatten().astype(np.float32) / 255.0

    # ======================================================
    # Store episode
    # ======================================================

    def store_episode(self, episode):
        self.buffer.push(episode)

    # ======================================================
    # Learning step
    # ======================================================

    def update(self):

        device = self.device
        gamma = self.config["training"]["gamma"]

        # ============================================================
        # Build batch tensors
        # ============================================================

        T = len(self.buffer.obs)
        n_agents = len(self.agents)

        obs_batch = []
        next_obs_batch = []
        action_batch = []
        reward_batch = []
        done_batch = []

        for t in range(T):

            obs_t = []
            next_obs_t = []
            actions_t = []

            for a in self.agents:
                obs_t.append(self.buffer.obs[t][a])

                # next obs
                if t < T - 1:
                    next_obs_t.append(self.buffer.obs[t + 1][a])
                else:
                    next_obs_t.append(self.buffer.obs[t][a])

                actions_t.append(self.buffer.actions[t][a])

            # use mean team reward
            reward = np.mean(list(self.buffer.rewards[t].values()))

            # episode done if all agents done
            done = float(all(self.buffer.dones[t].values()))

            obs_batch.append(obs_t)
            next_obs_batch.append(next_obs_t)
            action_batch.append(actions_t)
            reward_batch.append(reward)
            done_batch.append(done)

        obs_batch = torch.FloatTensor(
            np.array(obs_batch)
        ).to(device)                           # [T, n_agents, obs_dim]

        next_obs_batch = torch.FloatTensor(
            np.array(next_obs_batch)
        ).to(device)

        action_batch = torch.LongTensor(
            np.array(action_batch)
        ).to(device)                           # [T, n_agents]

        reward_batch = torch.FloatTensor(
            np.array(reward_batch)
        ).to(device)                           # [T]

        done_batch = torch.FloatTensor(
            np.array(done_batch)
        ).to(device)                           # [T]

        # ============================================================
        # Per-agent Q-values
        # ============================================================

        chosen_action_qvals = []
        target_max_qvals = []

        for i, agent in enumerate(self.agents):

            agent_net = self.policies[agent]
            target_agent_net = self.target_policies[agent]

            # ------------------------------------------------
            # Hidden states
            # ------------------------------------------------

            hidden = agent_net.init_hidden(1).to(device)
            target_hidden = target_agent_net.init_hidden(1).to(device)

            # ------------------------------------------------
            # Inputs
            # QMIX RNN expects:
            # [batch, time, obs_dim]
            # ------------------------------------------------

            agent_obs = obs_batch[:, i].unsqueeze(0)
            target_obs = next_obs_batch[:, i].unsqueeze(0)

            # Shapes:
            # [1, T, obs_dim]

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            q_values, _ = agent_net(agent_obs, hidden)
            target_q_values, _ = target_agent_net(
                target_obs,
                target_hidden
            )

            # Remove batch dimension
            # [1, T, n_actions] -> [T, n_actions]

            q_values = q_values.squeeze(0)
            target_q_values = target_q_values.squeeze(0)

            # ------------------------------------------------
            # Chosen Q-values
            # ------------------------------------------------

            chosen_q = torch.gather(
                q_values,
                dim=1,
                index=action_batch[:, i].unsqueeze(1)
            ).squeeze(1)

            # ------------------------------------------------
            # Target max Q-values
            # ------------------------------------------------

            target_max_q = target_q_values.max(dim=1)[0]

            chosen_action_qvals.append(chosen_q)
            target_max_qvals.append(target_max_q)

        # [T, n_agents]
        chosen_action_qvals = torch.stack(
            chosen_action_qvals,
            dim=1
        )

        target_max_qvals = torch.stack(
            target_max_qvals,
            dim=1
        )

        # ============================================================
        # Global state for mixer
        # ============================================================

        global_states = obs_batch.reshape(T, -1)
        next_global_states = next_obs_batch.reshape(T, -1)

        # ============================================================
        # Mixing network
        # ============================================================

        mixed_q = self.mixer(
            chosen_action_qvals,
            global_states
        ).squeeze(-1)

        with torch.no_grad():

            target_mixed_q = self.target_mixer(
                target_max_qvals,
                next_global_states
            ).squeeze(-1)

            targets = (
                reward_batch
                + gamma * (1.0 - done_batch) * target_mixed_q
            )

        # ============================================================
        # Loss
        # ============================================================

        loss = F.mse_loss(mixed_q, targets)

        # ============================================================
        # Optimize
        # ============================================================

        self.optimizer.zero_grad()

        loss.backward()

        for agent in self.agents:
            agent_net = self.policies[agent]
            no_grad = not any(
                param.grad is not None
                for param in agent_net.parameters()
            )
            if no_grad:
                print(f"[QMIX WARNING] {agent}: No gradients detected.")


        torch.nn.utils.clip_grad_norm_(
            self.mixer.parameters(),
            self.config["training"]["grad_norm_clip"]
        )

        for a in self.agents:
            torch.nn.utils.clip_grad_norm_(
                self.policies[a].parameters(),
                self.config["training"]["grad_norm_clip"]
            )

        self.optimizer.step()

        # ============================================================
        # Target update
        # ============================================================

        if self.train_step_count % self.target_update_interval == 0:

            self.target_mixer.load_state_dict(
                self.mixer.state_dict()
            )

            for a in self.agents:
                self.target_policies[a].load_state_dict(
                    self.policies[a].state_dict()
                )

        self.train_step_count += 1

    # ======================================================
    # Main training loop (PettingZoo parallel)
    # ======================================================

    def train(self, callback=None):

        timestep = 0

        episode_reward = 0.0

        ind_episode_reward = {a: 0.0 for a in self.agents}
        
        observations, _ = self.env.reset(seed=self.config['env']['seed'])

        self.buffer.clear()

        hidden = {
            a: self.policies[a].init_hidden(1)
            for a in self.agents
        }

        ###################################################
        if self.config['logging']['enabled']:
            # ensure folder exists 
            os.makedirs("results/QMIX", exist_ok=True)
            # generate unique filename
            while True:
                rand_code = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
                filename = f"results/QMIX/qmix_{self.config['env']['name']}_{rand_code}.csv"
                
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

            actions, hidden = self.select_actions(observations, hidden)

            next_obs, rewards, terminations, truncations, _ = self.env.step(actions)

            done_dict = {
                agent: (
                    terminations.get(agent, True)
                    or truncations.get(agent, True)
                )
                for agent in observations.keys()
            }

            self.buffer.obs.append(observations)
            self.buffer.actions.append(actions)
            self.buffer.rewards.append(rewards)
            self.buffer.dones.append(done_dict)

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
                        f'[QMIX] timestep={timestep} '
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
                        f'[QMIX] timestep={timestep} '
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

            # -------- Compute update policy ---------
            if timestep % self.config["buffer"]["capacity"] == 0 and timestep > 0:

                self.update()

                self.buffer.clear()

                hidden = {
                    a: self.policies[a].init_hidden(1)
                    for a in self.agents
                }

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

        for policy in self.policies.values():
            policy.eval()

        for _ in range(episodes):
            observations, infos = self.env.reset()

            done = False
            episode_reward = {a: 0.0 for a in self.agents}

            # ✅ INIT HIDDEN STATE (IMPORTANT FIX)
            hidden = {
                a: self.policies[a].init_hidden(1).to(self.device)
                for a in self.agents
            }

            while not done:
                actions = {}

                with torch.no_grad():
                    for agent, obs in observations.items():

                        obs_tensor = (
                            torch.FloatTensor(obs)
                            .unsqueeze(0)
                            .unsqueeze(0)
                            .to(self.device)
                        )

                        policy = self.policies[agent]

                        # ✅ FIX: pass hidden state
                        q_values, h = policy(obs_tensor, hidden[agent])

                        q_values = q_values.squeeze(0).squeeze(0)

                        action = torch.argmax(q_values, dim=-1).item()

                        actions[agent] = action

                        # ✅ update hidden state
                        hidden[agent] = h

                observations, reward, terminations, truncations, infos = \
                    self.env.step(actions)

                for a, r in reward.items():
                    episode_reward[a] += r

                done = all(
                    terminations[a] or truncations[a]
                    for a in terminations.keys()
                )

            rewards.append(episode_reward)

        for a in self.agents:
            mean_rewards[a] = np.mean([ep[a] for ep in rewards])

        print(f"[Evaluation] mean_rewards={mean_rewards}")

        for policy in self.policies.values():
            policy.train()

        return mean_rewards