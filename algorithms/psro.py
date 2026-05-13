import os
import copy
import random
import csv
from tomlkit import string
from run import load_config
import torch
import numpy as np
from collections import defaultdict
from itertools import product
from algorithms.ppo import PPO
from algorithms.qmix import QMIX


# ==========================================================
# PSRO
# ==========================================================

class PSRO:

    def __init__(self, config, env):

        self.config = config
        self.env = env

        self.device = self._device()

        self.agents = env.possible_agents

        # ======================================================
        # PSRO Config
        # ======================================================

        psro_cfg = config["psro"]

        self.iterations = psro_cfg["iterations"]
        self.oracle_training_steps = psro_cfg["oracle_training_steps"]
        self.max_population_size = psro_cfg["max_population_size"]
        self.meta_solver = psro_cfg["meta_solver"]
        self.response_sampling = psro_cfg["response_sampling"]
        self.add_only_if_improved = psro_cfg["add_only_if_improved"]
        self.payoff_gamma = psro_cfg["payoff_gamma"]

        # ======================================================
        # Logging / Evaluation
        # ======================================================
        self.timestep = 0
        self.rand_code = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        self.log_enabled = config["psro_logging"]["enabled"]
        self.log_interval = config["psro_logging"]["log_interval"]
        self.save_population_interval = config["psro_logging"]["save_population_interval"]

        self.eval_enabled = config["psro_evaluation"]["enabled"]
        self.eval_episodes = config["psro_evaluation"]["eval_episodes"]

        self.score_history = {
            a: [0.0] for a in self.agents
        }

        # ======================================================
        # Population
        # ======================================================

        # each agent has a population of policies
        self.population = {
            a: [] for a in self.agents
        }

        # ======================================================
        # Oracle learner
        # ======================================================

        oracle_algo = config["algorithm"]["oracle_algorithm"]
        print(f"[PSRO] Oracle algorithm: {oracle_algo}")

        if oracle_algo.lower() == "ppo":
            self.oracle = PPO(self.config, env)
            self.oracle.total_timesteps = self.oracle_training_steps
        elif oracle_algo.lower() == "qmix":
            qmix_config = load_config(
                f'configs/algos/qmix.yaml')
            self.config ={
                **self.config,
                'model': qmix_config['model'],
                'mixer': qmix_config['mixer'],
                'buffer': qmix_config['buffer'],
                'exploration': qmix_config['exploration']
            }
            self.oracle = QMIX(self.config, env)
            self.oracle.total_timesteps = self.oracle_training_steps
        else:
            raise ValueError(
                f"Unsupported oracle algorithm: {oracle_algo}"
            )

        # ======================================================
        # Initialize first population policy
        # ======================================================

        self._initialize_population()
    
    # ==========================================================
    # Device
    # ==========================================================

    def _device(self):

        if (
            self.config["device"]["use_cuda"]
            and torch.cuda.is_available()
        ):
            return torch.device(
                f'cuda:{self.config["device"]["gpu_id"]}'
            )

        return torch.device("cpu")

    # ==========================================================
    # Initialize Population
    # ==========================================================

    def _initialize_population(self):

        print("[PSRO] Initializing population...")

        if self.config["algorithm"]["oracle_algorithm"].lower() == "ppo" and self.oracle.shared:

            print("[PSRO] cannot work with shared policy.")

        else:

            for agent in self.agents:

                initial_policy = copy.deepcopy(
                    self.oracle.policies[agent].state_dict()
                )

                self.population[agent].append(initial_policy)
    
    def train(self, callback=None):

        print("[PSRO] Starting training...")

        for iteration in range(self.iterations):

            print(f"\n[PSRO] Iteration {iteration}")

            ###################################################
            if self.log_enabled:
                # ensure folder exists 
                os.makedirs("results/PSRO", exist_ok=True)
                # generate unique filename
                while True:
                    
                    filename = f"results/PSRO/psro_{self.config['env']['name']}_{self.rand_code}.csv"
                    
                    if not os.path.exists(filename):
                        break  # found unused name
                    
                # logger
                log_file = open(filename, "w", newline="")
                writer = csv.writer(log_file)

                header = ["iteration"] + ["timestep"] + [f"reward_{a}" for a in self.agents]
                writer.writerow(header)
            ###################################################

            payoff_matrix = self.get_payoff_matrix(self.eval_episodes)
            print(f"[PSRO] Payoff matrix:\n{payoff_matrix}")

            # --------------------------------------------------
            # Sample opponents
            # --------------------------------------------------
            for agent in self.agents:
                print(f"  {agent} population size: {len(self.population[agent])}")

            sampled_response, sampled_idx = self.sample_response(payoff_matrix=payoff_matrix)

            print(f"[PSRO] Sampled opponents: {sampled_idx}")
            # --------------------------------------------------
            # Train oracle
            # --------------------------------------------------

            oracle_info, new_policies = self.train_oracle(sampled_response)
            
            self.timestep += self.oracle_training_steps

            print(f"[PSRO] Oracle info: {oracle_info}")
            ##################################################
            if self.log_enabled and iteration % self.log_interval == 0:
                row = [iteration, self.timestep] + [oracle_info[a] for a in self.agents]
                writer.writerow(row)
                log_file.flush()
            ##################################################

            # --------------------------------------------------
            # Add oracle to population
            # --------------------------------------------------

            improved = self.should_add_oracle(oracle_info, sampled_idx)

            self.add_oracle_to_population(improved, new_policies)

            print("[PSRO] Score history:", self.score_history)

            # -------- Checkpointing ---------
            ''''''
            if self.config['logging']['enable_saving'] and \
                 iteration % self.save_population_interval == 0 and iteration > 0:

                path = f'checkpoints/{self.config["algorithm"]["name"]}_{self.config["env"]["name"]}_{self.rand_code}'
                os.makedirs(path, exist_ok=True)

                self.save(path = path + f'/populations_t{iteration}.pth')
                
                print(f'[Checkpoint] saved at timestep={iteration}')
            
            # --------------------------------


    # ==========================================================
    # Sample Opponents
    # ==========================================================

    def sample_response(self, payoff_matrix=None):

        sampled = {}
        sampled_idx = {}

        for i, agent in enumerate(self.agents):

            pop_size = len(self.population[agent])

            # ----------------------------------------------
            # Uniform sampling
            # ----------------------------------------------

            if self.response_sampling == "uniform":

                idx = random.randint(0, pop_size - 1)

            # ----------------------------------------------
            # Latest policy
            # ----------------------------------------------

            elif self.response_sampling == "latest":

                idx = pop_size - 1

            # ----------------------------------------------
            # Meta strategy
            # ----------------------------------------------

            elif self.response_sampling == "meta_strategy":

                meta_strateies = self.compute_meta_strategies(payoff_matrix=payoff_matrix)
                print(f"[PSRO] Meta strategies: {meta_strateies}")

                idx = np.random.choice(
                    np.arange(pop_size),
                    p=meta_strateies[agent]
                )

            else:
                raise ValueError(
                    f"Unknown response sampling: "
                    f"{self.response_sampling}"
                )

            sampled_idx[agent] = idx
            sampled[agent] = self.population[agent][idx]
        return sampled, sampled_idx

    def train_oracle(self, sampled_response):

        oracle_info = {}

        new_policies = {}

        def psro_callback(info):
            """
            PPO calls this during training.
            We only store summary stats.
            """
            episode_stats.append(info["episode_reward"])

        for i, agent in enumerate(self.agents):
            learning_agent = sampled_response[agent]
            opponent_agent = sampled_response[self.agents[1] if i == 0 else self.agents[0]]

            # ======================================================
            # 2. Inject opponent into PPO (frozen behavior)
            # ======================================================

            self.oracle.policies[self.agents[0]].load_state_dict(learning_agent)

            # IMPORTANT: ensure oracle optimize a0
            for p in self.oracle.policies[self.agents[0]].parameters():
                p.requires_grad = True

            self.oracle.policies[self.agents[1]].load_state_dict(opponent_agent)

            # IMPORTANT: ensure oracle does NOT try to optimize a1
            for p in self.oracle.policies[self.agents[1]].parameters():
                p.requires_grad = False

            episode_stats = []

            # ======================================================
            # 4. Run PPO training (oracle = best response)
            # ======================================================

            self.oracle.train(callback=psro_callback)

            new_policies[agent] = copy.deepcopy(
                self.oracle.policies[self.agents[0]].state_dict()
            )

            score = np.mean([ep[self.agents[0]] for ep in episode_stats])

            oracle_info[agent] = score

        return oracle_info, new_policies
    
    def should_add_oracle(self, oracle_info, sampled_idx):

        improved = {}

        for i, agent in enumerate(self.agents):

            new_score = oracle_info[agent]
            old_score = self.score_history[agent][sampled_idx[agent]]

            if new_score > old_score:
                improved[agent] = True
                self.score_history[agent][sampled_idx[agent]] = new_score
            else:
                improved[agent] = False

        return improved

    def add_oracle_to_population(self, improved, new_policies):

        for i, agent in enumerate(self.agents):

            if improved[agent]:

                self.population[agent].append(new_policies[agent])
                if i == 0:
                    self.score_history[self.agents[1]].append(0.0)
                else:
                    self.score_history[self.agents[0]].append(0.0)

                self.prune_population(agent)
                print(f"[PSRO] Added oracle to population for {agent}.")

    def get_payoff_matrix(self, episodes=5):

        pop0 = self.population[self.agents[0]]
        pop1 = self.population[self.agents[1]]

        payoff_matrix = {agent: np.zeros((len(pop0), len(pop1))) for agent in self.agents}

        for i, policy0 in enumerate(pop0):

            for j, policy1 in enumerate(pop1):

                # load policies
                self.oracle.policies[self.agents[0]].load_state_dict(policy0)
                self.oracle.policies[self.agents[1]].load_state_dict(policy1)

                # evaluate matchup
                rewards = self.oracle.evaluate(episodes=episodes)

                # store payoff
                payoff_matrix[self.agents[0]][i, j] = rewards[self.agents[0]]
                payoff_matrix[self.agents[1]][i, j] = rewards[self.agents[1]]

        return payoff_matrix
    

    def compute_meta_strategies(self, payoff_matrix):
        """
        Compute PSRO meta-strategies (mixed Nash equilibrium approximation)
        from a 2-player payoff matrix.

        Args:
            payoff_matrix: dict
                {
                    agent1_name: np.array (n x m),
                    agent2_name: np.array (n x m)
                }

        Returns:
            dict: {
                agent1_name: np.array (n,),
                agent2_name: np.array (m,)
            }
        """

        A = payoff_matrix[self.agents[0]]
        B = payoff_matrix[self.agents[1]]

        n, m = A.shape

        # --- helper: compute best response sets ---
        def best_response_support(payoff_matrix, opponent_dist, axis):
            expected = payoff_matrix @ opponent_dist if axis == 1 else payoff_matrix.T @ opponent_dist
            max_val = np.max(expected)
            support = np.isclose(expected, max_val)
            return support.astype(float)

        # --- fallback: uniform if degenerate (common early PSRO behavior) ---
        if np.allclose(A, 0) and np.allclose(B, 0):
            return {
                self.agents[0]: np.ones(n) / n,
                self.agents[1]: np.ones(m) / m
            }

        # --- compute mixed strategies via simple iterative best-response averaging ---
        # (stable, widely used in PSRO prototypes)

        p = np.ones(n) / n
        q = np.ones(m) / m

        lr = 0.1
        for _ in range(200):  # fixed-point iteration
            # best response direction for player 1
            u1 = A @ q
            p_br = (u1 == np.max(u1)).astype(float)
            p_br /= p_br.sum()

            # best response direction for player 2
            u2 = B.T @ p
            q_br = (u2 == np.max(u2)).astype(float)
            q_br /= q_br.sum()

            # smooth update (stabilizes training)
            p = (1 - lr) * p + lr * p_br
            q = (1 - lr) * q + lr * q_br

            p /= p.sum()
            q /= q.sum()

        return {
            self.agents[0]: p,
            self.agents[1]: q
        }
    
    def prune_population(self, agent):
        if len(self.population[agent]) > self.max_population_size:
            # remove oldest policy (FIFO)
            self.population[agent].pop(0)

            # keep score history aligned
            self.score_history[agent].pop(0)
            print(f"[PSRO] Pruned population for {agent}. New size: {len(self.population[agent])}")

    def save(self, path):

        save_dict = {}

        for a in self.agents:

            save_dict[a] = [
                policy.state_dict()
                for policy in self.population[a]
            ]

        torch.save(save_dict, path)