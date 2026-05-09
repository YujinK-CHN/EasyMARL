import yaml
import argparse
from scipy.io import savemat

from utils import make_env


# ==========================================================
# Config Loader
# ==========================================================

def load_config(path):
    with open(path, 'r') as fp:
        config = yaml.safe_load(fp)

    return config


# ==========================================================
# Main
# ==========================================================

def main():

    # ------------------------------------------------------
    # CLI Arguments
    # ------------------------------------------------------

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--algo',
        type=str,
        default='ppo'
    )

    parser.add_argument(
        '--env',
        type=str,
        default='boxing'
    )

    args = parser.parse_args()

    # ------------------------------------------------------
    # Load Configs
    # ------------------------------------------------------

    algo_config = load_config(
        f'configs/algos/{args.algo}.yaml'
    )

    env_config = load_config(
        f'configs/envs/{args.env}.yaml'
    )

    # Merge configs
    config = {
        **env_config,
        **algo_config
    }

    # ------------------------------------------------------
    # Experiment Loop
    # ------------------------------------------------------
    repeat = 1
    for exp_id in range(repeat):

        print(
            '[+] %s - %d/%d'
            % (
                args.algo.upper(),
                exp_id + 1,
                repeat
            )
        )

        # --------------------------------------------------
        # Build Environment
        # --------------------------------------------------

        env = make_env(config)

        # --------------------------------------------------
        # Build Algorithm
        # --------------------------------------------------

        if args.algo == 'ppo':

            from algorithms.ppo import PPO

            agent = PPO(config, env)

        else:
            raise NotImplementedError(
                f'Algorithm {args.algo} not implemented'
            )

        # --------------------------------------------------
        # Train
        # --------------------------------------------------

        agent.train()

        # --------------------------------------------------
        # Close Environment
        # --------------------------------------------------

        env.close()


if __name__ == '__main__':
    main()