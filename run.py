import yaml
import argparse
from utils import make_env

def callback(res):
    pass

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
        default='cooperative_pong'
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

    if algo_config['algorithm']['name'] == 'psro':
        oracle_config = load_config(
            f'configs/algos/{algo_config["algorithm"]["oracle_algorithm"]}.yaml'
        )
        # Merge configs
        config = {
            **env_config,
            **algo_config,
            'training': oracle_config['training'],
            'evaluation': oracle_config['evaluation'],
            'logging': oracle_config['logging']
        }
    else:
        # Merge configs
        config = {
            **env_config,
            **algo_config
        }
    print(config)

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

        elif args.algo == 'qmix':

            from algorithms.qmix import QMIX

            agent = QMIX(config, env)

        elif args.algo == 'psro':

            from algorithms.psro import PSRO

            agent = PSRO(config, env)
            
        else:
            raise NotImplementedError(
                f'Algorithm {args.algo} not implemented'
            )

        # --------------------------------------------------
        # Train
        # --------------------------------------------------

        agent.train(callback=callback)

        # --------------------------------------------------
        # Close Environment
        # --------------------------------------------------

        env.close()


if __name__ == '__main__':
    main()