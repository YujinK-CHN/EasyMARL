import torch
import os
# ==============================================
# Build Environment
# ==============================================

def make_env(config):

    env_family = config['env']['family']
    env_name = config['env']['name']
    env_kwargs = config['env'].get('kwargs', {})
    use_parallel = config['env'].get('parallel', True)

    # ======================================================
    # MPE
    # ======================================================

    if env_family == 'mpe':

        from pettingzoo.mpe import (
            simple_spread_v3,
            simple_tag_v3,
            simple_adversary_v3,
            simple_reference_v3,
            simple_speaker_listener_v4,
        )

        env_map = {
            'simple_spread_v3': simple_spread_v3,
            'simple_tag_v3': simple_tag_v3,
            'simple_adversary_v3': simple_adversary_v3,
            'simple_reference_v3': simple_reference_v3,
            'simple_speaker_listener_v4': simple_speaker_listener_v4,
        }

    # ======================================================
    # Butterfly
    # ======================================================

    elif env_family == 'butterfly':

        from pettingzoo.butterfly import (
            cooperative_pong_v6,
            pistonball_v6,
            knights_archers_zombies_v10,
        )

        env_map = {
            'cooperative_pong_v6': cooperative_pong_v6,
            'pistonball_v6': pistonball_v6,
            'knights_archers_zombies_v10': knights_archers_zombies_v10,
        }

    # ======================================================
    # SISL
    # ======================================================

    elif env_family == 'sisl':

        from pettingzoo.sisl import (
            waterworld_v4,
            pursuit_v4,
        )

        env_map = {
            'waterworld_v4': waterworld_v4,
            'pursuit_v4': pursuit_v4,
        }

    # ======================================================
    # Atari
    # ======================================================

    elif env_family == 'atari':

        from pettingzoo.atari import (
            basketball_pong_v3,
            boxing_v2,
            combat_tank_v2,
            combat_plane_v2,
            flag_capture_v2,
            pong_v3,
        )

        env_map = {
            'basketball_pong_v3': basketball_pong_v3,
            'boxing_v2': boxing_v2,
            'combat_tank_v2': combat_tank_v2,
            'combat_plane_v2': combat_plane_v2,
            'flag_capture_v2': flag_capture_v2,
            'pong_v3': pong_v3
        }

    else:
        raise ValueError(
            f'Unknown environment family: {env_family}'
        )

    # ======================================================
    # Build Environment
    # ======================================================
    print(env_name)
    if env_name not in env_map:
        raise ValueError(
            f'Unknown environment: {env_name}'
        )

    env_module = env_map[env_name]

    if use_parallel:
        env = env_module.parallel_env(**env_kwargs)
    else:
        env = env_module.env(**env_kwargs)

    return env

def generate_enemy_from_ppo_checkpoint(path, agent_name=None):
    checkpoint = torch.load(path)
    if agent_name != None:
        torch.save(checkpoint['models'][agent_name])
    else:
        for a in checkpoint['models'].key():
            torch.save(checkpoint['models'][a])
            break

def generate_enemy_from_psro_checkpoint(path, oracle=None, agent_name=None):

    if oracle == None:
        print('[Tool Warning] You need to specify the oracle algorithm for PSRO.')
    else:
        save_dir = f"checkpoints/{oracle}_enemy/"
        os.makedirs(save_dir, exist_ok=True)
        checkpoint = torch.load(path)
        #print(checkpoint[agent_name][0])
        if agent_name != None:
            torch.save(checkpoint[agent_name][0], save_dir + 'enemy_agent.pt')
            print(f'[Tool] enemy save to: {save_dir}')
        else:
            for a in checkpoint.keys():
                torch.save(checkpoint[a][0], save_dir + 'enemy_agent.pt')
                break

