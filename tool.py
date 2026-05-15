from utils import *

if __name__ == "__main__":
    generate_enemy_from_psro_checkpoint(
        path="checkpoints/psro_boxing_v2_143vlk/populations_t15.pth", 
        oracle='ppo', 
        agent_name='second_0')