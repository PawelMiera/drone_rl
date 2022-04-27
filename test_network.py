import os
import numpy as np
import torch
from flightgym import VisionEnv_v1
from ruamel.yaml import YAML, RoundTripDumper, dump
from stable_baselines3.common.utils import get_device
from stable_baselines3.ppo.policies import MlpPolicy
from rpg_baselines.torch.envs import vec_env_wrapper as wrapper
from stable_baselines3 import PPO


def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def random_flight(env, render=False, num_rollouts=2000):
    obs_dim = env.obs_dim
    act_dim = env.act_dim
    num_env = env.num_envs
    max_ep_length = env.max_episode_steps
    frame_id = 0
    current_min = 0
    current_max = 0
    import random

    if render:
        env.connectUnity()
    for n_roll in range(num_rollouts):
        obs, done, ep_len = env.reset(), False, 0
        while not (done or (ep_len >= max_ep_length)):
            #dummy_actions = np.random.rand(1, act_dim) * 2 - np.ones(shape=(num_env, act_dim))
            #dummy_actions[0][0] = dummy_actions[0][0] / 10
            #dummy_actions[0][0] = dummy_actions[0][0] / 10

            #dummy_actions = np.array([[random.uniform(-0.5,0.2),random.uniform(-1,1),random.uniform(-0.6,0.85),random.uniform(-1,1)]])

            dummy_actions = np.array([[0.1, 0.0, -1.0, 0.0]])

            obs, rew, done, info = env.step(dummy_actions)
            env.render(ep_len)
            ep_len += 1
            frame_id += 1

    env.save_array()
    if render:
        env.disconnectUnity()


def test_policy(env, model, render=False, num_rollouts=10):
    max_ep_length = env.max_episode_steps
    frame_id = 0
    current_min = 0
    current_max = 0
    if render:
        env.connectUnity()
    for n_roll in range(num_rollouts):

        obs, done, ep_len = env.reset(), False, 0
        print("NEXT")
        while not (done or (ep_len >= max_ep_length)):
            #print(obs)
            act, _ = model.predict(obs, deterministic=True)
            obs, rew, done, info = env.step(act)

            """if min(obs[0]) < current_min:
                current_min = min(obs[0])
                print("min", current_min)

            if max(obs[0]) > current_max:
                current_max = max(obs[0])
                print("max", current_max)"""

            #
            env.render(ep_len)
            
            if done:
            	print("reewww ", rew)

            # ======Gray Image=========
            # gray_img = np.reshape(
            #     env.getImage()[0], (env.img_height, env.img_width))
            # cv2.imshow("gray_img", gray_img)
            # cv2.waitKey(100)

            # ======RGB Image=========
            # img =env.getImage(rgb=True)
            # rgb_img = np.reshape(
            #    img[0], (env.img_height, env.img_width, 3))
            # cv2.imshow("rgb_img", rgb_img)
            # os.makedirs("./images", exist_ok=True)
            # cv2.imwrite("./images/img_{0:05d}.png".format(frame_id), rgb_img)
            # cv2.waitKey(100)

            # # # ======Depth Image=========
            # depth_img = np.reshape(env.getDepthImage()[
            #                        0], (env.img_height, env.img_width))
            # os.makedirs("./depth", exist_ok=True)
            # cv2.imwrite("./depth/img_{0:05d}.png".format(frame_id), depth_img.astype(np.uint16))
            # cv2.imshow("depth", depth_img)
            # cv2.waitKey(100)

            #
            ep_len += 1
            frame_id += 1

    #
    if render:
        env.disconnectUnity()


def main():
    random_seed = False
    render = True
    #policy_path = 'saved/PPO_73'
    policy_path = "/home/pawel/agh_code/saved/PPO_238"
    policy_number = 350
    
    flightmare_path = "/home/pawel/agile_flight/flightmare"

    cfg = YAML().load(open("configs/test_config.yaml"))

    cfg["simulation"]["num_envs"] = 1

    train_env = VisionEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
    train_env = wrapper.FlightEnvVec(train_env)

    # set random seed
    configure_random_seed(random_seed, env=train_env)

    if render:
        cfg["unity"]["render"] = "yes"


    eval_env = wrapper.FlightEnvVec(
        VisionEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
    )

    # save the configuration and other files
    rsg_root = os.path.dirname(os.path.abspath(__file__))
    log_dir = rsg_root + "/saved"
    os.makedirs(log_dir, exist_ok=True)

    os.system(flightmare_path + "/flightrender/RPG_Flightmare.x86_64 &")

    weight_path = os.path.join(policy_path, os.path.join("Policy", 'iter_{:05d}.pth'.format(policy_number)))

    env_rms_path = os.path.join(policy_path, os.path.join("RMS", 'iter_{:05d}.npz'.format(policy_number)))

    #print(weight_path, env_rms_path)

    device = get_device("auto")
    saved_variables = torch.load(weight_path, map_location=device)
    # Create policy object
    policy = MlpPolicy(**saved_variables["data"])
    #
    policy.action_net = torch.nn.Sequential(policy.action_net, torch.nn.Tanh())
    # Load weights
    policy.load_state_dict(saved_variables["state_dict"], strict=False)
    policy.to(device)

    eval_env.load_rms(env_rms_path)
    test_policy(eval_env, policy, render=render)
    #random_flight(eval_env, True)


if __name__ == "__main__":
    main()
