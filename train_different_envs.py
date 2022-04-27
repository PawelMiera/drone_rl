import argparse
import os
import random
import select
import sys
import time

import numpy as np
import torch
import yaml
from flightgym import VisionEnv_v1
from rpg_baselines.torch.common.ppo import PPO
from rpg_baselines.torch.envs import vec_env_wrapper as wrapper
from ruamel.yaml import YAML, RoundTripDumper, dump
from stable_baselines3.common.utils import get_device


def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ask_edit_cfg_parameters(cfg, timesteps):
    i = True
    timeout = 10
    while i:
        print("\nType config group you want to edit (timeout", timeout, "s)[rewards] or type timesteps or new_cfg")
        i, o, e = select.select([sys.stdin], [], [], timeout)

        cfg_group = ""
        cfg_name = ""

        if i:
            timeout = 60
            cfg_group = sys.stdin.readline().strip()

            if cfg_group == "":
                cfg_group = "rewards"
            elif cfg_group == "timesteps":
                print("Current value: ", timesteps)
                print("\nType new value  (timeout 60s) enter empty for no change\n")

                i, o, e = select.select([sys.stdin], [], [], 60)

                if i:
                    value = sys.stdin.readline().strip()

                    if value == "":
                        print("No change")
                    else:
                        try:
                            value = int(value)
                        except ValueError:
                            print("Error: only int allowed")
                            continue
                        timesteps = value
                        continue
            elif cfg_group == "new_cfg":
                print("\nType new config path  (timeout 60s) enter empty for no change\n")

                i, o, e = select.select([sys.stdin], [], [], 60)

                if i:
                    value = sys.stdin.readline().strip()

                    if value == "":
                        print("No change")
                        continue
                    else:
                        try:
                            new_cfg = YAML().load(open(value))
                            cfg = new_cfg
                        except FileNotFoundError:
                            print("File not found")
                        continue

            print("Current value:\n")

            try:
                for item in cfg[cfg_group].items():
                    print(item)
                print("\nType config name from", cfg_group, "you want to edit (timeout 60s)")
            except KeyError:
                print("\nWrong key\n")
                i = True
                continue

            i, o, e = select.select([sys.stdin], [], [], 60)

            if i:
                cfg_name = sys.stdin.readline().strip()
                try:
                    print("\nCurrent value: ", cfg[cfg_group][cfg_name], "\n")
                    print("Type value for [", cfg_group, "][", cfg_name, "] (timeout 60s) enter empty for no change\n")
                    i, o, e = select.select([sys.stdin], [], [], 60)
                except KeyError:
                    print("\nWrong key\n")
                    i = True
                    continue

                if i:
                    value = str(sys.stdin.readline().strip())

                    if value == "":
                        print("No change")
                    else:
                        if not (value == "no" or value == "yes"):
                            try:
                                value = int(value)
                            except ValueError:
                                print("Error: only float allowed")
                                continue
                        cfg[cfg_group][cfg_name] = value

        else:
            print("Continue the training...")

        return cfg, timesteps


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--render", type=int, default=0, help="Render with Unity")
    parser.add_argument("--policy_path", type=str, default=None, help="PPO path")
    parser.add_argument("--policy_number", type=int, default=100, help="PPO iter number")
    return parser


def get_next_dir(path):
    all_folders = os.listdir(path)
    number = 0
    if len(all_folders) > 0:
        all_folders = list(map(int, all_folders))
        all_folders.sort()
        number = int(all_folders[-1]) + 1
    return os.path.join(path, str(number))


def main():
    args = parser().parse_args()
    train_once_each_env = False
    prev_envs = []
    max_env = 100
    iteration_count = 20
    log_id = 0
    total_timesteps = int(3 * 1e7)

    levels = ["easy", "medium", "hard"]


    rsg_root = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(rsg_root + "/multiple_envs_saved", exist_ok=True)
    log_dir = get_next_dir(rsg_root + "/multiple_envs_saved")

    os.makedirs(log_dir, exist_ok=False)

    if args.policy_path is not None:
        cfg = YAML().load(open(args.policy_path + "/config.yaml"))
    else:
        cfg = YAML().load(open("/home/pawel/agile_flight/envtest/python/vision/config.yaml"))

    if args.render:
        cfg["unity"]["render"] = "yes"

    cfg, total_timesteps = ask_edit_cfg_parameters(cfg, total_timesteps)

    train_env = VisionEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
    train_env = wrapper.FlightEnvVec(train_env)

    old_num_envs = cfg["simulation"]["num_envs"]
    cfg["simulation"]["num_envs"] = 1
    eval_env = wrapper.FlightEnvVec(VisionEnv_v1(dump(cfg, Dumper=RoundTripDumper), False))
    cfg["simulation"]["num_envs"] = old_num_envs


    if args.policy_path is not None:
        env_rms_path = os.path.join(args.policy_path, os.path.join("RMS", 'iter_{:05d}.npz'.format(args.policy_number)))

        eval_env.load_rms(env_rms_path)
        train_env.load_rms(env_rms_path)

        device = get_device("auto")

        model_path = os.path.join(args.policy_path, os.path.join("Model", 'model_{:05d}'.format(args.policy_number)))

        model = PPO.load(model_path)
        model.tensorboard_log = log_dir

        model.env_cfg = cfg
        model.set_env(train_env)
        model.eval_env = eval_env
    else:
        model = PPO(
            tensorboard_log=log_dir,
            policy="MlpPolicy",
            policy_kwargs=dict(
                activation_fn=torch.nn.ReLU,
                net_arch=[dict(pi=[256, 256], vf=[512, 512])],
                log_std_init=-0.5,
            ),
            env=train_env,
            eval_env=eval_env,
            use_tanh_act=True,
            gae_lambda=0.95,
            gamma=0.99,
            n_steps=400,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            batch_size=100000,
            clip_range=0.2,
            use_sde=False,  # don't use (gSDE), doesn't work
            env_cfg=cfg,
            verbose=1,
            n_epochs=30,
        )


    model.learn(total_timesteps=total_timesteps, log_interval=(10, 50))

    for i in range(iteration_count):
        print("Iteration: ", i, " / ", iteration_count)

        while True:
            env_num = random.randint(0, max_env)
            if not env_num in prev_envs or not train_once_each_env:
                prev_envs.append(env_num)
                break

        #level = random.randint(0, max_env)

        print("Environment: ", env_num)
        cfg["environment"]["env_folder"] = "environment_" + str(env_num)

        train_env.close()
        eval_env.close()
        
        #cfg["rewards"]["touch_collision_coeff"] = float(cfg["rewards"]["touch_collision_coeff"]) - 0.1
        #cfg["rewards"]["collision_coeff"] = float(cfg["rewards"]["collision_coeff"]) - 0.1
        
        cfg, total_timesteps = ask_edit_cfg_parameters(cfg, total_timesteps)

        del train_env
        del eval_env

        train_env = VisionEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
        train_env = wrapper.FlightEnvVec(train_env)

        old_num_envs = cfg["simulation"]["num_envs"]
        cfg["simulation"]["num_envs"] = 1
        eval_env = wrapper.FlightEnvVec(VisionEnv_v1(dump(cfg, Dumper=RoundTripDumper), False))
        cfg["simulation"]["num_envs"] = old_num_envs

        print("Loading RMS from: ", model.last_rms_path)

        eval_env.load_rms(model.last_rms_path)
        train_env.load_rms(model.last_rms_path)

        model.env_cfg = cfg
        model.set_env(train_env)        #moze inaczej ????? bo tam jest wrap
        model.eval_env = eval_env

        model.learn(total_timesteps=total_timesteps, log_interval=(10, 50))

"""def main2():
    args = parser().parse_args()
    train_once_each_env = False
    prev_envs = []
    max_env = 100
    iteration_count = 80

    total_timesteps = int(1 * 1e7)

    cfg = YAML().load(open("/home/pawel/agile_flight/envtest/python/vision/config.yaml"))

    if args.render:
        cfg["unity"]["render"] = "yes"

    train_env = VisionEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
    train_env = wrapper.FlightEnvVec(train_env)

    old_num_envs = cfg["simulation"]["num_envs"]
    cfg["simulation"]["num_envs"] = 1
    eval_env = wrapper.FlightEnvVec(VisionEnv_v1(dump(cfg, Dumper=RoundTripDumper), False))
    cfg["simulation"]["num_envs"] = old_num_envs

    rsg_root = os.path.dirname(os.path.abspath(__file__))
    log_id = 0
    log_dir = rsg_root + "/saved/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir + "/Models", exist_ok=True)

    model = PPO(
        tensorboard_log=log_dir,
        policy="MlpPolicy",
        policy_kwargs=dict(
            activation_fn=torch.nn.ReLU,
            net_arch=[dict(pi=[256, 256], vf=[512, 512])],
            log_std_init=-0.5,
        ),
        env=train_env,
        eval_env=eval_env,
        use_tanh_act=True,
        gae_lambda=0.95,
        gamma=0.99,
        n_steps=400,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        batch_size=100000,
        clip_range=0.2,
        use_sde=False,  # don't use (gSDE), doesn't work
        env_cfg=cfg,
        verbose=1,
        n_epochs=30,
    )
    default_logger_dir = model.logger.get_dir()
    print("Defult logger dir: ", default_logger_dir)
    model.logger.dir = os.path.join(default_logger_dir, str(log_id))

    model.learn(total_timesteps=total_timesteps, log_interval=(10, 50))

    for i in range(iteration_count):
        print("Iteration: ", i)

        while True:
            env_num = random.randint(0, max_env)
            if not env_num in prev_envs or not train_once_each_env:
                prev_envs.append(env_num)
                break

        print("Environment", env_num)
        cfg["environment"]["env_folder"] = "environment_" + str(env_num)

        train_env.close()
        eval_env.close()

        train_env = VisionEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
        train_env = wrapper.FlightEnvVec(train_env)

        old_num_envs = cfg["simulation"]["num_envs"]
        cfg["simulation"]["num_envs"] = 1
        eval_env = wrapper.FlightEnvVec(VisionEnv_v1(dump(cfg, Dumper=RoundTripDumper), False))
        cfg["simulation"]["num_envs"] = old_num_envs

        log_id += 1
        model.logger.dir = os.path.join(default_logger_dir, str(log_id))

        model.set_env(train_env)        #moze inaczej ????? bo tam jest wrap
        model.eval_env = eval_env

        model.learn(total_timesteps=total_timesteps, log_interval=(10, 50))"""



if __name__ == "__main__":
    main()
