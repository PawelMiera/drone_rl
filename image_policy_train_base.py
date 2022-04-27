#!/usr/bin/env python3
import argparse
import math
#
import os
import random

import numpy as np
import torch
from flightgym import VisionEnv_v1
from ruamel.yaml import YAML, RoundTripDumper, dump
from stable_baselines3.common.utils import get_device
from stable_baselines3.ppo.policies import MlpPolicy, MultiInputPolicy, CnnPolicy

from rpg_baselines.torch.common.ppo import PPO
from rpg_baselines.torch.envs import vec_env_wrapper as wrapper
from rpg_baselines.torch.common.util import test_policy

# from stable_baselines3.ppo import PPO

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym
from torch import nn
from myWrapper import MyWrapper

from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import TensorDict


class MyCNN(BaseFeaturesExtractor):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(MyCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False), (
            "You should use NatureCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html"
        )
        n_input_channels = observation_space.shape[0]
        model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)

        self.cnn = nn.Sequential(
            model,
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


class CombinedExtractor(BaseFeaturesExtractor):
    """
    Combined feature extractor for Dict observation spaces.
    Builds a feature extractor for each key of the space. Input from each space
    is fed through a separate submodule (CNN or MLP, depending on input shape),
    the output features are concatenated and fed through additional MLP network ("combined").

    :param observation_space:
    :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
        256 to avoid exploding network sizes.
    """

    def __init__(self, observation_space: gym.spaces.Dict, cnn_output_dim: int = 256):
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super(CombinedExtractor, self).__init__(observation_space, features_dim=256 + 16)

        extractors = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace):
                extractors[key] = MyCNN(subspace, features_dim=cnn_output_dim)
                total_concat_size += cnn_output_dim
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> torch.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return torch.cat(encoded_tensor_list, dim=1)


# class CustomCombinedExtractor(BaseFeaturesExtractor):
#     def __init__(self, observation_space: gym.spaces.Box):
#         # We do not know features-dim here before going over all the items,
#         # so put something dummy for now. PyTorch requires calling
#         # nn.Module.__init__ before adding modules
#         super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=615)
#
#         self.extractors = {}
#
#         total_concat_size = 0
#
#         self.n_env = 15
#
#         n_input_channels = observation_space.shape
#
#         model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
#
#         self.extractors["image"] = nn.Sequential(
#             model,
#             nn.Flatten(),
#         ).cuda()
#
#         self.extractors["vector"] = nn.Flatten().cuda()
#         total_concat_size = 615
#
#
#         # Update the features dim manually
#         self._features_dim = total_concat_size
#
#     def forward(self, observations) -> torch.Tensor:
#         encoded_tensor_list = []
#
#         observations = torch.nan_to_num(observations, 0)
#         mo = observations[:, 0:76800]
#         mo_vec = observations[:, 76800:]
#
#         mo_img = torch.reshape(mo, (self.n_env, 1, 240, 320))
#
#         my_obs = [mo_img, mo_vec]
#
#
#         # self.extractors contain nn.Modules that do all the processing.
#         out_tensor = []
#         for i in range(0, self.n_env):
#             encoded_tensor_list.append(torch.reshape(torch.cat([torch.flatten(self.extractors["image"](my_obs[0][0])), my_obs[1][0]]), (1,615)))
#
#         out_tensor = torch.cat(encoded_tensor_list, 0)
#         # for key, extractor in self.extractors.items():
#         #     id = 0
#         #     if key == "vector":
#         #         id = 1
#         #
#         #     encoded_tensor_list.append(extractor(my_obs[id]))
#
#         # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
#         print("ELLLL", out_tensor.shape)
#         #return torch.cat(encoded_tensor_list, dim=1)
#         print(torch.cat(encoded_tensor_list, dim=0).shape)
#         return torch.cat(encoded_tensor_list, dim=0)



def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--train", type=int, default=1, help="Train the policy or evaluate the policy")
    parser.add_argument("--render", type=int, default=1, help="Render with Unity")
    parser.add_argument("--trial", type=int, default=1, help="PPO trial number")
    parser.add_argument("--iter", type=int, default=100, help="PPO iter number")
    return parser


def main():

    args = parser().parse_args()

    # load configurations
    cfg = YAML().load(open("/home/pawel/agh_code/configs/image.yaml"))

    if not args.train:
        cfg["simulation"]["num_envs"] = 1


    os.system(os.environ["FLIGHTMARE_PATH"] + "/flightrender/RPG_Flightmare.x86_64 &")

    if args.render:
        cfg["unity"]["render"] = "yes"
        # create training environment
    train_env = VisionEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
    train_env = wrapper.FlightEnvVec(train_env)
    train_env = MyWrapper(train_env)

    # set random seed
    configure_random_seed(args.seed, env=train_env)



    # create evaluation environment
    old_num_envs = cfg["simulation"]["num_envs"]
    cfg["simulation"]["num_envs"] = 1
    eval_env = wrapper.FlightEnvVec(
        VisionEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
    )

    # eval_env = MyWrapper(eval_env)
    cfg["simulation"]["num_envs"] = old_num_envs

    # save the configuration and other files
    rsg_root = os.path.dirname(os.path.abspath(__file__))
    log_dir = rsg_root + "/saved"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir + "/Models", exist_ok=True)

    policy_kwargs = dict(
        features_extractor_class=CombinedExtractor,
        activation_fn=torch.nn.ReLU,
        net_arch=[dict(pi=[512, 256, 256], vf=[1024, 512, 512])],
        log_std_init=-0.5,
    )        #policy_kwargs=policy_kwargs,

    # policy_kwargs = dict(
    #     activation_fn=torch.nn.ReLU,
    #     net_arch=[dict(pi=[512, 256, 256], vf=[1024, 512, 512])],
    #     log_std_init=-0.5,
    # ),

    model = PPO(
        tensorboard_log=log_dir,
        policy="MultiInputPolicy",
        policy_kwargs=policy_kwargs,
        env=train_env,
        eval_env=eval_env,
        use_tanh_act=True,
        gae_lambda=0.95,
        gamma=0.99,
        n_steps=2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        batch_size=20,
        clip_range=0.2,
        use_sde=False,  # don't use (gSDE), doesn't work
        env_cfg=cfg,
        verbose=1,
        n_epochs=30,
    )

    train_env.connectUnity()
    #
    for i in range(7):
        seed = random.randint(0, 150)
        print("SEED: ", seed)
        configure_random_seed(seed, env=train_env)
        model.learn(total_timesteps=int(6 * 1e7), log_interval=(5,50))


if __name__ == "__main__":
    main()
