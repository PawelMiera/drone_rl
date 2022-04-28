from stable_baselines3.common.vec_env.base_vec_env import (VecEnv,
                                                           VecEnvIndices,
                                                           VecEnvObs,
                                                           VecEnvStepReturn)
from rpg_baselines.torch.envs.vec_env_wrapper import FlightEnvVec
from gym import spaces
import numpy as np
import pickle
from typing import Any, Callable, List, Optional, Sequence, Type, Union
import gym
import cv2
import os


class MyWrapper(VecEnv):
    def __init__(self, impl: FlightEnvVec):
        self.wrapper = impl
        self.num_envs = self.wrapper.num_envs
        self.rew_dim = self.wrapper.rew_dim
        self.reward_names = self.wrapper.reward_names
        self.max_episode_steps = 1000

        self.observation_space = spaces.Dict(
            spaces={
                "vector": spaces.Box(np.ones(self.wrapper.obs_dim) * -np.Inf, np.ones(self.wrapper.obs_dim) * np.Inf,
                                  dtype=np.float64, ),
                "image": spaces.Box(0, 255, [3, 240, 320], dtype=np.uint8),
            }
        )

        self.action_space = spaces.Box(
            low=np.ones(self.wrapper.act_dim) * -1.0,
            high=np.ones(self.wrapper.act_dim) * 1.0,
            dtype=np.float64,
        )

        self.frame_id = 0
        self.save_num = 0

    def disconnectUnity(self):
        self.wrapper.disconnectUnity()

    def seed(self, seed=0):
        self.wrapper.seed(seed)

    def teststep(self, action):
        obs, reward, done, info = self.wrapper.step(action)
        print("frame test step ", self.frame_id)
        # self.render(self.frame_id)
        # self.frame_id += 1
        imgs = self.wrapper.getImage(True)
        imgs_list =[]
        for i in range(self.wrapper.num_envs):
            img = np.reshape(imgs[i], (3, self.wrapper.img_height, self.wrapper.img_width))
            imgs_list.append(img)
        imgs_list = np.array(imgs_list)
        return {"vector": obs, "image": imgs_list}, reward, done, info

    def reset(self, random=True):
        obs = self.wrapper.reset(random)
        print("frame res ", self.frame_id)
        if self.frame_id != 0:
            self.render(self.frame_id)
            self.frame_id += 1
        imgs = self.wrapper.getImage(True)
        imgs_list =[]
        for i in range(self.wrapper.num_envs):
            img = np.reshape(imgs[i], (3, self.wrapper.img_height, self.wrapper.img_width))
            imgs_list.append(img)
        imgs_list = np.array(imgs_list)
            # ob = obs[i]
            # states.append({"vector": ob, "image": img})
        return {"vector": obs, "image": imgs_list}

    def step(self, action):
        obs, reward, done, info = self.wrapper.step(action)
        self.render(self.frame_id)
        self.frame_id += 1

        imgs = self.wrapper.getImage(True)
        imgs_list =[]
        for i in range(self.wrapper.num_envs):
            img = np.reshape(imgs[i], (3, self.wrapper.img_height, self.wrapper.img_width))
            imgs_list.append(img)

        if self.save_num % 100 == 0:
            print("!!!!!!!!!!!!!\nMEAN ", np.mean(imgs_list[0][0]), "\n!!!!!!!!!!!!!!")
            cv2.imwrite("images/"+str(0)+".jpg", imgs_list[0][0])
        imgs_list = np.array(imgs_list)
        self.save_num += 1
        return {"vector": obs, "image": imgs_list}, reward, done, info

    def render(self, frame_id, mode="human"):
        return self.wrapper.render(frame_id, mode)

    def save(self, save_path: str) -> None:
        self.wrapper.save(save_path)

    def save_rms(self, save_dir, n_iter) -> None:
        self.wrapper.save_rms(save_dir, n_iter)

    def load_rms(self, data_dir) -> None:
        self.wrapper.load_rms(data_dir)

    def close(self):
        self.wrapper.close()

    def env_is_wrapped(
            self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None
    ) -> List[bool]:

        return self.wrapper.env_is_wrapped(wrapper_class, indices)


    def connectUnity(self):
        self.wrapper.connectUnity()

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs
    ) -> List[Any]:
        """Call instance methods of vectorized environments."""
        return self.wrapper.env_method(method_name, *method_args, indices, **method_kwargs)


    def start_recording_video(self, file_name):
        raise RuntimeError("This method is not implemented")

    def stop_recording_video(self):
        raise RuntimeError("This method is not implemented")

    def curriculum_callback(self):
        self.wrapper.curriculumUpdate()

    def step_async(self):
        raise RuntimeError("This method is not implemented")

    def step_wait(self):
        raise RuntimeError("This method is not implemented")

    def get_attr(self, attr_name, indices=None):
        """
        Return attribute from vectorized environment.
        :param attr_name: (str) The name of the attribute whose value to return
        :param indices: (list,int) Indices of envs to get attribute from
        :return: (list) List of values of 'attr_name' in all environments
        """
        raise RuntimeError("This method is not implemented")

    def set_attr(self, attr_name, value, indices=None):
        """
        Set attribute inside vectorized environments.
        :param attr_name: (str) The name of attribute to assign new value
        :param value: (obj) Value to assign to `attr_name`
        :param indices: (list,int) Indices of envs to assign value
        :return: (NoneType)
        """
        raise RuntimeError("This method is not implemented")

    def update_rms(self):
            self.wrapper.obs_rms = self.wrapper.obs_rms_new