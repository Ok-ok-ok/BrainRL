import os
import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

import torch

import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from gymnasium import spaces

from typing import Optional

from env import BrainEnv

from CNN_Policy import CNNPolicy

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.dqn.policies import DQNPolicy


class CustomCNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        hist_length = observation_space.shape[0]
        num_actions = 6 
        self.cnn = CNNPolicy(hist_length, num_actions)
        self._features_dim = 128 

    def forward(self, observations):

        x = observations.to(self.cnn.device) / 255.0
       
        x = self.cnn.conv1(x)
        x = self.cnn.prelu1(x)
        x = self.cnn.maxpool1(x)
        x = self.cnn.conv2(x)
        x = self.cnn.prelu2(x)
        x = self.cnn.maxpool2(x)
        x = self.cnn.conv3(x)
        x = self.cnn.prelu3(x)
        x = self.cnn.maxpool3(x)
        x = self.cnn.conv4(x)
        x = self.cnn.prelu4(x)
        x = x.view(x.size(0), -1)
        x = self.cnn.fc1(x)
        x = self.cnn.prelu5(x)
        x = self.cnn.fc2(x)
        x = self.cnn.prelu6(x)
       
        return x

from stable_baselines3.common.callbacks import BaseCallback

class ReplayBufferMonitorCallback(BaseCallback):
    def __init__(self, total_buffer_size, batch_size=4, verbose=0):
        super().__init__(verbose)
        self.total_buffer_size = total_buffer_size
        self.printed_thresholds = set()

        self.batch_size = batch_size
        self.episode_termination_reasons = []
        self.aggregated_stats = []

        
        self._last_done = False

    def _on_step(self) -> bool:
        
        current_size = self.model.replay_buffer.size()
        fraction_filled = current_size / self.total_buffer_size

        for i in range(1, 9):
            threshold = i / 8
            if fraction_filled >= threshold and threshold not in self.printed_thresholds:
                print(f"Replay buffer is {i}/8 full ({current_size}/{self.total_buffer_size})")
                self.printed_thresholds.add(threshold)

        
        done = False
        infos = []

        
        if "dones" in self.locals:
            dones = self.locals["dones"]
            if isinstance(dones, (list, tuple, np.ndarray)):
                done = dones[0]
            else:
                done = dones

        if "infos" in self.locals:
            infos = self.locals["infos"]
        
        
        if done and not self._last_done:
            #
            if isinstance(infos, (list, tuple)) and len(infos) > 0:
                info = infos[0]
                reason = info.get("episode_result", info.get("reason", "unknown"))
                self.episode_termination_reasons.append(reason)

                
                if len(self.episode_termination_reasons) % self.batch_size == 0:
                    batch = self.episode_termination_reasons[-self.batch_size:]
                    summary = {
                        "batch": len(self.aggregated_stats) + 1,
                        "goal_reached": batch.count("goal_reached"),
                        "oscillated": batch.count("oscillated"),
                        "timeout": batch.count("timeout"),
                        "out_of_bounds": batch.count("out_of_bounds"),
                        "unknown": batch.count("unknown")
                    }
                    self.aggregated_stats.append(summary)

                    print(f"\n[Batch {summary['batch']}] Termination reasons summary (last {self.batch_size} episodes):")
                    for k, v in summary.items():
                        if k != "batch":
                            print(f"  {k}: {v}")

        self._last_done = done
        return True

    def get_stats(self):
        return self.aggregated_stats
    
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env


env = BrainEnv(
    task="", ### Type in the task, defaults to L Hippocampus, look at labels.json and type in the task by the name of the strcture
    step_size=1,
    max_steps=200,
    dim_view=[27, 27, 27],
    good_spawn = True, #This is if you want it to spawn a decent distance away from the target
    trajectory=True # This is if you want to train your agent to avoid blood vessels aswell

)
env = Monitor(env) 

check_env(env)

policy_kwargs = dict(
    features_extractor_class=CustomCNNFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=[128, 128]
)

callback = ReplayBufferMonitorCallback(total_buffer_size=6000)


model = DQN(
    "CnnPolicy",
    env=env,
    policy_kwargs=policy_kwargs,
    batch_size=32,
    learning_rate=5e-4,
    buffer_size=6000,
    learning_starts=1000,
    train_freq=1,
    target_update_interval=1000,
    gamma=0.95,
    verbose=1,
    tensorboard_log="./dqn_tensorboard" 
)

model.learn(
    total_timesteps=100000,  
    log_interval=1,
    callback=callback  
)


model.save("DQN_task")
model.save_replay_buffer("DQN_task_replay_buffer")





