import gymnasium as gym
import os
import numpy as np
import pandas as pd
import time 
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import nibabel as nib
import matplotlib.pyplot as plt
import random
import json

from torchvision import transforms
import io
import imageio
from PIL import Image
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import time

from skimage.draw import line_nd


class BrainEnv(gym.Env):

    def __init__(self, step_size=1, task='L Hippocampus', p_num=32, max_steps=200, dim_view=[27, 27, 27], hist_len=6, render_mode=None, eval_patient_i = None, good_spawn = True, trajectory=False, video = False):
        
        self.trajectory = trajectory
        self.good_spawn = good_spawn
        self.video = video
        self.frames = []
        self.video_path = "episode_video.mp4"

        if self.trajectory:
            self.vessel_mask_path = r"C:\Users\rohan\Downloads\RL Projects\LandmarkBrainRL\mask_transformed_bin.nii.gz"
            self.vessel_mask = nib.load(self.vessel_mask_path).get_fdata().astype(np.uint8)
        else:
            self.vessel_mask = None

        self.eval_patient_i = eval_patient_i
        self.task = task
        self.p_num = p_num
        self.step_size = step_size
        self.base_dir = r"C:\Users\rohan\Downloads\RL Projects\LandmarkBrainRL\LPBA40_Unzipped\LPBA40_Cleaned - Copy"
        
        self.mri, self.label, self.label_dict, self.p_id = self.load_patient(self.base_dir)

        self.targid = self.label_dict[self.task]

        self.goal_voxels, self.goal_coord = self.get_goal(self.label, self.targid)

        self.action_space = gym.spaces.Discrete(6)

        self.agent_pos = self.get_random_brain_voxel(self.mri)

        self.episode_step_counter = 0

        self.max_steps = max_steps

        self.view_dimensions = dim_view

        self.oscillations_allowed = 4

        self.memory_length = hist_len

        self.state = self.voxel_to_state()

        self.observation_history = [self.voxel_to_state().copy() for _ in range(self.memory_length)]

        self.position_memory = [self.agent_pos for _ in range(self.memory_length)]

        self.value_memory = [np.zeros(self.action_space.n) for _ in range(self.memory_length)]

        self.distance_current = self.calc_distance(self.agent_pos, self.goal_coord)

        self.distance_initial = self.distance_current

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.memory_length, *self.view_dimensions),
            dtype=np.float32
        )

        self.render_mode = render_mode

        self.movement_directions = {
            0: (self.step_size, 0, 0),
            1: (-self.step_size, 0, 0),
            2: (0, self.step_size, 0),
            3: (0, -self.step_size, 0),
            4: (0, 0, self.step_size),
            5: (0, 0, -self.step_size),
        }

        self.previous_episode_outcome = None

        self.episode_results_history = []

        self.complete_position_trace = []

        if self.trajectory:
            self.linear_path_voxels = self.compute_linear_path(self.agent_pos, self.goal_coord)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.mri, self.label, self.label_dict, self.p_id = self.load_patient(self.base_dir)

        self.targid = self.label_dict[self.task]

        self.goal_voxels, self.goal_coord = self.get_goal(self.label, self.targid)

        self.agent_pos = self.get_random_brain_voxel(self.mri)

        self.episode_step_counter = 0

        self.state = self.voxel_to_state()

        self.observation_history = [self.state.copy() for _ in range(self.memory_length)]

        self.position_memory = [self.agent_pos] * self.memory_length

        self.complete_position_trace = []

        self.value_memory = [np.zeros(self.action_space.n)] * self.memory_length

        self.distance_current = self.calc_distance(self.agent_pos, self.goal_coord)

        self.distance_initial = self.distance_current

        self.oscillation_count = 0

        stacked_obs = np.stack(self.observation_history, axis=0).astype(np.float32)

        threshold = 1e-3
        mri_volume = self.mri
        brain_mask = mri_volume > threshold
        self.brain_voxels = np.argwhere(brain_mask)

        if self.trajectory:
            self.linear_path_voxels = self.compute_linear_path(self.agent_pos, self.goal_coord)

        return stacked_obs, {}

    def step(self, action):
        
        self.episode_step_counter += 1

        dx, dy, dz = self.movement_directions[int(action)]
        x, y, z = self.agent_pos

        new_position = (x + dx, y + dy, z + dz)

        new_position = tuple(
            np.clip(coord, 0, dim - 1)
            for coord, dim in zip(new_position, self.mri.shape)
        )

        self.agent_pos = new_position

        updated_distance = self.calc_distance(self.agent_pos, self.goal_coord)

        new_state = self.voxel_to_state()

        self.observation_history.pop(0)
        self.observation_history.append(new_state)

        stacked_obs = np.stack(self.observation_history, axis=0).astype(np.float32)

        self.position_memory.pop(0)
        self.position_memory.append(tuple(self.agent_pos))

        self.complete_position_trace.append(tuple(self.agent_pos))

        done, info = self.check_termination(updated_distance)

        reward = self.compute_step_reward(self.distance_current, updated_distance)

        if self.previous_episode_outcome == "oscillated":
            reward = -15

        self.distance_current = updated_distance

        self.state = new_state

        if done:
            info["episode_result"] = self.previous_episode_outcome
            self.save_vid()
        return stacked_obs, reward, done, False, info

    def compute_step_reward(self, old_dist, new_dist):
        dif = old_dist - new_dist
        reward = 1 if dif > 0 else -2
        
        x, y, z = self.agent_pos

        if new_dist < 2.5:
            print("Goal reached!")
            reward += 30
        
        if self.vessel_mask is not None:
            if self.vessel_mask[x, y, z] == 1:
                reward -= 10

        if self.trajectory:
            within_radius = any(
                np.linalg.norm(np.array(self.agent_pos) - np.array(p)) <= 2.0
                for p in self.linear_path_voxels)
        
            if within_radius:
                reward += 0.5

        return reward

    def check_termination(self, new_dist):
        done = False
        info = {}
        self.previous_episode_outcome = None

        x, y, z = self.agent_pos

        intensity = self.mri[x,y,z]

        if new_dist < 2.5:
            done = True
            info["success"] = True
            self.previous_episode_outcome = "goal_reached"
        elif self.episode_step_counter >= self.max_steps:
            done = True
            info["success"] = False
            self.previous_episode_outcome = "timeout"
        elif self._detect_oscillation:
            done = True
            info["success"] = False
            info["reason"] = "oscillation"
            self.previous_episode_outcome = "oscillated"
        elif intensity < 1e-3:
            done = True
            info["success"] = False
            info["reason"] = "outside_brain"
            print("Agent steeped outside the brain volume at voxel:", self.agent_pos)
            self.previous_episode_outcome = "outside_brain"
        
        if done:
            info["episode_result"] =  self.previous_episode_outcome 

        return done, info

    def is_valid_voxel(self, voxel):
        x, y, z = voxel
        sx, sy, sz = self.mri.shape
        return (0 <= x < sx) and (0 <= y < sy) and (0 <= z < sz)

    def load_patient(self, base_dir):

        if self.eval_patient_i is not None:
            patient_ids = [f"S{idx:02d}" for idx in self.eval_patient_i]
        else:
            patient_ids = [f"S{idx:02d}" for idx in range(1, 33)]

        chosen_id = random.choice(patient_ids)

        patient_path = os.path.join(base_dir, chosen_id)

        mri = nib.load(os.path.join(patient_path, "mri.nii.gz")).get_fdata()
        if mri.ndim == 4 and mri.shape[-1] == 1:
            mri = np.squeeze(mri, axis=-1)
        label = nib.load(os.path.join(patient_path, "labels.nii.gz")).get_fdata().astype(np.int16)

        with open(os.path.join(patient_path, "labels.json")) as f:
            label_dict = json.load(f)

        return mri, label, label_dict, chosen_id

    def get_goal(self, label, targid):
        label = np.squeeze(label)

        goal_voxels = np.argwhere(label == targid)

        if len(goal_voxels) == 0:
            raise ValueError(f"No voxels found for target ID {targid}")

        center = goal_voxels.mean(axis=0)

        dists = np.linalg.norm(goal_voxels - center, axis=1)
        closest_idx = np.argmin(dists)

        goal_coord = tuple(goal_voxels[closest_idx])

        return goal_voxels, goal_coord

    def get_random_brain_voxel(self, mri_volume, threshold=1e-3):
        brain_mask = mri_volume > threshold
        brain_voxels = np.argwhere(brain_mask)

        if len(brain_voxels) == 0:
            raise ValueError("No brain voxels found in the volume.")

        if self.good_spawn:
            min_distance = 30
            max_distance = 95
            max_attempts = 1000

            if self.trajectory:
                assert hasattr(self, "vessel_mask"), "vessel_mask not set in environment."
                assert self.vessel_mask.shape == mri_volume.shape, "vessel_mask shape mismatch."

                vessel_mask = self.vessel_mask > 0
                brain_voxels = np.array([v for v in brain_voxels if not vessel_mask[tuple(v)]])
                
                if len(brain_voxels) == 0:
                    raise ValueError("No brain voxels found outside vessel regions.")

            for _ in range(max_attempts):
                idx = np.random.choice(len(brain_voxels))
                voxel = tuple(brain_voxels[idx][:3])
                distance = self.calc_distance(voxel, self.goal_coord)

                if min_distance <= distance < max_distance:
                    return voxel

            print("Warning: No voxel ≥50 units from goal (and not on vessel, if enabled) found. Returning fallback.")

        idx = np.random.choice(len(brain_voxels))
        return tuple(brain_voxels[idx][:3])

    def voxel_to_state(self):
        dx, dy, dz = self.view_dimensions
        cx, cy, cz = self.agent_pos
        hx, hy, hz = dx // 2, dy // 2, dz // 2
        xmin, xmax = cx - hx, cx + hx + (dx % 2)
        ymin, ymax = cy - hy, cy + hy + (dy % 2)
        zmin, zmax = cz - hz, cz + hz + (dz % 2)

        sx, sy, sz = self.mri.shape[:3]
        patch = np.zeros(self.view_dimensions, dtype=self.mri.dtype)

        vxmin, vxmax = max(0, xmin), min(sx, xmax)
        vymin, vymax = max(0, ymin), min(sy, ymax)
        vzmin, vzmax = max(0, zmin), min(sz, zmax)

        pxmin = vxmin - xmin
        pxmax = pxmin + (vxmax - vxmin)
        pymin = vymin - ymin
        pymax = pymin + (vymax - vymin)
        pzmin = vzmin - zmin
        pzmax = pzmin + (vzmax - vzmin)

        patch[pxmin:pxmax, pymin:pymax, pzmin:pzmax] = self.mri[vxmin:vxmax, vymin:vymax, vzmin:vzmax]

        patch = self.patch_normalize_zscore(patch)

        return patch

    def calc_distance(self, voxel1, voxel2):
        v1 = np.array(voxel1, dtype=np.float32)
        v2 = np.array(voxel2, dtype=np.float32)

        return np.linalg.norm(v1 - v2)

    def patch_normalize_zscore(self, patch):
        patch = patch.astype(np.float32)

        mean = patch.mean()
        std = patch.std()
        norm = (patch - mean) / (std + 1e-5)
        norm = np.clip(norm, -3, 3)
        norm = (norm + 3) / 6.0 * 255
        return norm.astype(np.uint8)

    @property
    def _detect_oscillation(self):
        voxels = [v for v in self.position_memory if v != (0, 0, 0)]
        counter = Counter(voxels)

        current_voxel = self.position_memory[-1]

        if counter[current_voxel] >= 3:
            print(f"Oscillation detected: voxel {current_voxel} visited {counter[current_voxel]} times.")
            return True

        return False

    def compute_linear_path(self, start, goal):
        line_coords = line_nd(start, goal)
        return list(zip(*line_coords))

    def render(self):
        import numpy as np
        import matplotlib.pyplot as plt

        if self.render_mode != "human":
            return

        path = np.array(self.complete_position_trace)
        x, y, z = self.agent_pos
        gx, gy, gz = self.goal_coord

        
        actual_label = self.label[gx, gy, gz]
        if actual_label != self.targid: 
            print(f"WARNING: Goal voxel {self.goal_coord} has label {actual_label}, expected {self.targid} ({self.task})")
        else:
            print(f" Goal voxel {self.goal_coord} correctly labeled as {self.task}")

        if self.episode_step_counter == 1:
            fig2, axs2 = plt.subplots(1, 3, figsize=(18, 6))

            goal_sagittal = np.squeeze(self.mri[gx, :, :])
            goal_label_sagittal = np.squeeze((self.label[gx, :, :] == self.targid).astype(np.uint8))
            axs2[0].imshow(goal_sagittal.T, cmap='gray', origin='lower')
            axs2[0].imshow(np.ma.masked_where(goal_label_sagittal.T == 0, goal_label_sagittal.T), cmap='Reds', alpha=0.9, origin='lower')
            axs2[0].scatter(gy, gz, c='blue', s=100, marker='x', label='Goal')
            axs2[0].set_title(f'Goal Sagittal Slice X={gx}')
            axs2[0].axis('off')
            axs2[0].legend()

            goal_coronal = np.squeeze(self.mri[:, gy, :])
            goal_label_coronal = np.squeeze((self.label[:, gy, :] == self.targid).astype(np.uint8))
            axs2[1].imshow(goal_coronal.T, cmap='gray', origin='lower')
            axs2[1].imshow(np.ma.masked_where(goal_label_coronal.T == 0, goal_label_coronal.T), cmap='Reds', alpha=0.9, origin='lower')
            axs2[1].scatter(gx, gz, c='blue', s=100, marker='x', label='Goal')
            axs2[1].set_title(f'Goal Coronal Slice Y={gy}')
            axs2[1].axis('off')
            axs2[1].legend()

            goal_axial = np.squeeze(self.mri[:, :, gz])
            goal_label_axial = np.squeeze((self.label[:, :, gz] == self.targid).astype(np.uint8))
            axs2[2].imshow(goal_axial.T, cmap='gray', origin='lower')
            axs2[2].imshow(np.ma.masked_where(goal_label_axial.T == 0, goal_label_axial.T), cmap='Reds', alpha=0.9, origin='lower')
            axs2[2].scatter(gx, gy, c='blue', s=100, marker='x', label='Goal')
            axs2[2].set_title(f'Goal Axial Slice Z={gz}')
            axs2[2].axis('off')
            axs2[2].legend()

            plt.suptitle("Goal Voxel Location in All Views", fontsize=16)
            plt.tight_layout()
            plt.show()

        mri_axial = np.squeeze(self.mri[:, :, z])
        label_axial = np.squeeze((self.label[:, :, z] == self.targid).astype(np.uint8))

        mri_sagittal = np.squeeze(self.mri[x, :, :])
        label_sagittal = np.squeeze((self.label[x, :, :] == self.targid).astype(np.uint8))

        mri_coronal = np.squeeze(self.mri[:, y, :])
        label_coronal = np.squeeze((self.label[:, y, :] == self.targid).astype(np.uint8))

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        axs[0].imshow(mri_axial.T, cmap='gray', origin='lower')
        axs[0].imshow(np.ma.masked_where(label_axial.T == 0, label_axial.T), cmap='Reds', alpha=0.9, origin='lower')
        if self.vessel_mask is not None:
            vessel_axial = np.squeeze(self.vessel_mask[:, :, z])
            axs[0].imshow(np.ma.masked_where(vessel_axial.T == 0, vessel_axial.T), cmap='Purples', alpha=0.3, origin='lower')
        axs[0].contour(label_axial.T, colors='red', linewidths=1, origin='lower')
        axs[0].scatter(x, y, c='lime', s=50, label='Agent')
        axs[0].scatter(gx, gy, c='blue', marker='x', s=100, label='Goal')

        if hasattr(self, "linear_path_voxels"):
            line_points = [(lx, ly) for lx, ly, lz in self.linear_path_voxels if lz == z]
            if line_points:
                lp_arr = np.array(line_points)
                axs[0].scatter(lp_arr[:, 0], lp_arr[:, 1], c='magenta', s=8, label='Linear Path')

        if path.shape[0] > 1:
            path_xy = path[:, :2]
            axs[0].plot(path_xy[:, 0], path_xy[:, 1], c='cyan', linewidth=2, label='Path')
        axs[0].set_title(f'Axial Slice Z={z}')
        axs[0].axis('off')
        axs[0].legend(loc='upper right')

        axs[1].imshow(mri_sagittal.T, cmap='gray', origin='lower')
        axs[1].imshow(np.ma.masked_where(label_sagittal.T == 0, label_sagittal.T), cmap='Reds', alpha=0.9, origin='lower')
        if self.vessel_mask is not None:
            vessel_sagittal = np.squeeze(self.vessel_mask[x, :, :])
            axs[1].imshow(np.ma.masked_where(vessel_sagittal.T == 0, vessel_sagittal.T), cmap='Purples', alpha=0.3, origin='lower')
        axs[1].contour(label_sagittal.T, colors='red', linewidths=1, origin='lower')
        axs[1].scatter(y, z, c='lime', s=50, label='Agent')
        axs[1].scatter(gy, gz, c='blue', marker='x', s=100, label='Goal')

        if hasattr(self, "linear_path_voxels"):
            line_points = [(ly, lz) for lx, ly, lz in self.linear_path_voxels if lx == x]
            if line_points:
                lp_arr = np.array(line_points)
                axs[1].scatter(lp_arr[:, 0], lp_arr[:, 1], c='magenta', s=8, label='Linear Path')

        if path.shape[0] > 1:
            path_yz = path[:, 1:]
            axs[1].plot(path_yz[:, 0], path_yz[:, 1], c='cyan', linewidth=2, label='Path')
        axs[1].set_title(f'Sagittal Slice X={x}')
        axs[1].axis('off')
        axs[1].legend(loc='upper right')

        axs[2].imshow(mri_coronal.T, cmap='gray', origin='lower')
        axs[2].imshow(np.ma.masked_where(label_coronal.T == 0, label_coronal.T), cmap='Reds', alpha=0.9, origin='lower')
        if self.vessel_mask is not None:
            vessel_coronal = np.squeeze(self.vessel_mask[:, y, :])
            axs[2].imshow(np.ma.masked_where(vessel_coronal.T == 0, vessel_coronal.T), cmap='Purples', alpha=0.3, origin='lower')
        axs[2].contour(label_coronal.T, colors='red', linewidths=1, origin='lower')
        axs[2].scatter(x, z, c='lime', s=50, label='Agent')
        axs[2].scatter(gx, gz, c='blue', marker='x', s=100, label='Goal')

        if hasattr(self, "linear_path_voxels"):
            line_points = [(lx, lz) for lx, ly, lz in self.linear_path_voxels if ly == y]
            if line_points:
                lp_arr = np.array(line_points)
                axs[2].scatter(lp_arr[:, 0], lp_arr[:, 1], c='magenta', s=8, label='Linear Path')

        if path.shape[0] > 1:
            path_xz = path[:, [0, 2]]
            axs[2].plot(path_xz[:, 0], path_xz[:, 1], c='cyan', linewidth=2, label='Path')

        axs[2].set_title(f'Coronal Slice Y={y}')
        axs[2].axis('off')
        axs[2].legend(loc='upper right')

        plt.tight_layout()

        if self.video:
            self._save_current_figure()
        plt.show()
        plt.close(fig)
    
    def _save_current_figure(self):
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img = Image.open(buf)
        self.frames.append(img.copy())
        buf.close()

    def save_vid(self):
        if self.video and len(self.frames) > 0:
            try:
                timestamp = int(time.time())
                video_filename = f"episode_video_{timestamp}.mp4"
                
                frame_arrays = []
                for frame in self.frames:
                    if hasattr(frame, 'convert'):
                        frame_array = np.array(frame.convert('RGB'))
                    else:
                        frame_array = frame
                    frame_arrays.append(frame_array)
                
                with imageio.get_writer(video_filename, fps=2, codec='libx264') as writer:
                    for frame in frame_arrays:
                        writer.append_data(frame)
                
                print(f"Video saved to {video_filename} with {len(self.frames)} frames")
                
            except Exception as e:
                print(f"MP4 save failed: {e}")
                try:
                    gif_filename = f"episode_video_{timestamp}.gif"
                    frame_arrays = []
                    for frame in self.frames:
                        if hasattr(frame, 'convert'):
                            frame_array = np.array(frame.convert('RGB'))
                        else:
                            frame_array = frame
                        frame_arrays.append(frame_array)
                    
                    imageio.mimsave(gif_filename, frame_arrays, fps=2)
                    print(f"Video saved as GIF to {gif_filename} with {len(self.frames)} frames")
                except Exception as gif_error:
                    print(f"GIF save also failed: {gif_error}")
                    import traceback
                    traceback.print_exc()
            finally:
                self.frames = []
        elif self.video:
            print("No frames to save for video")
        else:
            print("Video recording not enabled")

    def test_axis_mapping(self):
        print("Initial voxel:", self.agent_pos)
        print("Goal voxel:", self.goal_coord)
        print("Initial distance:", self.calc_distance(self.agent_pos, self.goal_coord))
        for action in range(6):
            dx, dy, dz = self.movement_directions[action]
            new_voxel = tuple(
                np.clip(coord + delta, 0, dim - 1)
                for coord, delta, dim in zip(self.agent_pos, (dx, dy, dz), self.mri.shape)
            )
            new_dist = self.calc_distance(new_voxel, self.goal_coord)
            print(f"Action {action}: Move {dx, dy, dz}, New voxel: {new_voxel}, New distance: {new_dist}")
