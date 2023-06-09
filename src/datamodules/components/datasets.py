"""Module for experiments' dataset methods."""
import os
import random
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict
import pickle
import numpy as np
import av
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torchvision.datasets.video_utils import VideoClips
from torch.utils.data import Dataset


class UCFVideoDataset(Dataset):
    """Class for UCF video dataset."""

    def __init__(
        self,
        clip_length_in_frames: int,
        step_between_clips: int,
        path_to_data: str,
        path_to_annotation: str,
        path_to_clips: str, 
        video_transform: Optional[nn.Module] = None,
        num_workers: int = 0,
        fps: int = 30,
        sampler: str = "all",
    ) -> None:
        """Initialize UCF video dataset.

        :param clip_length_in_frames: length of clip
        :param step_between_clips: step between clips
        :param path_to_data: path to video data
        :param path_to_annotation: path to annotation labelling
        :param video_transform: necessary transformation
        :param num_workers: num of CPUs
        :param fps: frames per second
        :param sampler: type of sampling:
            - equal - number of clips with the anomaly is almost equal to the number of normal clips
            - downsample_norm - downsample number of normal clips
        """
        super().__init__()

        self.clip_length_in_frames = clip_length_in_frames
        self.step_between_clips = step_between_clips
        self.num_workers = num_workers
        self.fps = fps

        # IO
        self.path_to_data = path_to_data
        self.video_list = UCFVideoDataset._get_video_list(dataset_path=self.path_to_data)


        # annotation loading
        dict_metadata, _ = UCFVideoDataset._parse_annotation(path_to_annotation)

        path_to_clips_ = "{}_clips_len_{}_step_{}_fps_{}.pth".format(
            self.path_to_data.split("/")[1],
            self.clip_length_in_frames,
            self.step_between_clips,
            self.fps,
        )
        if "train" in path_to_annotation:
            path_to_clips = path_to_clips + 'train_' + path_to_clips_
        else:
            path_to_clips = path_to_clips + 'test_' + path_to_clips_

        if os.path.exists(path_to_clips):
            with open(path_to_clips, "rb") as clips:
                self.video_clips = pickle.load(clips)
        else:
            # data loading
            self.video_clips = VideoClips(
                video_paths=self.video_list,
                clip_length_in_frames=self.clip_length_in_frames,
                frames_between_clips=self.step_between_clips,
                frame_rate=self.fps,
                num_workers=self.num_workers,
            )

            # labelling
            self.video_clips.compute_clips(
                self.clip_length_in_frames, self.step_between_clips, self.fps
            )
            self._set_labels_to_clip(dict_metadata)

        # transforms and samplers
        self.video_transform = video_transform
        self.sampler = sampler

        # save dataset
        if not os.path.exists(path_to_clips):
            with open(path_to_clips, "wb") as clips:
                pickle.dump(self.video_clips, clips, protocol=pickle.HIGHEST_PROTOCOL)

        if sampler == "equal":
            self.video_clips.valid_idxs = self._equal_sampling()
        elif sampler == "downsample_norm":
            self.video_clips.valid_idxs = self._equal_sampling(downsample_normal=300)
        elif sampler == "all":
            pass
        else:
            raise ValueError("Wrong type of sampling")

    def __len__(self) -> int:
        """Get length of dataset."""
        return len(self.video_clips.valid_idxs)

    def __getitem__(self, idx: int) -> Union[np.array, np.array]:
        """Get clip and proper labelling with for a given index.

        :param idx: index
        :return: video and labels
        """
        idx = self.video_clips.valid_idxs[idx]
        video, _, _, _ = self.video_clips.get_clip(idx)
        video_idx, clip_idx = self.video_clips.get_clip_location(idx)
        label = np.array(self.video_clips.labels[video_idx][clip_idx], dtype=int)
        # should be channel, seq_len, height, width
        video = video.permute(3, 0, 1, 2)
        if self.video_transform is not None:
            video = self.video_transform(video)
        return video, label

    def _set_labels_to_clip(self, dict_metadata) -> None:
        self.video_clips.labels = []
        self.video_clips.valid_idxs = list(range(0, len(self.video_clips)))
        self.video_clips.normal_idxs = defaultdict(list)
        self.video_clips.cp_idxs = defaultdict(list)

        global_clip_idx = -1
        for video_idx, vid_clips in tqdm(
            enumerate(self.video_clips.clips), total=len(self.video_clips.clips)
        ):
            video_labels = []

            video_path = self.video_clips.video_paths[video_idx]
            video_name = os.path.basename(video_path)

            # get time unit to map frame with its time appearance
            time_unit = (
                av.open(video_path, metadata_errors="ignore").streams[0].time_base
            )

            # get start, change point, end from annotation
            annotated_time = dict_metadata[video_name]

            for clip in vid_clips:
                clip_labels = []
                global_clip_idx += 1

                clip_start = float(time_unit * clip[0].item())
                clip_end = float(time_unit * clip[-1].item())

                change_point = None
                not_suit_flag = True
                for start_time, change_point, end_time in annotated_time:
                    if end_time != -1.0:
                        if (
                            (clip_end > end_time)
                            or (clip_start > end_time)
                            or (clip_start < start_time)
                        ):
                            continue
                    else:
                        if clip_start < start_time:
                            continue
                    if (clip_start > change_point) and (change_point != -1.0):
                        # "abnormal" clip appears after change point
                        continue
                    else:
                        not_suit_flag = False
                        # proper clip
                        break

                # TODO: change_point definition isn't clear (last values which lead to the break)
                if not_suit_flag:
                    # drop clip idx from dataset
                    video_labels.append([])
                    self.video_clips.valid_idxs.remove(global_clip_idx)
                else:
                    if "Normal" in video_path:
                        clip_labels = list(np.zeros(len(clip)))
                        self.video_clips.normal_idxs[video_path].append(global_clip_idx)
                    else:
                        for frame in clip:
                            frame_time = float(time_unit * frame.item())
                            # due to different rounding while moving from frame to time
                            # the true change point is delayed by ~1 frame
                            # so, we've added the little margin
                            if (frame_time >= change_point - 1e-6) and (
                                change_point != -1.0
                            ):
                                clip_labels.append(1)
                            else:
                                clip_labels.append(0)
                        if sum(clip_labels) > 0:
                            self.video_clips.cp_idxs[video_path].append(global_clip_idx)
                        else:
                            self.video_clips.normal_idxs[video_path].append(
                                global_clip_idx
                            )
                    video_labels.append(clip_labels)
            self.video_clips.labels.append(video_labels)

    @staticmethod
    def _get_video_list(dataset_path: str) -> List[str]:
        """Get list of all videos in data folder."""
        assert os.path.exists(
            dataset_path
        ), "VideoIter:: failed to locate: `{}'".format(dataset_path)
        vid_list = []
        for path, _, files in os.walk(dataset_path):
            for name in files:
                if "mp4" not in name:
                    continue
                vid_list.append(os.path.join(path, name))
        return vid_list

    @staticmethod
    def _parse_annotation(path_to_annotation: str) -> Tuple[dict, dict]:
        """Parse file with labelling."""
        dict_metadata = defaultdict(list)
        dict_types = defaultdict(list)

        with open(path_to_annotation) as file:
            metadata = file.readlines()
            for annotation in metadata:
                # parse annotation
                annotation = annotation.replace("\n", "")
                annotation = annotation.split("  ")
                video_name = annotation[0]
                video_type = annotation[1]
                change_time = float(annotation[3])
                video_borders = (float(annotation[2]), float(annotation[4]))
                dict_metadata[video_name].append(
                    (video_borders[0], change_time, video_borders[1])
                )
                dict_types[video_name].append(video_type)
        return dict_metadata, dict_types

    def _equal_sampling(self, downsample_normal=None) -> List[int]:
        """Balance sets with and without changes.

        The equal number of clips are sampled from each video (if possible).
        """
        def _get_indexes(dict_normal: Dict[str, int], dict_cp: Dict[str, int]):
            """Get indexes of clips from totally normal video and from video with anomaly."""
            # TODO: Unnecessary use of a comprehension
            _cp_idxs = list(np.concatenate([idx for idx in dict_cp.values()]))

            _normal_idxs = []
            _normal_cp_idxs = []

            for key, value in zip(dict_normal.keys(), dict_normal.values()):
                if key in dict_cp.keys():
                    _normal_cp_idxs.extend(value)
                else:
                    _normal_idxs.extend(value)
            return _cp_idxs, _normal_idxs, _normal_cp_idxs

        def _uniform_sampling(paths: str, dict_for_sampling: Dict[str, List[int]], max_size: int):
            """Uniform sampling  of clips (with determinate step size)."""
            _sample_idxs = []
            for path in paths:
                idxs = dict_for_sampling[path]
                if len(idxs) > max_size:
                    step = len(idxs) // max_size
                    _sample_idxs.extend(idxs[::step][:max_size])
                else:
                    _sample_idxs.extend(idxs[:max_size])
            return _sample_idxs

        def _random_sampling(
            idxs_for_sampling: np.array, max_size: int, random_seed: int = 123
        ):
            """Random sampling  of clips."""
            np.random.seed(random_seed)
            random.seed(random_seed)
            _sample_idxs = random.choices(idxs_for_sampling, k=max_size)
            return _sample_idxs

        sample_idxs = []

        cp_paths = self.video_clips.cp_idxs.keys()
        normal_paths = [
            x for x in self.video_clips.normal_idxs.keys() if x not in cp_paths
        ]
        normal_cp_paths = [
            x for x in self.video_clips.normal_idxs.keys() if x in cp_paths
        ]

        cp_idxs, normal_idxs, normal_cp_idxs = _get_indexes(
            self.video_clips.normal_idxs, self.video_clips.cp_idxs
        )
        cp_number = len(cp_idxs)

        if downsample_normal is not None:
            max_samples = downsample_normal
        else:
            max_samples = cp_number

        # sample ~50% of normal clips from video with change point
        if len(cp_idxs) > len(normal_cp_paths):
            normal_from_cp = _uniform_sampling(
                paths=normal_cp_paths,
                dict_for_sampling=self.video_clips.normal_idxs,
                max_size=max_samples // (len(normal_cp_paths) * 2),
            )
        else:
            print("Equal sampling is impossible, do random sampling.")
            normal_from_cp = _random_sampling(
                idxs_for_sampling=normal_cp_idxs, max_size=max_samples // 2
            )

        # sample ~50% of normal clips from normal video
        max_rest = max_samples - len(normal_from_cp)

        if max_rest > len(self.video_clips.normal_idxs.keys()):
            normal = _uniform_sampling(
                paths=normal_paths,
                dict_for_sampling=self.video_clips.normal_idxs,
                max_size=max_rest // len(normal_paths),
            )

        else:
            print("Equal sampling is impossible, do random sampling.")
            normal = _random_sampling(
                idxs_for_sampling=normal_idxs, max_size=max_rest
            )

        # sometimes it's still not enough because of different video length
        if len(normal_from_cp) + len(normal) < max_samples:
            extra = _random_sampling(
                idxs_for_sampling=normal_idxs + normal_cp_idxs,
                max_size=max_samples - len(normal_from_cp) - len(normal),
            )
        else:
            extra = []
        sample_idxs = cp_idxs + normal_from_cp + normal + extra
        return sample_idxs

    
class SyntheticNormalDataset(Dataset):

    def __init__(self, seq_len: int, num: int, D=1, random_seed=123):

        super().__init__()

        self.data, self.labels = SyntheticNormalDataset.generate_synthetic_nD_data(seq_len, num, 
                                                                                   D=D, random_seed=123)
    def __len__(self) -> int:
        """Get datasets' length.

        :return: length of dataset
        """
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[np.array, np.array]:

        return self.data[idx], self.labels[idx]

    @staticmethod
    def generate_synthetic_nD_data(seq_len, num, D=1, random_seed=123, multi_dist=False):
        fix_seeds(random_seed)

        idxs_changes = torch.randint(1, seq_len, (num // 2, ))
        
        data = []
        labels = []

        for idx in idxs_changes:
            mu = torch.randint(1, 100, (2, ))
            
            while mu[0] == mu[1]:
                mu = torch.randint(1, 100, (2, ))
            
            if not multi_dist:
                mu[0] = 1 
           
            m = MultivariateNormal(mu[0] * torch.ones(D), torch.eye(D))    
            x1 = []
            for _ in range(seq_len):
                x1_ = m.sample()
                x1.append(x1_)
            x1 = torch.stack(x1)

            m = MultivariateNormal(mu[1] * torch.ones(D), torch.eye(D))    
            x2 = []
            for _ in range(seq_len):
                x2_ = m.sample()
                x2.append(x2_)
            x2 = torch.stack(x2)            
                       
            x = torch.cat([x1[:idx], x2[idx:]])
            label = torch.cat([torch.zeros(idx), torch.ones(seq_len-idx)])
            data.append(x)
            labels.append(label)

        for idx in range(0, num - len(idxs_changes)):            
            m = MultivariateNormal(torch.ones(D), torch.eye(D))    
            x = []
            for _ in range(seq_len):
                x_ = m.sample()
                x.append(x_)
            x = torch.stack(x)            
            label = torch.zeros(seq_len)
            
            data.append(x)
            labels.append(label)

        return data, labels
