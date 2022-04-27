"""Module for experiments' dataset methods."""
import os
import random
from typing import Tuple

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from collections import defaultdict
from torchvision.datasets.video_utils import VideoClips
from tqdm import tqdm
import av
import pickle

from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)

class CPDDatasets:
    """Class for experiments' datasets."""

    def __init__(self, experiments_name,
                 random_seed=123) -> None:
        """Initialize class.

        :param experiments_name: type of experiments (only mnist available now!)
        """
        super().__init__()
        self.random_seed = random_seed
        #TODO make
        if experiments_name in [
            "explosion",
            "road_accidents"
        ]:
            self.experiments_name = experiments_name
        else:
            raise ValueError("Wrong experiment_name {}.".format(experiments_name))

    def get_dataset_(self) -> Tuple[Dataset, Dataset]:
        """Load experiments' dataset."""

        if self.experiments_name == "explosion":
            path_to_data = "data/explosion/"
            path_to_train_annotaion = path_to_data + "UCF_train_time_markup.txt"
            path_to_test_annotaion = path_to_data + "UCF_test_time_markup.txt"
            
        elif self.experiments_name == "road_accidents":
            path_to_data = "data/road_accidents/"
            path_to_train_annotaion = path_to_data + "UCF_road_train_time_markup.txt"
            path_to_test_annotaion = path_to_data + "UCF_road_test_time_markup.txt"

            # https://pytorch.org/hub/facebookresearch_pytorchvideo_resnet/

        #TODO try another transforms
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        side_size = 256
        crop_size = 256

        transform = Compose([
            Lambda(lambda x: x / 255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(size=side_size),
            CenterCropVideo(crop_size=(crop_size, crop_size))
        ])

        train_dataset = UCFVideoDataset(clip_length_in_frames=16, step_between_clips=5,
                                        path_to_data=path_to_data,
                                        path_to_annotation=path_to_train_annotaion,
                                        video_transform=transform,
                                        num_workers=0, fps=30, sampler='equal')
        test_dataset = UCFVideoDataset(clip_length_in_frames=16, step_between_clips=16,
                                       path_to_data=path_to_data,
                                       path_to_annotation=path_to_test_annotaion,
                                       video_transform=transform,
                                       num_workers=0, fps=30, sampler='downsample_norm')
            
        return train_dataset, test_dataset

class UCFVideoDataset(Dataset):
    def __init__(self,
                 clip_length_in_frames,
                 step_between_clips,
                 path_to_data,
                 path_to_annotation,
                 video_transform=None,
                 num_workers=0, fps=30, sampler='all'):

        super().__init__()

        self.clip_length_in_frames = clip_length_in_frames
        self.step_between_clips = step_between_clips
        self.num_workers = num_workers
        self.fps = fps

        # IO
        self.path_to_data = path_to_data
        self.video_list = self._get_video_list(dataset_path=self.path_to_data)

        # annotation loading
        dict_metadata, dict_types = self._parce_annotation(path_to_annotation)

        path_to_clips = '{}_clips_len_{}_step_{}_fps_{}.pth'.format(self.path_to_data.split('/')[1],
                                                                    self.clip_length_in_frames,
                                                                    self.step_between_clips,
                                                                    self.fps)
        if 'train' in path_to_annotation:
            path_to_clips = 'saves/train_' + path_to_clips
        else:
            path_to_clips = 'saves/test_' + path_to_clips

        if os.path.exists(path_to_clips):
            with open(path_to_clips, 'rb') as clips:
                self.video_clips = pickle.load(clips)
        else:
            # data loading
            self.video_clips = VideoClips(video_paths=self.video_list,
                                          clip_length_in_frames=self.clip_length_in_frames,
                                          frames_between_clips=self.step_between_clips,
                                          frame_rate=self.fps,
                                          num_workers=self.num_workers)

            # labelling
            self.video_clips.compute_clips(self.clip_length_in_frames, self.step_between_clips, self.fps)
            self._set_labels_to_clip(dict_metadata)

        # transforms and samplers
        self.video_transform = video_transform
        self.sampler = sampler

        # save dataset
        if not os.path.exists(path_to_clips):
            with open(path_to_clips, 'wb') as clips:
                pickle.dump(self.video_clips, clips, protocol=pickle.HIGHEST_PROTOCOL)

        if sampler == 'equal':
            self.video_clips.valid_idxs = self._equal_sampling()
        elif sampler == 'downsample_norm':
            self.video_clips.valid_idxs = self._equal_sampling(downsample_normal=300)
        elif sampler == 'all':
            pass
        else:
            raise ValueError('Wrong type of sampling')

    def __len__(self):
        return len(self.video_clips.valid_idxs)

    def __getitem__(self, idx):
        idx = self.video_clips.valid_idxs[idx]
        video, _, _, _ = self.video_clips.get_clip(idx)
        video_idx, clip_idx = self.video_clips.get_clip_location(idx)
        video_path = self.video_clips.video_paths[video_idx]
        label = np.array(self.video_clips.labels[video_idx][clip_idx], dtype=int)
        # shoud be channel, seq_len, height, width
        video = video.permute(3, 0, 1, 2)
        if self.video_transform is not None:
            video = self.video_transform(video)
        return video, label

    def _set_labels_to_clip(self, dict_metadata):
        # self.video_clips.labels = []
        self.video_clips.labels = []
        self.video_clips.valid_idxs = list(range(0, len(self.video_clips)))
        self.video_clips.normal_idxs = defaultdict(list)
        self.video_clips.cp_idxs = defaultdict(list)

        global_clip_idx = -1
        for video_idx, vid_clips in tqdm(enumerate(self.video_clips.clips), total=len(self.video_clips.clips)):
            video_labels = []

            video_path = self.video_clips.video_paths[video_idx]
            video_name = os.path.basename(video_path)

            # get time unit to map frame with its time appearance
            time_unit = av.open(video_path, metadata_errors='ignore').streams[0].time_base

            # get start, change point, end from annotation
            annotated_time = dict_metadata[video_name]

            cp_video_idx = len(vid_clips)

            for clip_idx, clip in enumerate(vid_clips):
                clip_labels = []
                global_clip_idx += 1

                clip_start = float(time_unit * clip[0].item())
                clip_end = float(time_unit * clip[-1].item())

                not_suit_flag = True
                for start_time, change_point, end_time in annotated_time:
                    if end_time != -1.0:
                        if (clip_end > end_time) or (clip_start > end_time) or (clip_start < start_time):
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

                if not_suit_flag:
                    # drop clip idx from dataset
                    video_labels.append([])
                    self.video_clips.valid_idxs.remove(global_clip_idx)
                else:
                    if 'Normal' in video_path:
                        clip_labels = list(np.zeros(len(clip)))
                        self.video_clips.normal_idxs[video_path].append(global_clip_idx)
                    else:
                        for frame in clip:
                            frame_time = float(time_unit * frame.item())
                            # due to different rounding while moving from frame to time
                            # the true change point is delayed by ~1 frame
                            # so, we've added the little margin
                            if (frame_time >= change_point - 1e-6) and (change_point != -1.0):
                                clip_labels.append(1)
                            else:
                                clip_labels.append(0)
                        if sum(clip_labels) > 0:
                            self.video_clips.cp_idxs[video_path].append(global_clip_idx)
                        else:
                            self.video_clips.normal_idxs[video_path].append(global_clip_idx)
                    video_labels.append(clip_labels)
            self.video_clips.labels.append(video_labels)
        return self

    def _get_video_list(self, dataset_path):
        assert os.path.exists(dataset_path), "VideoIter:: failed to locate: `{}'".format(dataset_path)
        vid_list = []
        for path, subdirs, files in os.walk(dataset_path):
            for name in files:
                if 'mp4' not in name:
                    continue
                vid_list.append(os.path.join(path, name))
        return vid_list

    def _parce_annotation(self, path_to_annotation):
        dict_metadata = defaultdict(list)
        dict_types = defaultdict(list)

        with open(path_to_annotation) as f:
            metadata = f.readlines()
            for f in metadata:
                # parce annotation
                f = f.replace('\n', '')
                f = f.split('  ')
                video_name = f[0]
                video_type = f[1]
                change_time = float(f[3])
                video_borders = (float(f[2]), float(f[4]))
                dict_metadata[video_name].append((video_borders[0], change_time, video_borders[1]))
                dict_types[video_name].append(video_type)
        return dict_metadata, dict_types

    def _equal_sampling(self, downsample_normal=None):
        """Balance sets with and without changes so that the equal number
        of clips are sampled from each video (if possible)."""

        def _get_indexes(dict_normal, dict_cp):
            """Get indexes of clips from totally normal video and from video with anomaly"""
            cp_idxs = list(np.concatenate([idx for idx in dict_cp.values()]))

            normal_idxs = []
            normal_cp_idxs = []

            for k, v in zip(dict_normal.keys(), dict_normal.values()):
                if k in dict_cp.keys():
                    normal_cp_idxs.extend(v)
                else:
                    normal_idxs.extend(v)
            return cp_idxs, normal_idxs, normal_cp_idxs

        def _uniform_sampling(paths, dict_for_sampling, max_samples):
            sample_idxs = []
            for path in paths:
                idxs = dict_for_sampling[path]
                if (len(idxs) > max_samples):
                    step = len(idxs) // max_samples
                    sample_idxs.extend(idxs[::step][:max_samples])
                else:
                    sample_idxs.extend(idxs[:max_samples])
            return sample_idxs

        def _random_sampling(idxs_for_sampling, max_samples, random_seed=123):
            np.random.seed(random_seed)
            random.seed(random_seed)
            sample_idxs = random.choices(idxs_for_sampling, k=max_samples)
            return sample_idxs

        sample_idxs = []

        cp_paths = self.video_clips.cp_idxs.keys()
        normal_paths = [x for x in self.video_clips.normal_idxs.keys() if x not in cp_paths]
        normal_cp_paths = [x for x in self.video_clips.normal_idxs.keys() if x in cp_paths]

        cp_idxs, normal_idxs, normal_cp_idxs = _get_indexes(self.video_clips.normal_idxs, self.video_clips.cp_idxs)
        cp_number = len(cp_idxs)

        if downsample_normal is not None:
            max_samples = downsample_normal
        else:
            max_samples = cp_number

        # sample ~50% of normal clips from video with change point
        if len(cp_idxs) > len(normal_cp_paths):
            normal_from_cp = _uniform_sampling(paths=normal_cp_paths,
                                               dict_for_sampling=self.video_clips.normal_idxs,
                                               max_samples=max_samples // (len(normal_cp_paths) * 2))
        else:
            print('Equal sampling is impossible, do random sampling.')
            normal_from_cp = _random_sampling(idxs_for_sampling=normal_cp_idxs,
                                              max_samples=max_samples // 2)

        # sample ~50% of normal clips from normal video
        max_rest = (max_samples - len(normal_from_cp))

        if max_rest > len(self.video_clips.normal_idxs.keys()):
            normal = _uniform_sampling(paths=normal_paths,
                                       dict_for_sampling=self.video_clips.normal_idxs,
                                       max_samples=max_rest // len(normal_paths))

        else:
            print('Equal sampling is impossible, do random sampling.')
            normal = _random_sampling(idxs_for_sampling=normal_idxs,
                                      max_samples=max_rest)

        # sometimes it's still not enough because of different video length
        if len(normal_from_cp) + len(normal) < max_samples:
            extra = _random_sampling(idxs_for_sampling=normal_idxs + normal_cp_idxs,
                                     max_samples=max_samples - len(normal_from_cp) - len(normal))
        else:
            extra = []
        sample_idxs = cp_idxs + normal_from_cp + normal + extra
        return sample_idxs
