# data_utils/all_datasets.py


import os
import h5py
import math
import random
import numpy as np
import torch
import torch.nn.functional as F

# import torchdata.datapipes as dp
# from torchdata.datapipes.iter import IterDataPipe

import torch.utils.data.datapipes as dp
from torch.utils.data.datapipes.datapipe import IterDataPipe

from logging import getLogger

logger = getLogger()

DatasetIdx = {
    "react_diff": 0,
    "shallow_water": 1,
    "incom_ns": 2,
    "com_ns": 3,
    "incom_ns_arena": 4,
    "incom_ns_arena_u": 5,
    "cfdbench": 6,
}


class myIterDp(IterDataPipe):
    """
    Base class for all iterable datasets, and contains some shared helper methods.
    """

    def __init__(self, params, symbol_env, split="train", train=True):
        super().__init__()

        # general initialization, should be called by all subclasses

        # self.train = split == "train"
        self.train = train
        self.params = params
        self.symbol_env = symbol_env
        self.split = split

        self.num_workers = params.num_workers if train else params.num_workers_eval
        self.local_rank = params.local_rank
        self.n_gpu_per_node = params.n_gpu_per_node

        self.t_num = params.data.t_num
        self.x_num = params.data.x_num

        # if params.overfit_test:
        #     self.random_start = params.data.random_start["test"]
        # else:
        #     self.random_start = params.data.random_start[split]

        # I changed the main code to always use the same random start for all splits, so that we can test the effect of random start more easily. We can change this back later if needed.
        self.random_start = params.data.random_start

        self.rng = None
        self.type_label = ""
        self.fully_shuffled = False
        self.symbol_ids = None

    def init_rng(self):
        """
        Initialize different random generator for each worker.
        """
        if self.rng is not None:
            return

        worker_id = self.get_worker_id()
        self.worker_id = worker_id
        params = self.params
        if self.train:
            # base_seed = params.base_seed
            base_seed = np.random.randint(1_000_000_000)  # ensure seed is different for each epoch

            seed = [worker_id, DatasetIdx[self.type_label], params.global_rank, base_seed]
            self.rng = np.random.default_rng(seed)
            # logger.info(f"Initialize random generator with seed {seed} (worker, dataset, rank, base_seed)")
        else:
            seed = [worker_id, DatasetIdx[self.type_label], params.global_rank, params.test_seed]
            self.rng = np.random.default_rng(seed)
            # logger.info(f"Initialize random generator with seed {seed} (worker, dataset, rank, test_seed)")

    def get_worker_id(self):
        worker_info = torch.utils.data.get_worker_info()
        return 0 if worker_info is None else worker_info.id

    def augment_data(self, data: np.ndarray):
        """
        data: (t_num, x_num, x_num, data_dim)
        """
        if True: #self.train:
            # self.init_rng()
            if self.params.noise > 0:
                # add noise
                gamma = self.params.noise

                if self.params.noise_type == "multiplicative":
                    cur_noise = self.rng.normal(size=data.shape).astype(np.single)
                    data = data + gamma * np.abs(data) * cur_noise
                elif self.params.noise_type == "additive":
                    cur_noise = self.rng.normal(size=data.shape).astype(np.single)
                    eps = 1e-6
                    sigma = gamma * np.linalg.norm(data) / (np.linalg.norm(cur_noise) + eps)
                    data = data + sigma * cur_noise

            if self.params.flip:
                # flip data
                flip = self.rng.choice(4)
                if flip == 1:
                    data = np.flip(data, axis=1)
                elif flip == 2:
                    data = np.flip(data, axis=2)
                elif flip == 3:
                    data = np.flip(data, axis=(1, 2))

            if self.params.rotate:
                # rotate data
                rot = self.rng.choice(4)
                if rot > 0:
                    data = np.rot90(data, axes=(1, 2), k=rot)

        return np.ascontiguousarray(data)

    def get_iter_range(self, total_len):
        # split data based on train/val/test ratio and number of workers
        ratio = self.params.data.train_val_test_ratio
        start1 = int(total_len * ratio[0])
        start2 = int(total_len * (ratio[0] + ratio[1]))
        start3 = int(total_len * (ratio[0] + ratio[1] + ratio[2]))
        if self.split == "train":
            start = 0
            end = start1
        elif self.split == "val":
            start = start1
            end = start2
        else:  # test
            start = start2
            end = start3

        if self.num_workers <= 1:
            # return start, end
            return np.arange(start, end)
        else:
            # subdivide based on number of workers
            return np.arange(start + self.worker_id, end, self.num_workers)

    def sample_initial_time(self, max_len):
        data_limit = max_len - self.t_num * self.t_step
        start_limit = self.random_start.start_max
        # start_limit = self.params.data.random_start.start_max
        if start_limit > 0:
            data_limit = min(data_limit, start_limit)
        if data_limit <= 0:
            return 0
        else:
            return self.rng.integers(0, data_limit)



class ShallowWater2D(myIterDp):
    """
    PDEBench 2D shallow_water dataset.
        size:  1000
        t_num: 101            [0, 1] dt=0.01
        x_num: (128, 128)     (-2.5, 2.5)
        data_dim: 1
        bc: neumann

    Dataset structure:
    0000 - 0999
        data: (101, 128, 128, 1)
        grid
            t: (101,)
            x: (128,)
            y: (128,)
    """

    def __init__(self, params, symbol_env, split="train", train=True):
        super().__init__(params, symbol_env, split, train)

        # dataset specific initialization

        self.type_label = "shallow_water"

        self.t_step = params.data.shallow_water.t_step
        self.x_step = params.data.shallow_water.x_num // params.data.x_num
        self.data_path = params.data.shallow_water.data_path
        self.fully_shuffled = True  # no need to shuffle since we shuffle in __iter__

        if self.params.symbol.symbol_input:
            tree = self.symbol_env.generator.get_tree(self.type_label)
            tree_encoded = self.symbol_env.equation_encoder.encode(tree)
            symbol_input = self.symbol_env.word_to_idx([tree_encoded], float_input=False)[0]
            self.symbol_ids = symbol_input

        if not self.params.data.tie_fields:
            self.c_mask = torch.Tensor(params.data[self.type_label].c_mask)
            self.c_mask_bool = self.c_mask.bool()

    def __len__(self):
        # This method is required by DataLoader to determine the dataset length
        with h5py.File(self.data_path, "r") as hf:
            total_samples = len(hf)
        # get_iter_range returns an array, so its length is the number of samples for this worker/split
        iter_range = self.get_iter_range(total_samples)
        # Apply the same restriction as in __iter__ for consistency
        max_samples = 20 # Defined in __iter__ for testing
        if len(iter_range) > max_samples:
            iter_range = iter_range[:max_samples]
        return len(iter_range)

    def __iter__(self):
        self.init_rng()

        with h5py.File(self.data_path, "r") as hf:
            iter_range = self.get_iter_range(len(hf))[self.local_rank :: self.n_gpu_per_node]

            # Restrict the iter range due to limited RAM during testing. Remove this restriction for full training.
            # LIMIT DATASET SIZE HERE
            max_samples = 20
            iter_range = iter_range[:max_samples]
            # Remove the lines above for full training.


            if self.train:
                iter_range = self.rng.permutation(iter_range)

            for i in iter_range:
                sample = hf[f"{i:04d}"]
                t0 = self.sample_initial_time(sample["data"].shape[0]) if self.random_start else 0

                data = sample["data"][
                    t0 : (t0 + self.t_num * self.t_step) : self.t_step, :: self.x_step, :: self.x_step
                ]  # (t_num, x_num, x_num, 1)

                d = {}
                # d["type"] = self.type_label

                data = self.augment_data(data)
                data = torch.from_numpy(data).float()

                if not self.params.data.tie_fields:
                    d["data_mask"] = self.c_mask
                    nt, nx, ny, _ = data.size()
                    tmp = torch.zeros(nt, nx, ny, self.params.data.max_output_dimension, dtype=data.dtype)
                    tmp[..., self.c_mask_bool] = data
                    data = tmp

                d["data"] = data

                if self.params.use_raw_time:
                    t_grid = sample["grid"]["t"][t0 : (t0 + self.t_num * self.t_step) : self.t_step]  # (t_num, )
                    t_grid = torch.from_numpy(t_grid).float()
                    d["t"] = t_grid

                if self.params.symbol.symbol_input:
                    d["symbol_input"] = self.symbol_ids

                # x_grid = sample["grid"]["x"][::self.x_step]  # (x_num, )
                # y_grid = sample["grid"]["y"][::self.x_step]  # (x_num, )


                yield d
