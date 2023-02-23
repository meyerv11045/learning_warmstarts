import glob
import itertools

import numpy as np
import torch

from learning_warmstarts import problem_specs


class BenchmarkDataset(torch.utils.data.IterableDataset):
    def __init__(self, foldername):
        super(BenchmarkDataset).__init__()
        self.foldername = foldername

        n = problem_specs["control_intervals"]

        self.x0_end_idx = 4
        self.obst_end_idx = 4 + (3 * problem_specs["num_obstacles"])
        self.x_end_idx = self.obst_end_idx + 4 * n
        self.u_end_idx = self.x_end_idx + 2 * n

        self.lamg_end_idx = self.u_end_idx + problem_specs["num_constraints"]
        self.prev_x_end_idx = self.lamg_end_idx + 4 * n
        self.prev_u_end_idx = self.prev_x_end_idx + 2 * n

    def read_file(self, file):
        data = np.load(file, allow_pickle=True)
        for row in range(data.shape[0]):

            yield dict(
                x0=data[row, 0 : self.x0_end_idx],
                obstacles=data[row, self.x0_end_idx : self.obst_end_idx],
                gt_x=data[row, self.obst_end_idx : self.x_end_idx],
                gt_u=data[row, self.x_end_idx : self.u_end_idx],
                gt_lamg=data[row, self.u_end_idx : self.lamg_end_idx],
                prev_x=data[row, self.lamg_end_idx : self.prev_x_end_idx],
                prev_u=data[row, self.prev_x_end_idx : self.prev_u_end_idx],
                prev_lam_g=data[row, self.prev_u_end_idx :],
            )

    def __iter__(self):
        files = glob.glob(f"{self.foldername}/*.npy")
        return itertools.chain.from_iterable(map(self.read_file, files))
