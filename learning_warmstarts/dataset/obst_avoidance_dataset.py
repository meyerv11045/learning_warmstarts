import torch
import numpy as np
from learning_warmstarts import problem_specs
import glob
import itertools

class ObstacleAvoidanceDataset(torch.utils.data.IterableDataset):
    def __init__(self, foldername, load_target_duals = False):
        super(ObstacleAvoidanceDataset).__init__()
        self.foldername = foldername
        self.n_primary_vars = 6 * problem_specs['control_intervals']
        self.n_params = 4 + 3 * problem_specs['num_obstacles']

        self.load_target_duals = load_target_duals

    def read_file(self, file):
        data = np.load(file, allow_pickle=True)
        for row in range(data.shape[0]):
            
            x = data[row, 0:self.n_params]
            if self.load_target_duals:
                y = data[row, self.n_params:]
            else:
                y = data[row, self.n_params:self.n_params + self.n_primary_vars]
            yield (x,y)

    def __iter__(self):
        """ Returns a sample from the dataset at the specified index
            as an (x,y) tuple of input and label
        """
        files = glob.glob(f'{self.foldername}/chunk*.npy')
        return itertools.chain.from_iterable(map(self.read_file, files)) #itertools.cycle(files)


class ObstacleAvoidanceBatchDataset(torch.utils.data.IterableDataset):
    def __init__(self, foldername, load_target_duals = False):
        super(ObstacleAvoidanceDataset).__init__()
        self.foldername = foldername
        self.n_primary_vars = 6 * problem_specs['control_intervals']
        self.n_constraints = problem_specs['num_constraints']
        self.n_params = 4 + 3 * problem_specs['num_obstacles']
        self.load_target_duals = load_target_duals

    def read_file(self, file):
        data = np.load(file, allow_pickle=True)
        x = data[:, 0:self.n_params]
        
        if self.load_target_duals:
            y = data[:, self.n_params: ]
        else:
            y = data[:, self.n_params:self.n_params+self.n_primary_vars]
        yield (x,y)

    def __iter__(self):
        """ Returns a sample from the dataset at the specified index
            as an (x,y) tuple of input and label
        """
        files = glob.glob(f'{self.foldername}/chunk*.npy')
        return itertools.chain.from_iterable(map(self.read_file, files)) #itertools.cycle(files)



class PreprocessedObstacleAvoidanceDataset(torch.utils.data.IterableDataset):
    def __init__(self, foldername, load_target_duals = False):
        super(PreprocessedObstacleAvoidanceDataset).__init__()
        self.foldername = foldername
        self.n_primary_vars = 6 * problem_specs['control_intervals']

        self.load_target_duals = load_target_duals

    def read_file(self, file):
        data = np.load(file, allow_pickle=True)
        for row in range(data.shape[0]):
            x = data[row, :12]
            if self.load_target_duals:
                y = data[row, 12:]
            else:
                y = data[row, 12:12+self.n_primary_vars]
            yield (x,y)

    def __iter__(self):
        """ Returns a sample from the dataset at the specified index
            as an (x,y) tuple of input and label
        """
        files = glob.glob(f'{self.foldername}/chunk*.npy')
        return itertools.chain.from_iterable(map(self.read_file, files)) #itertools.cycle(files)


class PreprocessedObstacleAvoidanceBatchDataset(torch.utils.data.IterableDataset):
    def __init__(self, foldername, load_target_duals = False):
        super(PreprocessedObstacleAvoidanceBatchDataset).__init__()
        self.foldername = foldername
        self.n_primary_vars = 6 * problem_specs['control_intervals']
        self.n_constraints = problem_specs['num_constraints']
        self.load_target_duals = load_target_duals

    def read_file(self, file):
        """ assume preprocessed input data 
            sin(theta), cos(theta), velocity, normalized obstacles, x target, u target, lamg target
        """
        data = np.load(file, allow_pickle=True)
        x = data[:, 0:12]
        
        if self.load_target_duals:
            y = data[:,12: ]
            
        else:
            y = data[:, 12:12+self.n_primary_vars]
        yield (x,y)

    def __iter__(self):
        """ Returns a sample from the dataset at the specified index
            as an (x,y) tuple of input and label
        """
        files = glob.glob(f'{self.foldername}/chunk*.npy')
        return itertools.chain.from_iterable(map(self.read_file, files)) #itertools.cycle(files)