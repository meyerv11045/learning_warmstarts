import argparse
from learning_warmstarts.dataset.benchmark_dataset import BenchmarkDataset
import numpy as np
from learning_warmstarts.neural_nets.loss_fns import ConstraintLoss
from learning_warmstarts import problem_specs
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', required=True)
    parser.add_argument('-n', '--control_intervals', required=True, type=int)
    parser.add_argument('-no', '--n_obstacles', required=True, type=int)
    parser.add_argument('-nc', '--n_constraints', required=True, type=int)
    parser.add_argument('-l', '--loss_fn', default='mse')
    args = parser.parse_args()

    dataset = BenchmarkDataset(args.dataset, args.control_intervals, args.n_obstacles, args.n_constraints)
    
    m = len(dataset)
    running_error = 0

    running_error_no_tail = 0
    u_tail_error = 0
    x_tail_error = 0
    u_no_tail_error = 0
    x_no_tail_error = 0

    x_error = 0
    u_error = 0

    lin_a_error = 0
    ang_a_error = 0 

    criterion = ConstraintLoss(problem_specs) if args.loss_fn == 'custom' else torch.nn.MSELoss()
    for sample in iter(dataset):
        
        # prev_x has already had the dynamics applied to extend it to the proper length
        # prev_u has already had a 0 control extension
        
        prev_no_tail = torch.FloatTensor(sample['prev_x'][:-4] + sample['prev_u'][:-2])
        gt_no_tail = torch.FloatTensor(sample['gt_x'][:-4] + sample['gt_u'][:-2])

        px = torch.FloatTensor(sample['prev_x'])
        gx = torch.FloatTensor(sample['gt_x'])        
        x_error += criterion(px,gx)

        px_no_tail = torch.FloatTensor(sample['prev_x'][:-4])
        gx_no_tail = torch.FloatTensor(sample['gt_x'][:-4])

        x_no_tail_error += criterion(px_no_tail, gx_no_tail)


        px_tail = torch.FloatTensor(sample['prev_x'][-4:])
        gx_tail = torch.FloatTensor(sample['gt_x'][-4:])

        x_tail_error += criterion(px_tail, gx_tail)
        
        pu = torch.FloatTensor(sample['prev_u'])
        gu = torch.FloatTensor(sample['gt_u'])
        u_error += criterion(pu, gu)        

        pu_lina = torch.empty(40,1)
        pu_anga = torch.empty(40,1)

        gu_lina = torch.empty(40,1)
        gu_anga = torch.empty(40,1)
        j = 0
        for i in range(0,len(pu) -1,2):
            pu_lina[j] = pu[i]
            pu_anga[j] = pu[i+1]

            gu_lina[j] = gu[i]
            gu_anga[j] = gu[i+1]
            j += 1

        lin_a_error += criterion(pu_lina, gu_lina)
        ang_a_error += criterion(pu_anga, gu_anga)

        pu_no_tail = torch.FloatTensor(sample['prev_u'][:-2])
        gu_no_tail = torch.FloatTensor(sample['gt_u'][:-2])

        u_no_tail_error += criterion(pu_no_tail, gu_no_tail)

        pu_tail = torch.FloatTensor(sample['prev_u'][-2:])
        gu_tail = torch.FloatTensor(sample['gt_u'][-2:])

        running_error_no_tail += criterion(prev_no_tail, gt_no_tail)
        
        u_tail_error += criterion(pu_tail, gu_tail)

        prev = torch.FloatTensor(sample['prev_x'] + sample['prev_u'])
        gt= torch.FloatTensor(sample['gt_x'] + sample['gt_u'])

        running_error += criterion(prev,gt)
    
    print(f'x + u: \t {running_error / m:4f}')
    print(f'x + u (no tail): \t {running_error_no_tail/m:4f}')
    print(f'x:\t \t {x_error / m:4f}')
    print(f'x (no tail):\t {x_no_tail_error / m:4f}')
    print(f'x (tail): \t {x_tail_error / m:4f}')
    print(f'u: \t \t {u_error / m:4f}')
    print(f'u (no tail): \t {u_no_tail_error / m:4f}')
    print(f'u (tail): \t {u_tail_error / m:4f}')
    print()
    print(f'lin a: \t {lin_a_error/m:4f}')
    print(f'ang a: \t {ang_a_error/m:4f}')
    
    print(f'Calculated on a benchmark dataset of size {m}')