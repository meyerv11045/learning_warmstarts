import argparse
from learning_warmstarts import problem_specs
from learning_warmstarts.opti_problem.gen_dynamics import generate_list_dynamics
from learning_warmstarts.viz import Visualizations
from learning_warmstarts.dataset.benchmark_dataset import BenchmarkDataset
from learning_warmstarts.opti_problem.gen_problem import generate_problem
from learning_warmstarts.opti_problem.gen_dynamics import generate_MX_dynamics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_csv')
    parser.add_argument('-o', '--output_folder')
    parser.add_argument('-n', '--num_imgs', type=int)
    parser.add_argument('-t', '--type', type=int)
    args = parser.parse_args()

    viz = Visualizations()
    if args.type == 1:
        """ Visualize the random obstacle generation
        """
        viz.save_obstacle_gen_imgs(args.num_imgs)
    elif args.type == 2:
        """ Visualize trajectories from ones saved to a csv file
        """
        viz.create_imgs_from_csv_results(args.input_csv, args.output_folder, args.num_imgs)
    elif args.type == 3:
        """ Visualize the previous iteration warmstarts compared to the solver's solution and ground truth 
        """
        dynamics = generate_MX_dynamics(problem_specs['dynamics'])

        dataset = BenchmarkDataset(args.input_csv, problem_specs['control_intervals'], problem_specs['num_obstacles'], problem_specs['num_constraints'])
        dataiter = iter(dataset)

        problem, x_var, u_var, x0_param, obst_param, t_param = generate_problem(problem_specs['control_intervals'], problem_specs['interval_duration'], problem_specs['lane_width'], problem_specs['num_obstacles'], problem_specs['car_width'], dynamics)

        from learning_warmstarts.neural_nets.models import FFNet
        import torch

        model = FFNet([13,512,1024,2048,240],torch.nn.functional.relu)
        model.load_state_dict(torch.load('good_model.pt'))

        problem.solver('ipopt', {},{'print_level': 5})
        for i,sample in enumerate(dataiter):
            if i > args.num_imgs:
                break 

            x0 = sample['x0']
            obstacles = sample['obstacles']
            # inputs = torch.FloatTensor(x0 + obstacles)

            # prediction = model.forward(inputs)

            prev_x = sample['prev_x']
            prev_u = sample['prev_u']

            # problem.set_value(x0_param[[0,1,2,3]], x0)
            # problem.set_value(obst_param[[i for i in range(obst_param.shape[0])]], obstacles)

            # problem.set_initial(x_var, prev_x)
            # problem.set_initial(u_var, prev_u)

            # problem.set_initial(x_var, prediction[:160].tolist())
            # problem.set_initial(u_var, prediction[160:].tolist())

            # solution = problem.solve()
            # solved_x = solution.value(x_var)
            # iters = solution.stats()['iter_count']
            # time = solution.stats()['t_proc_total']

            # viz.save_compared_trajectories(x0,obstacles,prediction[:160].tolist(),solved_x,f'prev_sol{i}.jpg',iters,time)
            
            lst_dynamics = generate_list_dynamics('unicycle')
            
            unrolled_x = []
            cur_x = x0
            for j in range(0, len(prev_u), 2):
                cur_x = lst_dynamics(cur_x,prev_u[j:j+2], 0.01)
                unrolled_x.extend(cur_x)
            
            viz.save_compared_trajectories(x0,obstacles,unrolled_x,sample['gt_x'],f'unrollled{i}.jpg')