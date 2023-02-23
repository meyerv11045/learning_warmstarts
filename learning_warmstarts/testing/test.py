import torch
from casadi import *
from learning_warmstarts.neural_nets.models import FFNet
from learning_warmstarts import problem_specs, solver_options
from learning_warmstarts.opti_problem.gen_problem import generate_problem
from learning_warmstarts.opti_problem.gen_dynamics import generate_list_dynamics, generate_MX_dynamics
from learning_warmstarts.dataset.benchmark_dataset import BenchmarkDataset
from learning_warmstarts.dataset.obst_avoidance_dataset import ObstacleAvoidanceDataset
from learning_warmstarts.viz import Visualizations

class Test:
    """ Base class to be derived for different kinds of testing/metrics
        Sets up problem and loads in all the needed problem specs
    """

    def __init__(self, hidden_layers, predict_duals):
            self.LANE_WIDTH = problem_specs['lane_width']
            self.CAR_LENGTH = problem_specs['car_length']
            self.CAR_WIDTH = problem_specs['car_width']
            self.CAR_HORIZON = problem_specs['car_horizon']
            self.N_OBST = problem_specs['num_obstacles']
            self.N_CONSTRAINTS = problem_specs['num_constraints']

            self.N = problem_specs['control_intervals']
            self.T = problem_specs['interval_duration']

            self.list_dynamics = generate_list_dynamics(problem_specs['dynamics'])
            self.dynamics = generate_MX_dynamics(problem_specs['dynamics'])

            self.problem, self.x_var, self.u_var, self.x0_param, self.obstacle_param, _ = generate_problem(self.N, self.T, self.LANE_WIDTH, self.N_OBST, self.CAR_WIDTH, self.dynamics)

            inputs = 4 + 3 * problem_specs['num_obstacles']
            if predict_duals:
                outputs = 6 * problem_specs['control_intervals'] + problem_specs['num_constraints']
            else:       
                outputs = 6 * problem_specs['control_intervals'] 

            self.shape = [inputs] + hidden_layers + [outputs]
            
            self.predict_duals = predict_duals

            self.viz = Visualizations()
    
    def load_model(model_path, shape, activation_fn):
        """ Loads a PyTorch model saved as a state dictionary to a .pt file
        
        Arguments:
            model_path: Path to the model from the root of the project or absolute path
        """
        if activation_fn == 'relu':
            act_fn = torch.nn.functional.relu
        elif activation_fn == 'tanh':
            act_fn = torch.nn.functional.tanh
        elif activation_fn == 'leaky_relu':
            act_fn = torch.nn.functional.leaky_relu
        else:
            raise NotImplementedError('Activation function not supported')
            
        model = FFNet(shape, act_fn)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    def load_data(self, dataset_type, bench_folder = None, train_folder = None, test_folder = None):
        if dataset_type == 'benchmark' and bench_folder:
            dataset = BenchmarkDataset(bench_folder)
            
            return iter(dataset)

        elif dataset_type == 'obstacle_avoidance' and train_folder and test_folder:
            train_data = ObstacleAvoidanceDataset(train_folder, False)
            test_data = ObstacleAvoidanceDataset(test_folder, False)

            train_loader = torch.utils.data.DataLoader(train_data, batch_size=None, batch_sampler=None)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=None, batch_sampler=None)
            return train_loader, test_loader
        
        else:
            raise NotImplementedError('Dataset type does not exist')


    def set_solver_params(self, use_warmstart_params):
        """ Sets the solver and associated parameters
            If how params are stored is changed in future, 
            make any needed changes here
        
        Arguments:
            use_warmstart_params:   bool indicating whether to use the warmstart params
                                    or the default params with a modified print level
        """
        
        if use_warmstart_params:
            self.problem.solver('ipopt', {}, solver_options['warmstart'])
        else:    
            self.problem.solver('ipopt', {},{'print_level': 0})

    def solve(self):
        """ Solves the setup problem and returns the basic stats and the solved variables
            Can be overridden in children classes for custom solve behavior
        """
        solution = self.problem.solve()
        
        x = solution.value(self.x_var).tolist()
        u = solution.value(self.u_var).tolist()
        lam_g = solution.value(self.problem.lam_g).tolist()

        return (solution.stats()['iter_count'], solution.stats()['t_wall_total']), (x, u, lam_g)

    def viz_warmstart_vs_solution(self, x0, obstacles, warmstart_x, solution_x, filename, iters, time):
        self.viz.save_compared_trajectories(x0, obstacles, warmstart_x, solution_x, filename, iters, time)
        
    def run(self):
        raise NotImplementedError('run() not implemented')