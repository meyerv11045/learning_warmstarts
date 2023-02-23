from casadi import *
import random
import numpy as np
from learning_warmstarts import problem_specs, solver_options
from learning_warmstarts.opti_problem.gen_dynamics import generate_np_dynamics
from learning_warmstarts.opti_problem.gen_obstacles import generate_np_obstacles
from learning_warmstarts.opti_problem.gen_problem import generate_problem_v2

def generate_training_problems(rng_seed, n):
    """ Generate n data samples
        Note: these n samples will be similar to each other
    """
    random.seed(rng_seed)

    n = problem_specs['control_intervals']

    results = np.zeros((n, 4 + 3*problem_specs['num_obstacles'] + 6*n + problem_specs['num_constraints']), dtype=np.float32)
    np_dynamics = generate_np_dynamics(problem_specs['dynamics'])

    problem = generate_problem_v2()
    problem.solver('ipopt', {}, solver_options['print_level'])

    i = 0
    x, u, prev_obsts = None, None, None
    prev_unsolvable = False

    while i < n:
        try:
            if prev_unsolvable:
                print("New obstacles")
                x, u, prev_obsts = None, None, None
                prev_unsolvable = False

            (x0, obst, z, lam_g, stats) = gen_and_solve_problem(problem, np_dynamics, prev_obsts, x, u)

            prev_obsts = obst

            results[i,:] = np.concatenate([x0, obst, z, lam_g])
            i += 1

            print(f'Problem {i} solved in {stats[0]} iters in {stats[1]} seconds')

        except RuntimeError as e:
            print(f'ERROR: {e}')
            prev_unsolvable = True

    return results

def gen_and_solve_problem(problem, np_dynamics, prev_obsts = None, x = None, u = None):
    max_vel = problem_specs['max_vel']
    skip_n_steps = 4
    x0 = x[skip_n_steps * 4: skip_n_steps*4+4] if x else None


    if x and u:
        # Warmstart with part of previously solved for trajectory for faster solving
        x_warmstart = np.zeros(len(x), dtype=np.float32)
        x_warmstart[:len(x) - 4*skip_n_steps] = x[4*skip_n_steps:]

        u_warmstart = np.zeros(len(u), dtype=np.float32)
        u_warmstart[:len(u) - 2*skip_n_steps] = u[2*skip_n_steps:] # zero controls for the nub
        cur_x = x[-4:]
        cur_u = u[-2:]

        start_x_idx = len(x) - 4*skip_n_steps
        start_u_idx = len(u) - 2*skip_n_steps
        for i in range(skip_n_steps):
            # extend the state vector warmstart using the control vector warmstart
            cur_x = np_dynamics(cur_x, cur_u, problem_specs['interval_duration'])
            cur_u = u_warmstart[start_u_idx + 2*i: start_u_idx + 2 + 2*i]
            x_warmstart[start_x_idx + 4*i :start_x_idx + 4 + 4*i] = cur_x


        problem.set_initial(problem.x[[i for i in range(len(x) + len(u))]], np.concatenate([x_warmstart, u_warmstart]))

        obsts = generate_np_obstacles(problem_specs['num_obstacles'], problem_specs['lane_width'], problem_specs['car_horizon'], problem_specs['car_width'], x0, prev_obsts)

        for i in range(0,len(obsts),3):
            obsts[i]   -= x0[0]
            obsts[i+1] -= x0[1]

        x0[0] -= x0[0]
        x0[1] -= x0[1]

    else:
        x0 = get_rand_x0(max_vel)
        obsts = generate_np_obstacles(problem_specs['num_obstacles'], problem_specs['lane_width'], problem_specs['car_horizon'], problem_specs['car_width'], x0, prev_obsts)

    problem.set_value(problem.p[[i for i in range(4 + 3 * problem_specs['num_obstacles'])]], np.concatenate([x0, obsts]))

    solution = problem.solve()
    stats = (solution.stats()['iter_count'], solution.stats()['t_proc_total'])
    return (x0, obsts, solution.value(problem.x), solution.value(problem.lam_g), stats)

def get_rand_x0(max_vel):
    """ Returns a random initial state of the car [x, y, v, theta]

    Arguments:
        lane_width: assumes 0 to lane_width for the y_bounds of initial states
        max_vel:    maximum velocity of an initial state
        max_x:      maximum x position of the car- note the larger, the less dense the training data
    """
    getRand = lambda a,b: (b-a) * random.random() + a

    x0 = np.zeros(4, dtype=np.float32)

    x0[2] = getRand(0, max_vel)
    x0[3] = getRand(-np.pi / 4, np.pi / 4)

    return x0

def generate_benchmark_problems(rng_seed, n):
    """ Generates the training and testing data for a neural network

    Arguments:
        rng_seed:       default of None for random number generator seed based on current timestamp
        n:              number of datapoints to generate for this chunk of benchmark problems
    """
    random.seed(rng_seed)

    n = problem_specs['control_intervals']
    t = problem_specs['interval_duration']

    np_dynamics = generate_np_dynamics(problem_specs['dynamics'])

    problem = generate_problem_v2()
    problem.solver('ipopt', {}, solver_options['print_level'])

    results = np.zeros((n, 4 + 3*problem_specs['num_obstacles'] + 12*n + 2*problem_specs['num_constraints']), dtype=np.float32)

    i = 0
    while i < n:
        try:
            (x0, obstacles, z, prev_lam_g, _) = gen_and_solve_problem(problem, np_dynamics)

            prev_x = z[:4 * n]
            prev_u = z[4 * n:]

            prev_x_warmstart = np.concatenate([prev_x[4:], np_dynamics(prev_x[-4:], prev_u[-2:], t)])
            prev_u_warmstart = np.zeros(len(prev_u), dtype=np.float32)
            prev_u_warmstart[:-2] = prev_u[2:]

            # make initial state for benchmark the next step
            # so we have a previous iteration warmstart to use
            nxt_x0 = np_dynamics(x0,prev_u[:2],t)

            (z_truth, lam_g_truth, stats) = solve_for_ground_truth(problem, x0, obstacles, prev_x_warmstart, prev_u_warmstart)

            results[i,:] = np.concatenate([nxt_x0, obstacles, z_truth, lam_g_truth, prev_x_warmstart, prev_u_warmstart, prev_lam_g])

            print(f'Problem {i+1} solved in {stats[0]} iters in {stats[1]} seconds with warmstart')
            i += 1

        except RuntimeError as e:
            print(f'ERROR: {e}')

    return results

def solve_for_ground_truth(problem, x0, obsts, x_warmstart, u_warmstart):
    problem.set_initial(problem.x, np.concatenate([x_warmstart, u_warmstart]))

    for i in range(0,len(obsts),3):
        obsts[i]   -= x0[0]
        obsts[i+1] -= x0[1]

    x0[0] -= x0[0]
    x0[1] -= x0[1]

    problem.set_value(problem.p[:-1], np.concatenate([x0, obsts]))

    solution = problem.solve()
    stats = (solution.stats()['iter_count'], solution.stats()['t_wall_total'])

    return (solution.value(problem.x), solution.value(problem.lam_g), stats)