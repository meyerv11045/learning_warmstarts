from types import SimpleNamespace
import math
from learning_warmstarts.dataset.dataset import StateControlDataset
from learning_warmstarts.models import FCNet
from learning_warmstarts.train import train_baseline, train_ipopt_loss, save_results
from learning_warmstarts.utils.logs import setup_logging
from learning_warmstarts.utils.io import read_file, write_obj

cfg = SimpleNamespace(
    lane_width = 20,
    car_length = 4.48, # 14.7 ft is avg car length
    car_width = 1.77, # 5.8 ft is avg car width
    car_horizon = 40,
    dist_b4_obst = 10,
    min_obst_radius = 2,
    max_obst_radius = 3,
    min_theta = -math.pi / 4,
    max_theta = math.pi / 4,
    n_obstacles = 3,
    n_intervals = 40,
    interval_dur = 0.25,
    max_vel = 15.65, # 35 mph
    max_ang_vel = 1, # 1 G = 1 m/s^2 is max force you want to feel in the car
    max_accel = 1,
    rng_seed = 0,
    traj_length = 160,
    controls_length = 80,
    params_length = 9,
    warmstart_opts={
        "warm_start_init_point": "yes",
        "warm_start_bound_frac": 1e-16,
        "warm_start_bound_push": 1e-16,
        "warm_start_mult_bound_push": 1e-16,
        "warm_start_slack_bound_frac": 1e-16,
        "warm_start_slack_bound_push": 1e-16,
        "print_level": 0,
    },
    noprint_opt={"print_level": 0},
)