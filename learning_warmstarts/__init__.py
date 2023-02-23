""" Learning Warmstarts Package
    
    Distances- meters 
    Time- seconds
"""

problem_specs = {
    'lane_width': 20,
    'car_length': 5,
    'car_width': 2,
    'car_horizon': 60,   
    'num_obstacles': 3,

    'control_intervals': 40,
    'interval_duration': 0.01,
    'dynamics': 'unicycle',
    'num_constraints': 480,
    
    'max_vel': 200
}

solver_options = {
    'warmstart' : {
        'warm_start_init_point': 'yes',
        'warm_start_bound_frac': 1e-16,
        'warm_start_bound_push': 1e-16,
        'warm_start_mult_bound_push': 1e-16,
        'warm_start_slack_bound_frac': 1e-16,
        'warm_start_slack_bound_push': 1e-16,
        'print_level': 0,
    },
    'print_level' : {
        'print_level': 0,
    }
}