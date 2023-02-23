from types import SimpleNamespace

cfg = SimpleNamespace(
    lane_width=20,
    car_length=5,
    car_width=2,
    car_horizon=60,
    n_obstacles=3,
    n_intervals=40,
    interval_dur=0.01,
    max_vel=200,
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
