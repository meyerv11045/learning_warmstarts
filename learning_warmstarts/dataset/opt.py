import numpy as np
from casadi import MX, Opti, cos, sin


def dynamics(x: MX, u: MX, t: float) -> MX:
    """Casadi MX unicycle dynamics
    x_k = f(x_{k-1}, u_k)
    """
    nxt_x = MX(4, 1)
    nxt_x[0] = x[0] + t * x[2] * cos(x[3])
    nxt_x[1] = x[1] + t * x[2] * sin(x[3])
    nxt_x[2] = x[2] + t * u[0]
    nxt_x[3] = x[3] + t * u[1]
    return nxt_x


def setup_problem(cfg, casadi_options={}, solver_options={"print_level": 0}) -> Opti:
    """Setup the optimization problem where the decision variables are the state trajectory
    and control trajectory: [x,y,v,theta] and [accel, ang vel]
    """
    n = cfg.n_intervals

    problem = Opti()
    traj = problem.variable(4 * n)
    u = problem.variable(2 * n)
    x0 = problem.parameter(4)
    problem.set_value(x0, np.array([0.0, 0.0, 0.0, 0.0]))
    obstacles = problem.parameter(3 * cfg.n_obstacles)
    t = problem.parameter()
    problem.set_value(t, cfg.interval_dur)

    ## Select Decision Variables
    select_x = MX(n, 4 * n)
    select_y = MX(n, 4 * n)
    select_v = MX(n, 4 * n)
    select_theta = MX(n, 4 * n)

    j = 0
    for i in range(n):
        select_x[i, j] = 1
        select_y[i, j + 1] = 1
        select_v[i, j + 2] = 1
        select_theta[i, j + 3] = 1
        j += 4

    x_poses = select_x @ traj
    y_poses = select_y @ traj
    velocities = select_v @ traj
    thetas = select_theta @ traj

    select_a = MX(n, 2 * n)
    select_w = MX(n, 2 * n)
    j = 0
    for i in range(n):
        select_a[i, j] = 1
        select_w[i, j + 1] = 1
        j += 2
    accels = select_a @ u
    ang_vels = select_w @ u

    # min accceleration & ang vel, max forward progress, min deviation from center
    cost = u.T @ u - 10000 * x_poses.T @ MX.ones(n, 1) + 1000 * y_poses.T @ y_poses
    problem.minimize(cost)

    ## STATE Constraints

    # x limits (enforce forward progress)
    problem.subject_to(x_poses > MX.zeros(n, 1))

    # y limits (lane boundary constraints)
    problem.subject_to(
        problem.bounded(
            -(cfg.lane_width / 2) * MX.ones(n, 1),
            y_poses + cfg.car_width / 2,
            (cfg.lane_width / 2) * MX.ones(n, 1),
        )
    )
    problem.subject_to(
        problem.bounded(
            -(cfg.lane_width / 2) * MX.ones(n, 1),
            y_poses - cfg.car_width / 2,
            (cfg.lane_width / 2) * MX.ones(n, 1),
        )
    )

    # velocity limits
    problem.subject_to(
        problem.bounded(
            -cfg.max_vel * MX.ones(n, 1), velocities, cfg.max_vel * MX.ones(n, 1)
        )
    )

    # theta limits (0, 2pi)
    problem.subject_to(
        problem.bounded(
            cfg.min_theta * MX.ones(n, 1), thetas, cfg.max_theta * MX.ones(n, 1)
        )
    )

    ## CONTROLS Constraints

    # acceleration limits
    problem.subject_to(
        problem.bounded(
            -cfg.max_accel * MX.ones(n, 1), accels, cfg.max_accel * MX.ones(n, 1)
        )
    )

    # ang velocity limits
    problem.subject_to(
        problem.bounded(
            -cfg.max_ang_vel * MX.ones(n, 1), ang_vels, cfg.max_ang_vel * MX.ones(n, 1)
        )
    )

    # Vehicle Dynamics Constraints
    problem.subject_to(traj[:4] == dynamics(x0, u[:2], cfg.interval_dur))

    j = 2
    for i in range(4, traj.shape[0], 4):
        problem.subject_to(
            traj[i : i + 4] == dynamics(traj[i - 4 : i], u[j : j + 2], cfg.interval_dur)
        )
        j += 2

    # Obstacle Avoidance
    for i in range(0, obstacles.shape[0], 3):
        for j in range(0, traj.shape[0], 4):
            problem.subject_to(
                (traj[j] - obstacles[i]) ** 2 + (traj[j + 1] - obstacles[i + 1]) ** 2
                > (obstacles[i + 2] + cfg.car_width) ** 2
            )

    problem.solver("ipopt", casadi_options, solver_options)

    return problem
