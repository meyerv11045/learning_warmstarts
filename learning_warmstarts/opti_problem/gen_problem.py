from casadi import *

from learning_warmstarts import problem_specs
from learning_warmstarts.opti_problem.gen_dynamics import generate_MX_dynamics


def generate_problem(N, t, lane_width, num_obstacles, car_width, dynamics):
    opti = Opti()
    X = opti.variable(4 * N)
    U = opti.variable(2 * N)
    X0 = opti.parameter(4)
    obstacles = opti.parameter(3 * num_obstacles)
    T = opti.parameter()
    opti.set_value(T, t)

    R = MX.eye(2 * N)
    W = MX.zeros(4 * N, 1)
    for i in range(0, 4 * N, 4):
        W[i] = 1

    select_y = MX.zeros(N, 4 * N)

    j = 1
    for i in range(N):
        select_y[i, j] = 1
        j += 4

    y_poses = select_y @ X

    # min accceleration, max forward progress, min deviation from center
    cost = U.T @ R @ U - 10000 * X.T @ W + 1000 * y_poses.T @ y_poses
    opti.minimize(cost)

    # acceleration limits
    opti.subject_to(U <= 500 * MX.ones(2 * N, 1))

    # velocity limits
    select_v = MX(N, 4 * N)
    j = 2
    for i in range(N):
        select_v[i, j] = 1
        j += 4

    velocities = select_v @ X
    opti.subject_to(velocities <= 200 * MX.ones(N, 1))

    # Vehicle Dynamics Constraints
    opti.subject_to(X[:4] == dynamics(X0, U[:2], T))

    j = 2
    for i in range(4, X.shape[0], 4):
        opti.subject_to(X[i : i + 4] == dynamics(X[i - 4 : i], U[j : j + 2], T))
        j += 2

    # Obstacle Avoidance
    for i in range(0, obstacles.shape[0], 3):
        for j in range(0, X.shape[0], 4):
            opti.subject_to(
                (X[j] - obstacles[i]) ** 2 + (X[j + 1] - obstacles[i + 1]) ** 2
                > (obstacles[i + 2] + car_width) ** 2
            )

    # Lane Boundary Constraints
    opti.subject_to(
        opti.bounded(
            -(lane_width / 2) * MX.ones(N, 1),
            y_poses + car_width / 2,
            (lane_width / 2) * MX.ones(N, 1),
        )
    )
    opti.subject_to(
        opti.bounded(
            -(lane_width / 2) * MX.ones(N, 1),
            y_poses - car_width / 2,
            (lane_width / 2) * MX.ones(N, 1),
        )
    )

    return (opti, X, U, X0, obstacles, T)


def generate_problem_v2():
    """Cleaner interface
    No need to pass any params to create the problem, the problem_spec is used
    Only returns the problem instance instead of refernces to all the variables/params
    """
    problem = Opti()

    n = problem_specs["control_intervals"]
    t = problem_specs["interval_duration"]
    lane_width = problem_specs["lane_width"]
    car_width = problem_specs["car_width"]

    X = problem.variable(4 * n)
    U = problem.variable(2 * n)
    X0 = problem.parameter(4)
    obstacles = problem.parameter(3 * problem_specs["num_obstacles"])
    T = problem.parameter()
    problem.set_value(T, t)

    R = MX.eye(2 * n)
    W = MX.zeros(4 * n, 1)
    for i in range(0, 4 * n, 4):
        W[i] = 1

    select_y = MX.zeros(n, 4 * n)

    j = 1
    for i in range(n):
        select_y[i, j] = 1
        j += 4

    y_poses = select_y @ X

    # min accceleration, max forward progress, min deviation from center
    cost = U.T @ R @ U - 10000 * X.T @ W + 1000 * y_poses.T @ y_poses
    problem.minimize(cost)

    # acceleration limits
    problem.subject_to(
        problem.bounded(-500 * MX.ones(2 * n, 1), U, 500 * MX.ones(2 * n, 1))
    )

    # velocity limits
    select_v = MX(n, 4 * n)
    j = 2
    for i in range(n):
        select_v[i, j] = 1
        j += 4

    velocities = select_v @ X
    problem.subject_to(
        problem.bounded(-200 * MX.ones(n, 1), velocities, 200 * MX.ones(n, 1))
    )

    # Vehicle Dynamics Constraints
    dynamics = generate_MX_dynamics(problem_specs["dynamics"])
    problem.subject_to(X[:4] == dynamics(X0, U[:2], T))

    j = 2
    for i in range(4, X.shape[0], 4):
        problem.subject_to(X[i : i + 4] == dynamics(X[i - 4 : i], U[j : j + 2], T))
        j += 2

    # Obstacle Avoidance
    for i in range(0, obstacles.shape[0], 3):
        for j in range(0, X.shape[0], 4):
            problem.subject_to(
                (X[j] - obstacles[i]) ** 2 + (X[j + 1] - obstacles[i + 1]) ** 2
                > (obstacles[i + 2] + car_width) ** 2
            )

    # Lane Boundary Constraints
    problem.subject_to(
        problem.bounded(
            -(lane_width / 2) * MX.ones(n, 1),
            y_poses + car_width / 2,
            (lane_width / 2) * MX.ones(n, 1),
        )
    )
    problem.subject_to(
        problem.bounded(
            -(lane_width / 2) * MX.ones(n, 1),
            y_poses - car_width / 2,
            (lane_width / 2) * MX.ones(n, 1),
        )
    )

    return problem
