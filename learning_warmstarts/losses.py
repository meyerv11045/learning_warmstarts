import logging
import torch
import torch.nn as nn

from learning_warmstarts.dataset.opt import setup_problem


class IpoptLoss(nn.Module):
    def __init__(self, max_iters, cfg):
        super(IpoptLoss, self).__init__()

        self.problem = setup_problem(cfg)
        options = {
            "print_time": False,
            "ipopt": {"max_iter": max_iters, "print_level": 0},
        }
        self.n_obsts = cfg.n_obstacles
        self.problem.solver("ipopt", options)

        self.mse = nn.MSELoss(reduction="mean")
        logging.info("ipopt loss initialized succesfully")

    def forward(self, predictions, inputs):
        predictions_np = predictions.detach().numpy()
        inputs = inputs.numpy()

        self.problem.set_value(self.problem.p[4 : 4 + 3 * self.n_obsts], inputs)
        self.problem.set_initial(self.problem.x, predictions_np)

        solution = self.problem.solve_limited()
        target = solution.value(self.problem.x)

        target = torch.from_numpy(target)
        target.requires_grad = False

        return self.mse(predictions, target.float())


# TODO
# def tensor_dynamics(x, u, t):
#     nxt_x = torch.zeros_like(x)
#     nxt_x[0] = x[0] + t * x[2] * torch.cos(x[3])
#     nxt_x[1] = x[1] + t * x[2] * torch.sin(x[3])
#     nxt_x[2] = x[2] + t * u[0]
#     nxt_x[3] = x[3] + t * u[1]
#     return nxt_x
# class SoftLoss(nn.Module):
#     """ Incoporate dynamics and collision avoidance constraints
#         into the objective function as a regularization term
#     """
#     def __init__(self, problem_specs):
#         super(SoftLoss, self).__init__()
#         self.problem_specs = problem_specs

#         self.n = problem_specs["control_intervals"]
#         self.t = problem_specs["interval_duration"]
#         self.mse = nn.MSELoss(reduction="mean")
#         self.dynamics = generate_tensor_dynamics(problem_specs["dynamics"])

#     def forward(self, predictions, targets, inputs):
#         mse_error = self.mse(predictions, targets)  # avg for the entire batch

#         predictions = predictions.view(-1)
#         inputs = inputs.view(-1)
#         m = 0
#         constraint_error = 0

#         # penalize dynamics constraint violations in the prediction/input
#         j = 0
#         for i in range(0, len(predictions), 240):
#             constraint_error += self.calc_constraint_violation(
#                 predictions[i : i + 240], inputs[j : j + 13]
#             )
#             j += 13
#             m += 1

#         constraint_error = constraint_error / m

#         res = mse_error + constraint_error
#         return res

#     def calc_constraint_violation(self, prediction, inputs):
#         """Calculates the constraint violation for a single input"""
#         x = prediction[: 4 * self.n]
#         u = prediction[4 * self.n : 6 * self.n]

#         x0 = inputs[:4]
#         obstacles = inputs[4:]
#         dynamics_error = x[:4] - self.dynamics(x0, u[:2], self.t)

#         j = 2
#         for i in range(4, len(x), 4):
#             dynamics_error += x[i : i + 4] - self.dynamics(
#                 x[i - 4 : i], u[j : j + 2], self.t
#             )
#             j += 2

#         dynamics_error = torch.norm(dynamics_error)

#         # penalize being inside obstacles (outside is no error)
#         obst_error = 0
#         for i in range(0, len(obstacles), 3):
#             for j in range(0, len(x), 4):
#                 obst_error += (
#                     math.sqrt(
#                         (x[j] - obstacles[i]) ** 2 + (x[j + 1] - obstacles[i + 1]) ** 2
#                     )
#                     - (obstacles[i + 2] + self.problem_specs["car_width"]) ** 2
#                 )

#         return dynamics_error + obst_error
