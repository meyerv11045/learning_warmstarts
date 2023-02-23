import torch.nn as nn
import torch
import math
import numpy as np
import casadi
from learning_warmstarts.opti_problem.gen_problem import generate_problem_v2    
from learning_warmstarts.opti_problem.gen_dynamics import generate_tensor_dynamics
from learning_warmstarts.opti_problem.gen_problem import generate_problem_v2

class WeightedMSE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, predictions, target):
        primals = self.mse(predictions[:140], target[:140])
        duals = self.mse(predictions[140:], target[140:])

        return 1000 * primals + duals
class IpoptLoss(nn.Module):
    """ Loss function that uses IPOPT online to generate partially solved trajectory targets
        based on the network's prediction
    """
    
    def __init__(self, max_iters):
        """
            max_iters: a value of 25 worked well in achieving results that beat MSE with full trajectory target
        """

        super(IpoptLoss, self).__init__()
    
        self.problem = generate_problem_v2()
        options = {"print_time": False, "ipopt": {"max_iter": max_iters, 'print_level': 0}}

        self.problem.solver('ipopt', options)
        
        self.mse = nn.MSELoss(reduction="mean")

    def forward(self, predictions, inputs):        
        predictions_np = predictions.detach().numpy()
        inputs = inputs.numpy()

        self.problem.set_value(self.problem.p[:-1], inputs)
        self.problem.set_initial(self.problem.x, predictions_np)
        
        solution = self.problem.solve_limited()
        target = solution.value(self.problem.x)
        
        target = torch.from_numpy(target)
        target.requires_grad = False
        
        return self.mse(predictions, target.float())


class LagrangianFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, z, lamg, params, lag_fn, grad_fn):
        """ only returns 1 output- a scalar representing the loss
            pass the casadi loss fn and its gradient into the forward pass
            so they are not created for each evaluation of the function
        """
        ctx.z = z.numpy()
        ctx.lamg = lamg.numpy()
        ctx.p = params.numpy()
        
        ctx.lag_fn = lag_fn
        ctx.grad_fn = grad_fn

        return torch.from_numpy(np.array(lag_fn(ctx.z, ctx.p, ctx.lamg)))

    @staticmethod
    def backward(ctx, grad_output):
        """ It must accept a context ctx as the first argument, followed by as many outputs as the forward() returned 
        (None will be passed in for non tensor outputs of the forward function), and it should return as many tensors, 
        as there were inputs to forward(). Each argument is the gradient w.r.t the given output, and each returned value should be the 
        gradient w.r.t. the corresponding input. If an input is not a Tensor or is a Tensor not requiring grads, you can just pass None as a 
        gradient for that input.

        grad_output will always be None because no operations will be performed after this loss fn
        """
            
        jacobian_wrt_predictions = torch.from_numpy(np.array(ctx.grad_fn(ctx.z, ctx.p, ctx.lamg,0)[:240]))

        return jacobian_wrt_predictions, None, None, None, None

class SimpleLagrangianLoss(nn.Module):
    def __init__(self):
        super(SimpleLagrangianLoss, self).__init__()

        self.problem = generate_problem_v2()
        self.lagrangian_fn = self.gen_lagrangian()
        self.lagrangian_jacobian = self.lagrangian_fn.jacobian()

    def gen_lagrangian(self):
        x = self.problem.x
        p = self.problem.p
        lamg = self.problem.lam_g
        lbg = self.problem.lbg

        g = casadi.Function('g', [x, p], [self.problem.g])
        f = casadi.Function('f', [x], [self.problem.f])
        b = casadi.Function('b', [p], [lbg])
        l = casadi.Function('L', [x,p,lamg], [f(x) - casadi.dot(lamg, (g(x,p) - b(p)))])

        return l

    def forward(self, predictions, targets, params):
        """ predictions- x_var, u_var 
            targets- x_var, u_var, lam_g
            params- x0, obstacles, t
        """
        lamg = targets[240:]
        return LagrangianFn.apply(predictions, lamg, params, self.lagrangian_fn,  self.lagrangian_jacobian)

class LagrangianLoss(nn.Module):
    """ Loss function from Chen, et al.
        https://arxiv.org/pdf/1910.10835.pdf
        Implemented on the obstacle avoidance optimization problem
        Note: not implemented for batched inputs, only single inputs
    """
    
    def __init__(self) -> None:
        super(LagrangianLoss, self).__init__()
        
        self.problem = generate_problem_v2()
        self.lagrangian_fn = self.gen_lagrangian()
        self.lagrangian_jacobian = self.lagrangian_fn.jacobian()

    def gen_lagrangian(self):
        x = self.problem.x
        p = self.problem.p
        lamg = self.problem.lam_g
        lbg = self.problem.lbg

        g = casadi.Function('g', [x, p], [self.problem.g])
        f = casadi.Function('f', [x], [self.problem.f])
        b = casadi.Function('b', [p], [lbg])
        l = casadi.Function('L', [x,p,lamg], [f(x) - casadi.dot(lamg, (g(x,p) - b(p)))])

        return l

    def forward(self, predictions, targets, params):
        """ predictions- x_var, u_var 
            targets- x_var, u_var, lam_g
            params- x0, obstacles, t
        """
        lamg = targets[240:]
        diff = LagrangianFn.apply(predictions, lamg, params, self.lagrangian_fn,  self.lagrangian_jacobian) - LagrangianFn.apply(targets[:240], lamg, params,  self.lagrangian_fn,  self.lagrangian_jacobian)
        return diff ** 2

class ConstraintLoss(nn.Module):
    """ Custom loss function to more explicitly teach the NN
        to avoid obstacles and satisfy vehicle dynamics
        Note: Uses for loops to handle batches (probably slow for batches <100)
        https://discuss.pytorch.org/t/custom-loss-function-for-a-batch/128248/2
    """
    def __init__(self, problem_specs):
        super(ConstraintLoss, self).__init__()
        self.problem_specs = problem_specs

        self.n = problem_specs['control_intervals']
        self.t = problem_specs['interval_duration']
        self.mse = nn.MSELoss(reduction="mean")
        self.dynamics = generate_tensor_dynamics(problem_specs['dynamics'])

    def forward(self, predictions, targets, inputs):        
        # use to get rid of the outliers and get general trajectory 
        # also implicitly learns the rough acceleration and velocity limits 
        
        mse_error = self.mse(predictions,targets) # avg for the entire batch 

        predictions = predictions.view(-1)
        inputs = inputs.view(-1)
        m = 0
        constraint_error = 0

        # penalize dynamics constraint violations in the prediction/input
        j = 0
        for i in range(0, len(predictions), 240):
            constraint_error += self.calc_constraint_violation(predictions[i:i+240],inputs[j:j+13])
            j += 13
            m += 1
        
        constraint_error = constraint_error / m

        # probably want to weight them equally initially
        # controlling the weighting of these will control what is learned by each iteration
        # even as time goes we can change them to teach the model different things
        
        res = mse_error + constraint_error
        return res

    def calc_constraint_violation(self, prediction, inputs):
        """ Calculates the constraint violation for a single input
        """
        x = prediction[:4*self.n]
        u = prediction[4*self.n:6*self.n]

        x0 = inputs[:4]
        obstacles = inputs[4:]
        dynamics_error = x[:4] - self.dynamics(x0,u[:2],self.t)

        j = 2
        for i in range(4, len(x), 4):
            dynamics_error += x[i:i+4] - self.dynamics(x[i-4:i],u[j:j+2],self.t)
            j += 2

        dynamics_error = torch.norm(dynamics_error)

        # penalize being inside obstacles (outside is no error)
        obst_error = 0
        for i in range(0, len(obstacles), 3):
            for j in range(0, len(x), 4):
                obst_error += math.sqrt((x[j] - obstacles[i])**2 + (x[j+1] - obstacles[i+1])**2) - (obstacles[i+2] + self.problem_specs['car_width'])**2
        
        return dynamics_error + obst_error