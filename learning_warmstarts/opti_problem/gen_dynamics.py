from casadi import *
import math
import torch

def generate_MX_dynamics(version):
    """ Generates the specified dynamics function
        The dynamics function operates on MX input state and control vectors
        and returns an MX output of the next stte

    Arguments:
        version: string specifying either linear or unicycle dynamics
    """

    if version == 'linear':
        """ X = [x, y, vx, vy]
            U = [ax, ay]
        """
        def dynamics(x,u,t):
            nxt_x = MX(4,1)
            nxt_x[0] = x[0] + t*x[2]
            nxt_x[1] = x[1] + t*x[3]
            nxt_x[2] = x[2] + t*u[0]
            nxt_x[3] = x[3] + t*u[1]
            return nxt_x
        return dynamics

    elif version == 'unicycle':
        """ X = [x, y, v, theta]
            U = [a, angular_accel]
        """
        def dynamics(x,u,t):
            nxt_x = MX(4,1)
            nxt_x[0] = x[0] + t*x[2]*cos(x[3])
            nxt_x[1] = x[1] + t*x[2]*sin(x[3])
            nxt_x[2] = x[2] + t*u[0]
            nxt_x[3] = x[3] + t*u[1]
            return nxt_x
        return dynamics

    else:
        raise NotImplementedError(f'{version} dynamics is not implemented yet')    

def generate_list_dynamics(version):
    """ Generates the specified dynamics function
        The dynamics function operates on list input state and control vectors 
        and returns a list output of the next stte

    Arguments:
        version: string specifying either linear or unicycle dynamics
    """

    if version == 'linear':
        """ X = [x, y, vx, vy]
            U = [ax, ay]
        """
        def dynamics(x,u,t):
            nxt_x = [0.0 for i in range(4)]
            nxt_x[0] = x[0] + t*x[2]
            nxt_x[1] = x[1] + t*x[3]
            nxt_x[2] = x[2] + t*u[0]
            nxt_x[3] = x[3] + t*u[1]
            return nxt_x
        return dynamics

    elif version == 'unicycle':
        """ X = [x, y, v, theta]
            U = [a, angular_accel]
        """
        def dynamics(x,u,t):
            nxt_x = [0.0 for i in range(4)]
            nxt_x[0] = x[0] + t*x[2]*math.cos(x[3])
            nxt_x[1] = x[1] + t*x[2]*math.sin(x[3])
            nxt_x[2] = x[2] + t*u[0]
            nxt_x[3] = x[3] + t*u[1]
            return nxt_x
        return dynamics
        
    else:
        raise NotImplementedError(f'{version} dynamics is not implemented yet')    
    
def generate_tensor_dynamics(version):
    """ Generates the specified dynamics function
        The dynamics function operates on tensor input state and control vectors 
        and returns a tensor output of the next stte

    Arguments:
        version: string specifying either linear or unicycle dynamics
    """

    if version == 'linear':
        """ X = [x, y, vx, vy]
            U = [ax, ay]
        """
        def dynamics(x,u,t):
            nxt_x = torch.empty(4)
            nxt_x[0] = x[0] + t*x[2]
            nxt_x[1] = x[1] + t*x[3]
            nxt_x[2] = x[2] + t*u[0]
            nxt_x[3] = x[3] + t*u[1]
            return nxt_x
        return dynamics

    elif version == 'unicycle':
        """ X = [x, y, v, theta]
            U = [a, angular_accel]
        """
        def dynamics(x,u,t):
            nxt_x = torch.empty(4)
            nxt_x[0] = x[0] + t*x[2]*math.cos(x[3])
            nxt_x[1] = x[1] + t*x[2]*math.sin(x[3])
            nxt_x[2] = x[2] + t*u[0]
            nxt_x[3] = x[3] + t*u[1]
            return nxt_x
        return dynamics
        
    else:
        raise NotImplementedError(f'{version} dynamics is not implemented yet')    

def generate_np_dynamics(version):
    """ Generates the specified dynamics function
        The dynamics function operates on tensor input state and control vectors 
        and returns a tensor output of the next stte

    Arguments:
        version: string specifying either linear or unicycle dynamics
    """

    if version == 'linear':
        """ X = [x, y, vx, vy]
            U = [ax, ay]
        """
        def dynamics(x,u,t):
            nxt_x = np.zeros(4)
            nxt_x[0] = x[0] + t*x[2]
            nxt_x[1] = x[1] + t*x[3]
            nxt_x[2] = x[2] + t*u[0]
            nxt_x[3] = x[3] + t*u[1]
            return nxt_x
        return dynamics

    elif version == 'unicycle':
        """ X = [x, y, v, theta]
            U = [a, angular_accel]
        """
        def dynamics(x,u,t):
            nxt_x = np.zeros(4)
            nxt_x[0] = x[0] + t*x[2]*np.cos(x[3])
            nxt_x[1] = x[1] + t*x[2]*np.sin(x[3])
            nxt_x[2] = x[2] + t*u[0]
            nxt_x[3] = x[3] + t*u[1]
            return nxt_x
        return dynamics
        
    else:
        raise NotImplementedError(f'{version} dynamics is not implemented yet')    