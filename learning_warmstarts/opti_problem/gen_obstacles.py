import numpy as np
from math import sqrt
from random import random

def get_dist(c1,c2):
    """ Calc distance between two circlular obstacles c1 = [x1, y1, r1] c2 = [x2, y2, r2] 
        sqrt((x2 - x1)**2 + (y2 - y1)**2) - (r2 + r1)
    """
    return sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2) - (c1[2] + c2[2])

def generate_obstacles(n_obst, lane_width, car_horizon, car_width, x0, prev_obst = None):
    obstacles = [0.0 for i in range(3*n_obst)]
    obst_idx = 0
    
    if prev_obst:
        for i in range(0, len(prev_obst), 3):
            if x0[0] <= prev_obst[i] + prev_obst[i+2]:
                obstacles[obst_idx:obst_idx+3] = prev_obst[i:i + 3] 
                obst_idx += 3
   
    DIST_THRESHOLD = car_width + 2.0 # 1 meter clearance on either side of the car
      
    getRand = lambda a,b: (b-a) * random() + a
    # distance between two circlular obstacles = sqrt((x2 − x1)**2 + (y2 − y1)**2) − (r2 + r1)

    while obst_idx < 3 * n_obst:
        x = getRand(x0[0] + 15, x0[0] + car_horizon)
        y = getRand(-lane_width/2, lane_width/2)
        r = getRand(1, lane_width / 3)
        new_obstacle = [x , y, r]
        collision = False
        
        # check for path btw it and other obstacles
        j = 0
        while not collision and j < obst_idx:
            collision = get_dist(obstacles[j : j + 3], new_obstacle) < DIST_THRESHOLD
            j = j + 3

        if not collision:
            obstacles[obst_idx : obst_idx + 3] = new_obstacle
            obst_idx += 3
    return obstacles

def generate_np_obstacles(n_obst, lane_width, car_horizon, car_width, x0, prev_obst = None):
    obstacles = np.zeros(3*n_obst, dtype=np.float32)
    obst_idx = 0
    
    if prev_obst is not None:
        for i in range(0, len(prev_obst), 3):
            if x0[0] <= prev_obst[i] + prev_obst[i+2]:
                obstacles[obst_idx:obst_idx+3] = prev_obst[i:i + 3] 
                obst_idx += 3
   
    DIST_THRESHOLD = car_width + 2.0 # 1 meter clearance on either side of the car
      
    getRand = lambda a,b: (b-a) * random() + a

    while obst_idx < 3 * n_obst:
        x = getRand(x0[0] + 15, x0[0] + car_horizon)
        y = getRand(-lane_width/2, lane_width/2)
        r = getRand(1, lane_width / 3)
        new_obstacle = np.array([x , y, r], dtype=np.float32)
        collision = False
        
        # check for path btw it and other obstacles
        j = 0
        while not collision and j < obst_idx:
            collision = get_dist(obstacles[j : j + 3], new_obstacle) < DIST_THRESHOLD
            j = j + 3

        if not collision:
            obstacles[obst_idx : obst_idx + 3] = new_obstacle
            obst_idx += 3

    return obstacles