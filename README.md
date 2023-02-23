# learning_warmstarts
Remember to set any parameters for the problem or training in the `__init__.py` of the learning_warmstarts package

## training pipeline
To generate training/benchmarking data: `python3 gen_data.py`
- `-n` number of datapoints to generate in each process
- `-p` number of processes to run (defaults to 1)
- `s` seed for random num generator (repeat as many times as you have processes)
- `-o` file path for output results as csv
- `-b` optional flag that is included to generate benchmarking data. Defaults to false 
- `-i` optional flag that is included to save images of trajectories as they are generated. Defaults to false 

To train the neural network: `python3 train_model.py`
- `-d` file path to dataset input csv file 
- `-D` optional flag that is included to train a model to also predict the dual variables. Defaults to false
- Will save model at end of training into `models/` as `model_<batch size>_<epochs>_<learning rate>_<shape>.pt`

To benchmark the models against the solver: `python3 run_benchmark.py`
- `-p` path to model that predicts only primary variables
- `-pd` path to model that predicts primary + dual variables (optional and deafults to None)
- `-d` path to benchmarking dataset csv file
- `-o` path to an output csv file containing the benchmarking's results

## table of contents
- `learning_warmstarts`: python package for all the components of the learning warmstarts for the obstacle avoidance optimization problem
    - `dataset/`: anyting related to generating and loading datasets of the obstacle avoidance problem
    - `opti_problem/`: anythig related to the casadi optimization problem (e.g. generating the problem, obstacles, dynamics, etc. )
    - `neural_nets/`: contains the PyTorch Feed forward neural network module and the training pipeline
- `matlab_mpc/`: original matlab implementation of obstacle avoidance simulation
    - `avoid_obstacle.m`: run single obstacle avoidance optimization problem
    - `mpc_loop.m`: run full mpc loop for specified number of iterations
    - All the other files contain helper functions used by the two above files (e.g. plotting, creating gifs, creating the optimization problem, creating obstacles)