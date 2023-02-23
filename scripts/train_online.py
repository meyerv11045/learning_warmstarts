

import argparse
from learning_warmstarts.neural_nets.online_train import OnlineTrainer
import signal

def handler(signum,frame):
    quit = input("Control C was pressed. Do you want to quit? y/n (lowercase)")
    if quit == 'y':
        model_trainer.save_model()
        exit(1)

if __name__ == '__main__':
    """ Supported cost functions: mse, custom, lag, slag, or ipopt, wmse
        Supported activation fns: relu, tanh, leaky_relu
    """

    signal.signal(signal.SIGINT, handler)
    global model_trainer
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--n', type=int, default=224_000)
    parser.add_argument('-lr','--learning_rate', type=float, default=1e-4)
    parser.add_argument('-l', '--layers', required=True)
    parser.add_argument('-ct', '--convergence_threshold', type=float, default=0.001)
    parser.add_argument('-a', '--act_fn', default='relu')
    parser.add_argument('-t', '--debug_mode', action='store_true', default=False)
    args = parser.parse_args()
    
    args.layers = [int(layer_size) for layer_size in args.layers.split(',')]
    model_trainer = OnlineTrainer(**vars(args))
    model_trainer.run_train_loop()