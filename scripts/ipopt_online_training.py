import argparse
from pathlib import Path
from learning_warmstarts.models import FCNet
from learning_warmstarts.train import train_ipopt_loss, save_results
from learning_warmstarts import cfg
from learning_warmstarts.utils.logs import setup_logging

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--save_folder', required=True)
parser.add_argument('-l', '--layers', default="9,256,512,1024,240")
parser.add_argument('-i','--max_iters', type=int, default=25)
parser.add_argument('-lr','--learning_rate', type=float, default=3e-4)
parser.add_argument('-ct', '--convergence_threshold', type=float, default=1e-3)
parser.add_argument('-m', '--max_steps', type=int, default=50_000)
parser.add_argument('-ll', '--log_level', default='INFO', help='DEBUG, INFO, WARNING, ERROR')
args = parser.parse_args()
args.layers = [int(layer_size.strip()) for layer_size in args.layers.split(',')]

save_folder = Path(f'results/{args.save_folder}')
if not save_folder.exists():
    save_folder.mkdir()

setup_logging(args.log_level, True, save_folder/"ipopt_training.log")

model = FCNet(args.layers)

trained_model, losses = train_ipopt_loss(model, args.max_iters, args.convergence_threshold, args.learning_rate, args.max_steps, cfg)

save_results(trained_model, losses, save_folder, args)
