import torch
import argparse
from pathlib import Path

from learning_warmstarts import cfg, FCNet, train_baseline, save_results, setup_logging, StateControlDataset
from learning_warmstarts.utils.repro import set_seed

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--save_folder', required=True)
parser.add_argument('-td', '--train_data', required=True)
parser.add_argument('-vd', '--val_data', required=True)
parser.add_argument('-l', '--layers', default="9,256,512,1024,240")
parser.add_argument('-e','--epoch', type=int, default=1000)
parser.add_argument('-b','--batchsize', type=int, default=64)
parser.add_argument('-lr','--learning_rate', type=float, default=3e-4)
parser.add_argument('-ct', '--convergence_threshold', type=float, default=1e-3)
parser.add_argument('-m', '--max_steps', type=int, default=50_000)
parser.add_argument('--seed',type=int, default=42)
parser.add_argument('-ll', '--log_level', default='INFO', help='DEBUG, INFO, WARNING, ERROR')
args = parser.parse_args()
args.layers = [int(layer_size.strip()) for layer_size in args.layers.split(',')]

save_folder = Path(f'results/{args.save_folder}')
if not save_folder.exists():
    save_folder.mkdir()

setup_logging(args.log_level, True, save_folder/"baseline_training.log")

set_seed(args.seed)

model = FCNet(args.layers)

train_data = StateControlDataset(cfg, args.train_data)
val_data = StateControlDataset(cfg, args.val_data)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batchsize, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batchsize, shuffle=True)

trained_model, losses = train_baseline(model, train_loader, val_loader, args)

save_results(trained_model, losses, save_folder, args)
