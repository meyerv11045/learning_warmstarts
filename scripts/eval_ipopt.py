import torch
import statistics
import argparse
import logging
from pathlib import Path
from learning_warmstarts.dataset.obstacles import generate_obstacles
from learning_warmstarts.dataset.opt import setup_problem
from learning_warmstarts import cfg, FCNet, setup_logging, write_obj

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--save_folder', required=True)
parser.add_argument('-n','--n', type=int, default=200)
parser.add_argument('-ll', '--log_level', default='INFO', help='DEBUG, INFO, WARNING, ERROR')
args = parser.parse_args()

save_folder = Path(f'results/{args.save_folder}')
if not save_folder.exists():
    save_folder.mkdir()

setup_logging(args.log_level, True, save_folder/"eval.log")

problem = setup_problem(cfg)

error_during_solve = 0
all_iters = []
all_t_proc = []
all_t_wall = []

with torch.no_grad():
    for i in range(args.n):
        obsts = generate_obstacles(cfg)
        problem.set_value(problem.p[4:4+3*cfg.n_obstacles], obsts)

        try:
            solution = problem.solve()
            iters = solution.stats()["iter_count"]
            t_proc = solution.stats()["t_proc_total"]
            t_wall = solution.stats()["t_wall_total"]
            logging.info('%f iters in %f s', iters, t_wall)
            all_iters.append(iters)
            all_t_proc.append(t_proc)
            all_t_wall.append(t_wall)

        except RuntimeError as e:
            logging.warning('error %s', e)
            logging.warning('obstacles: %s', str(list(obsts)))
            error_during_solve += 1

def analyze(lst, caption=''):
    avg = statistics.mean(lst)
    std = statistics.stdev(lst)
    logging.info('%s avg: %f std: %f', caption, avg, std)
    return avg, std

logging.info('%d randomly generated problems', len(all_iters))
logging.info('%d warmstarts resulted in infeasible problems', error_during_solve)
analyze(all_iters, 'iters')
analyze(all_t_proc, 't_proc')
analyze(all_t_wall, 't_wall')

write_obj({'iters': all_iters, 't_procs': all_t_proc, 't_walls': all_t_wall, 'infeasible': error_during_solve}, save_folder/"results.pkl")
