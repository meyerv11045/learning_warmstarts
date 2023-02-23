import statistics

import numpy as np
import torch
import wandb

from learning_warmstarts import problem_specs
from learning_warmstarts.dataset.data_generator import get_rand_x0
from learning_warmstarts.neural_nets.loss_fns import IpoptLoss
from learning_warmstarts.neural_nets.models import FFNet
from learning_warmstarts.opti_problem.gen_obstacles import generate_obstacles


class OnlineTrainer:
    def __init__(
        self, layers, learning_rate, convergence_threshold, act_fn, debug_mode, n
    ):
        self.debug_mode = debug_mode
        self.learning_rate = float(learning_rate)
        self.convergence_threshold = float(convergence_threshold)
        self.shape = layers
        self.act_fn = act_fn

        self.train_setup()

        self.model_file = "model.pt"

        config = dict(
            epochs=n,
            lr=self.learning_rate,
            shape=self.shape,
            activation=act_fn,
            cost_fn="ipopt",
        )

        self.n = n

        if not self.debug_mode:
            self.run = wandb.init(
                project="learning_warmstarts",
                entity="laine-lab",
                config=config,
                job_type="online-train",
            )

    def train_setup(self):
        if self.act_fn == "relu":
            self.model = FFNet(self.shape, torch.nn.functional.relu)
        elif self.act_fn == "tanh":
            self.model = FFNet(self.shape, torch.nn.functional.tanh)
        elif self.act_fn == "leaky_relu":
            self.model = FFNet(self.shape, torch.nn.functional.leaky_relu)
        else:
            raise ValueError("Specified activation function not supported")

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

        self.criterion = IpoptLoss(max_iters=30)

    def run_train_loop(self):
        self.train()
        self.save_model()

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_file)
        print("Saved Model")

        if not self.debug_mode:
            artifact = wandb.Artifact("trained_model", type="model")
            artifact.add_file(self.model_file)

            self.run.log_artifact(artifact)

    def train(self):
        print("Starting Training...")
        self.model.train()
        i = 0
        while i < self.n:
            x0 = get_rand_x0(problem_specs["max_vel"])
            obsts = np.array(
                generate_obstacles(
                    problem_specs["num_obstacles"],
                    problem_specs["lane_width"],
                    problem_specs["car_horizon"],
                    problem_specs["car_width"],
                    x0,
                )
            )
            x = np.concatenate([x0, obsts])
            x = torch.from_numpy(x).float()

            prediction = self.model(x)
            loss = self.criterion(prediction, x)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            i += 1
            if not self.debug_mode:
                wandb.log({"train loss": loss, "test loss": 0})

            print(f"Problem {i:<10} / {self.n}: \t Train Loss: {loss:.4e}")

            # if(i > 9 and statistics.stdev(train_losses) < self.convergence_threshold * train_losses[(i-1) % 10]):
            #     print('Converged')
            #     break
