import wandb

from learning_warmstarts.testing.benchmark import Test


class OverfitPredictions(Test):
    def __init__(
        self,
        hidden_layers,
        model_path,
        train_folder,
        test_folder,
        use_warmstart_params,
        n_samples,
        save_trajectories,
        dynamic_smoothing,
        activation_fn,
        version,
    ):
        super().__init__(hidden_layers, predict_duals=False)

        self.use_warmstart_params = use_warmstart_params
        self.n_samples = n_samples
        self.save_trajectories = save_trajectories
        self.dynamic_smoothing = dynamic_smoothing

        config = dict(warmstart_params=use_warmstart_params, n_samples=n_samples)
        if self.dynamic_smoothing:
            self.wandb_run = wandb.init(
                project="learning_warmstarts",
                entity="laine-lab",
                config=config,
                job_type="dynamic_smoothing_predictions",
            )
        else:
            self.wandb_run = wandb.init(
                project="learning_warmstarts",
                entity="laine-lab",
                config=config,
                job_type="overfit_predictions",
            )

        model_at = self.wandb_run.use_artifact(version, type="model")
        model_dir = model_at.download()

        self.model = OverfitPredictions.load_model(
            model_dir + "/" + model_path, self.shape, activation_fn
        )

        self.train_folder = train_folder
        self.test_folder = test_folder

    def run(self):
        train_iter, test_iter = self.load_data(
            dataset_type="obstacle_avoidance",
            train_folder=self.train_folder,
            test_folder=self.test_folder,
        )
        cols = ["Predictions", "Ground Truth", "Iters", "Time", "x0", "obstacles"]

        # Model's warmstart performance on seen data
        self.train_table = wandb.Table(columns=cols)
        avg_iters, avg_time = self.loop_dataset(train_iter, self.train_table)
        wandb.log(
            {"train set avg iterations": avg_iters, "train set avg time": avg_time}
        )

        # Model's warmstart performance on unseen data
        self.test_table = wandb.Table(columns=cols)
        avg_iters, avg_time = self.loop_dataset(test_iter, self.test_table, train=False)
        wandb.log({"test set avg iterations": avg_iters, "test set avg time": avg_time})

        # Model's predictions visualized
        wandb.log({"train_predictions": self.train_table})
        wandb.log({"test_predictions": self.test_table})

    def loop_dataset(self, dataiter, table, train=True):
        running_iters = 0
        running_time = 0
        n_unsolvable = 0
        m = 0
        save_img_freq = 5

        self.set_solver_params(self.use_warmstart_params)
        for x, y in dataiter:
            if self.n_samples and m > self.n_samples:
                break

            try:
                nn_warmstart = self.model(x.float()).tolist()

                x_list = x.tolist()

                x0 = x_list[:4]
                obstacles = x_list[4:]

                self.problem.set_value(self.x0_param[[0, 1, 2, 3]], x0)
                self.problem.set_value(
                    self.obstacle_param[[i for i in range(3 * self.N_OBST)]], obstacles
                )

                u_warmstart = nn_warmstart[self.N * 4 :]

                if self.dynamic_smoothing:
                    x_warmstart = []
                    cur_x = x0
                    for i in range(0, len(u_warmstart), 2):
                        cur_x = self.list_dynamics(
                            cur_x, u_warmstart[i : i + 2], self.T
                        )
                        x_warmstart.extend(cur_x)
                else:
                    x_warmstart = nn_warmstart[: self.N * 4]

                self.problem.set_initial(self.x_var, x_warmstart)
                self.problem.set_initial(self.u_var, u_warmstart)

                (iters, time), (solved_x, u, lam_g) = self.solve()

                if self.save_trajectories and m % save_img_freq == 0:
                    row = []
                    if train:
                        filename = f"i_train{m / save_img_freq}.png"
                    else:
                        filename = f"i_test{m / save_img_freq}.png"
                    self.viz_warmstart_vs_solution(
                        x0, obstacles, x_warmstart, solved_x, filename, iters, time
                    )

                    y = y.tolist()
                    gt_x = y[: 4 * self.N]

                    if train:
                        gt_filename = f"gt_train{m/1}.png"
                    else:
                        gt_filename = f"gt_test{m/1}.png"

                    self.viz.save_trajectory(gt_x, x0, obstacles, gt_filename)

                    row.append(wandb.Image(filename))
                    row.append(wandb.Image(gt_filename))
                    row.append(iters)
                    row.append(time)
                    row.append(str(x0))
                    row.append(str(obstacles))
                    table.add_data(*row)

                running_iters += iters
                running_time += time

                m += 1

                print(f"Solved problem {m}")

            except RuntimeError as e:
                n_unsolvable += 1
                print(f"Unsolvable Problem", e)

        avg_iters = running_iters / m
        avg_time = running_time / m

        print(f"Avg Iters: {avg_iters:4f} \t Avg Time: {avg_time:4f}")
        return avg_iters, avg_time
