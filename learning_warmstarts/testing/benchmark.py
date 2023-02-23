import torch
import wandb
from casadi import *

from learning_warmstarts import problem_specs
from learning_warmstarts.testing.test import Test


class Benchmark(Test):
    def __init__(
        self,
        hidden_layers,
        model_path,
        benchmark_folder,
        predict_duals,
        num_samples,
        activation_fn,
        version,
        ws_params,
    ):
        super().__init__(hidden_layers, predict_duals)

        self.num_samples = num_samples
        self.ws_params = ws_params

        config = dict(num_samples=num_samples)

        inputs = 4 + 3 * problem_specs["num_obstacles"]
        if predict_duals:
            outputs = (
                6 * problem_specs["control_intervals"]
                + problem_specs["num_constraints"]
            )
        else:
            outputs = 6 * problem_specs["control_intervals"]

        self.shape = [inputs] + hidden_layers + [outputs]
        self.activation_fn = activation_fn

        self.wandb_run = wandb.init(
            project="learning_warmstarts",
            entity="laine-lab",
            config=config,
            job_type="benchmark_predictions",
        )

        model_at = self.wandb_run.use_artifact(version, type="model")
        model_dir = model_at.download()

        self.model = Benchmark.load_model(
            model_dir + "/" + model_path, self.shape, activation_fn
        )
        self.benchmark_folder = benchmark_folder

    def run(self):
        """Compare:
        1) Default params + no warmstart
        2) Warmstart params + no warmstart
        3) Default params + previous iteration warmstart (no lam_g)
        4) Warmstart params + previous iteration warmstart (no lam_g)
        5) Default params + neural net warmstart (no lam_g)
        6) Warmstart params +  neural net warmstart (no lam_g)
        7) Default params + neural net warmstart (including dual vars)
        8) Warmstart params + neural net warmstart (including dual vars)
        """
        running_iters = 8 * [0] if not self.predict_duals else 6 * [0]
        running_time = 8 * [0] if not self.predict_duals else 6 * [0]
        m = 0

        dataiter = self.load_data(
            dataset_type="benchmark", bench_folder=self.benchmark_folder
        )

        for sample in dataiter:
            if self.num_samples and m > self.num_samples:
                break

            self.problem.set_value(self.x0_param[[0, 1, 2, 3]], sample["x0"])
            self.problem.set_value(
                self.obstacle_param[[i for i in range(3 * self.N_OBST)]],
                sample["obstacles"],
            )

            iters, time = self.no_warmstart(
                False, sample["gt_x"], sample["gt_u"], sample["gt_lamg"]
            )
            running_iters[0] += iters
            running_time[0] += time

            print(
                f"Sample {m:<3}, Def params: no warmstart solved in {iters} iterations ({time:.4f} s)"
            )

            if self.ws_params:
                iters, time = self.no_warmstart(
                    True, sample["gt_x"], sample["gt_u"], sample["gt_lamg"]
                )
                running_iters[1] += iters
                running_time[1] += time

                print(
                    f"Sample {m:<3}, WS Params: no warmstart solved in {iters} iterations ({time:.4f} s)"
                )

            iters, time = self.prev_iter_warmstart(
                sample["prev_x"],
                sample["prev_u"],
                False,
                sample["gt_x"],
                sample["gt_u"],
                sample["gt_lamg"],
            )
            running_iters[2] += iters
            running_time[2] += time

            print(
                f"Sample {m:<3}, Def params: prev iter solved in {iters} iterations ({time:.4f} s)"
            )

            if self.ws_params:
                iters, time = self.prev_iter_warmstart(
                    sample["prev_x"],
                    sample["prev_u"],
                    True,
                    sample["gt_x"],
                    sample["gt_u"],
                    sample["gt_lamg"],
                )
                running_iters[3] += iters
                running_time[3] += time

                print(
                    f"Sample {m:<3}, WS params: prev iter solved in {iters} iterations ({time:.4f} s)"
                )

            iters, time = self.primary_learned_warmstart(
                sample["x0"],
                sample["obstacles"],
                False,
                sample["gt_x"],
                sample["gt_u"],
                sample["gt_lamg"],
            )
            running_iters[4] += iters
            running_time[4] += time

            print(
                f"Sample {m:<3}, Def params: primal vars solved in {iters} iterations ({time:.4f} s)"
            )

            if self.ws_params:
                iters, time = self.primary_learned_warmstart(
                    sample["x0"],
                    sample["obstacles"],
                    True,
                    sample["gt_x"],
                    sample["gt_u"],
                    sample["gt_lamg"],
                )
                running_iters[5] += iters
                running_time[5] += time

                print(
                    f"Sample {m:<3}, WS params: primal vars solved in {iters} iterations ({time:.4f} s)"
                )

            if self.predict_duals:
                iters, time = self.primary_dual_learned_warmstart(
                    sample["x0"],
                    sample["obstacles"],
                    False,
                    sample["gt_x"],
                    sample["gt_u"],
                    sample["gt_lamg"],
                )
                running_iters[4] += iters
                running_time[4] += time

                print(
                    f"Sample {m:<3}, Def params: primals & duals solved in {iters} iterations ({time:.4f} s)"
                )

                if self.ws_params:
                    iters, time = self.primary_dual_learned_warmstart(
                        sample["x0"],
                        sample["obstacles"],
                        True,
                        sample["gt_x"],
                        sample["gt_u"],
                        sample["gt_lamg"],
                    )
                    running_iters[5] += iters
                    running_time[5] += time

                    print(
                        f"Sample {m:<3}, WS params: primals & dua;s solved in {iters} iterations ({time:.4f} s)"
                    )

            m += 1

        avg_iters = [iters / m for iters in running_iters]
        avg_time = [time / m for time in running_time]

        results = self.create_results_dict(avg_iters, avg_time)
        wandb.log(results)

    def no_warmstart(
        self, use_warmstart_params, ground_truth_x, ground_truth_u, ground_truth_lam_g
    ):
        self.set_solver_params(use_warmstart_params)

        self.problem.set_initial(self.x_var, [0.0 for i in range(self.x_var.shape[0])])
        self.problem.set_initial(self.u_var, [0.0 for i in range(self.u_var.shape[0])])

        return self.solve()

    def prev_iter_warmstart(
        self,
        x_warmstart,
        u_warmstart,
        use_warmstart_params,
        ground_truth_x,
        ground_truth_u,
        ground_truth_lam_g,
    ):
        """Warmstart with previous iteration's solution"""

        self.set_solver_params(use_warmstart_params)

        self.problem.set_initial(self.x_var, x_warmstart)
        self.problem.set_initial(self.u_var, u_warmstart)

        return self.solve()

    def primary_learned_warmstart(
        self,
        x0,
        obstacles,
        use_warmstart_params,
        ground_truth_x,
        ground_truth_u,
        ground_truth_lam_g,
    ):
        """Neural Network's predicted warmstart for x and u. Does not include lam_g"""

        self.set_solver_params(use_warmstart_params)

        inputs = torch.from_numpy(np.concatenate((x0, obstacles)))
        outputs = self.model(inputs).detach().numpy()

        self.problem.set_initial(self.x_var, outputs[: 4 * self.N])
        self.problem.set_initial(self.u_var, outputs[4 * self.N : 6 * self.N])

        return self.solve()

    def primary_dual_learned_warmstart(
        self,
        x0,
        obstacles,
        use_warmstart_params,
        ground_truth_x,
        ground_truth_u,
        ground_truth_lam_g,
    ):
        """Neural Network's predicted warmstart for x, u, and lam_g"""

        self.set_solver_params(use_warmstart_params)

        inputs = torch.from_numpy(np.concatenate((x0, obstacles)))
        outputs = self.model(inputs).detach().numpy()

        self.problem.set_initial(self.x_var, outputs[: 4 * self.N])
        self.problem.set_initial(self.u_var, outputs[4 * self.N : 6 * self.N])
        self.problem.set_initial(self.problem.lam_g, outputs[6 * self.N :])

        return self.solve()

    def solve(self):
        """Solves the setup problem and returns the desired stats to be logged
        Easily change stats that will be returned by all comparison methods
        by modifying this method
        """
        try:
            solution = self.problem.solve()
            return (solution.stats()["iter_count"], solution.stats()["t_wall_total"])
        except RuntimeError as e:
            print(e)
            return (
                self.problem.debug.stats()["iter_count"],
                self.problem.debug.stats()["t_wall_total"],
            )

    def create_results_dict(self, avg_iters, avg_time):
        n_sections = 8 if self.predict_duals else 6

        params = ["Default" if i % 2 == 0 else "Warmstart" for i in range(n_sections)]
        warmstart = [
            "None",
            "None",
            "Prev Iter",
            "Prev Iter",
            "Neural Net",
            "Neural Net",
        ]

        iters_headings = []
        for param, method in zip(params, warmstart):
            iters_headings.append(f"{param} | {method} | Iters")

        time_headings = []
        for param, method in zip(params, warmstart):
            time_headings.append(f"{param} | {method} | Time")

        results = {}
        for i, heading in enumerate(iters_headings):
            results[heading] = avg_iters[i]

        for i, heading in enumerate(time_headings):
            results[heading] = avg_time[i]

        return results
