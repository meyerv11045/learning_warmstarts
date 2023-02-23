import argparse
import os
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import wandb

from learning_warmstarts.dataset.data_generator import (
    generate_benchmark_problems,
    generate_training_problems,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--num_datapoints",
        type=int,
        required=True,
        help="Number of datapoints to generate",
    )
    parser.add_argument(
        "-p",
        "--processes",
        type=int,
        help="Number of processes to generate data in parallel",
        default=1,
    )
    parser.add_argument(
        "-b", "--benchmark", action="store_true", help="Generate benchmark data"
    )
    parser.add_argument(
        "-s", "--seed", help="Random number generator seeds for each process"
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        required=True,
        help="Output folder for .npy files from each process",
    )
    parser.add_argument(
        "-nc",
        "--num_per_chunk",
        type=int,
        default=10,
        help="Number of similar datapoints to generate in a chunk",
    )
    parser.add_argument(
        "-t",
        "--testing",
        action="store_true",
        help="Disables wandb logging for testing purposes",
    )
    args = parser.parse_args()

    nc = args.num_per_chunk

    timeout = 3 * 60 * nc  # 300 s (3 min) for each of the problems in a chunk

    # generate n datapoints so n / nc unique problems

    num_problems = int(args.num_datapoints / nc)

    # starter seed to create seeds for each function
    if args.seed:
        random.seed(args.seed)
    seeds = [random.random() * (i + 1) for i in range(num_problems)]

    if not os.path.isdir(args.output_folder):
        os.mkdir(args.output_folder)

    with ProcessPoolExecutor(max_workers=args.processes) as executor:

        if args.benchmark:
            futures = [
                executor.submit(generate_benchmark_problems, seed, nc) for seed in seeds
            ]
        else:
            futures = [
                executor.submit(generate_training_problems, seed, nc) for seed in seeds
            ]

        i = 1
        for future in futures:
            try:
                results = future.result(timeout=timeout)
                np.save(f"./{args.output_folder}/chunk{i}.npy", results)
                i += 1

            except Exception as e:
                print("Error with a problem chunk: ", e)
