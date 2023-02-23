import argparse
from learning_warmstarts.testing.benchmark import Benchmark

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-v', '--version', required=True, help='input ex: trained_model:v30')
    parser.add_argument('-p','--model_path', default='model.pt')
    parser.add_argument('-b', '--benchmark_folder', required=True)
    parser.add_argument('-hl','--hidden_layers', required=True)
    parser.add_argument('-pd', '--predict_duals', action='store_true')
    parser.add_argument('-n', '--num_samples', type=int, required=True)
    parser.add_argument('-a', '--activation_fn', default='relu')
    parser.add_argument('-ws', '--ws_params', action='store_true', default=False)
    args = parser.parse_args()

    args.hidden_layers = [int(layer_size) for layer_size in args.hidden_layers.split(',')]

    bnchmrk = Benchmark(**vars(args))
    
    bnchmrk.run()