import argparse
from learning_warmstarts.testing.overfit import OverfitPredictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', required=True, help='input ex: trained_model:v30')
    parser.add_argument('-p','--model_path', default='model.pt')
    parser.add_argument('-tr','--train_folder', default='train_data')
    parser.add_argument('-te','--test_folder', default='test_data')
    parser.add_argument('-hl','--hidden_layers', required=True)
    parser.add_argument('-wp', '--use_warmstart_params', action='store_true')
    parser.add_argument('-n', '--n_samples', type=int, default=None)
    parser.add_argument('-s', '--save_trajectories', action='store_true')
    parser.add_argument('-ds', '--dynamic_smoothing', action='store_true')
    parser.add_argument('-a', '--activation_fn', default='relu')
    args = parser.parse_args()

    args.hidden_layers = [int(layer_size) for layer_size in args.hidden_layers.split(',')]
    
    overfit_predictions = OverfitPredictions(**vars(args))
    
    overfit_predictions.run()