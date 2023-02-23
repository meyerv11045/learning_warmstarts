import random
import glob
import wandb
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_folder')
parser.add_argument('-o', '--output_folder')
parser.add_argument('-s', '--seed')
parser.add_argument('-nc', '--num_chunk', type=int)
args = parser.parse_args()

if not os.path.isdir(args.output_folder): os.mkdir(args.output_folder)

if args.seed:
    random.seed(args.seed)

files = glob.glob(f'{args.input_folder}/*.npy')

random.shuffle(files)

data = np.load(files[0],allow_pickle=True)

for file in files[1:]:
    chunk = np.load(file, allow_pickle=True)
    data = np.concatenate([data, chunk], axis=0)
    np.random.shuffle(data)

c_n = 0    
for i in range(0, data.shape[0], args.num_chunk):
    np.save(f'{args.output_folder}/chunk{c_n}.npy',data[i:i+args.num_chunk,:])
    c_n += 1

# config = dict(
#     input = args.input_path,
#     seed = args.seed
# )

# run = wandb.init(project='learning_warmstarts', entity='laine-lab', job_type='dataset_upload')

# lines = []
# with open(args.input_path, 'r') as f:
#     reader = csv.reader(f)

#     for row in reader:
#         if len(row) > 0:
#             lines.append(row)
    
# random.shuffle(lines)

# with open(args.output_path, 'w') as f:
#     writer = csv.writer(f)

#     for row in lines:
#         writer.writerow(row)

# artifact = wandb.Artifact(name='Obstacle Avoidance Dataset', type='dataset', description = args.notes)
# artifact.add_file(args.output_path)
# run.log_artifact(artifact)