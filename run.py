from clearml import Task
from subprocess import Popen, PIPE
import pdb
print("Starting RaMViD")

Task.init()
# No. of devices
n = 4
args = {
    'data_dir': '../NeurIPS2021-traffic4cast/data/raw/',
    'image_size': 128,
    'num_channels': 16,
    'num_res_blocks': 1,
    'scale_time_dim': 0,
    'diffusion_steps': 250,
    'noise_schedule': 'linear',
    'lr': 2e-5,
    'batch_size': 1,
    'microbatch': 1,
    'seq_len': 3,
    'max_num_mask_frames': 2,
    'uncondition_rate': 0.25,
    'rgb': False,
    'schedule_sampler': 'loss-second-moment',
    }
# convert args to Popen list of strings
# append keys and values of args to a list
args_str = ['python', 'scripts/video_train.py']
for k, v in args.items():
    args_str.append(f'--{k}')
    args_str.append(f'{v}')
p = Popen(args_str, stdout=PIPE, stderr=PIPE, shell=False)

stdout, stderr = p.communicate()
print (stdout)
print (stderr)

#print("\n!!\n")

