from clearml import Task
from subprocess import Popen, PIPE
import pdb
print("Starting RaMViD")

task = Task.init()
# No. of devices
n = 4
# Set clearml task parameter n for number of devices
task.set_parameter("n", n)

args = {
    'data_dir': '/data/raw/',
    'image_size': 128,
    'num_channels': 64,
    'num_res_blocks': 0,
    'scale_time_dim': 0,
    'diffusion_steps': 250,
    'noise_schedule': 'linear',
    'lr': 2e-5,
    'batch_size': 16,
    'microbatch': 3,
    'seq_len': 72,
    'max_num_mask_frames': 10,
    'uncondition_rate': 0.25,
    'rgb': False,
    'schedule_sampler': 'loss-second-moment',
    }
task.connect(args)
# convert args to Popen list of strings
# append keys and values of args to a list
if n>1:
    # run with mpirun
    args_str = ['mpirun', '-n', str(n), 'python', 'scripts/video_train.py']
else:
    args_str = ['python', 'scripts/video_train.py']
for k, v in args.items():
    args_str.append(f'--{k}')
    args_str.append(f'{v}')
p = Popen(args_str, stdout=PIPE, stderr=PIPE, shell=False)

stdout, stderr = p.communicate()
print (stdout)
print (stderr)

#print("\n!!\n")
