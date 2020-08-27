#!/bin/sh



#BSUB -q gpuv100
#BSUB -J BiGAN
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:35
#BSUB -R "rusage[mem=4GB]"
#BSUB -u benja.lazar@gmail.com
##BSUB -B
##BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu-%J.out
#BSUB -e gpu_%J.err

nvidia-smi
module load python3
module load cuda/10.2
module load matplotlib/2.0.2-python-3.6.2

python3 BiGAN_runner_hpc.py
