#!/bin/bash -l
#SBATCH --requeue
#SBATCH --job-name=%x
#SBATCH --partition=use-everything
##SBATCH --constraint=volta
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=8:0:0     # walltime
#SBATCH --output="slurm_logs/%x.out"
#SBATCH --error="slurm_logs/%x.err"
#SBATCH --open-mode=append
#SBATCH --mail-user=hvgazula@umich.edu
#SBATCH --mail-type=BEGIN,FAIL,END

# module load openmind/anaconda/3-2023.09-0
source activate nobrainer_satra
echo "Submitted job: $SLURM_JOB_ID"
python -u 1.2.0/kwyk_train.py
