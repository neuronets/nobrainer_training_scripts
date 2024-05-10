#!/bin/bash -l
##SBATCH --requeue
#SBATCH --job-name=%x
#SBATCH --partition=gablab
##SBATCH --constraint=volta
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100-gablab:1
#SBATCH --mem=96G
#SBATCH --time=8:0:0     # walltime
#SBATCH --output="logs/%x.out"
#SBATCH --error="logs/%x.err"
#SBATCH --open-mode=append
#SBATCH --mail-user=hvgazula@umich.edu
#SBATCH --mail-type=BEGIN,FAIL,END

# module load openmind/anaconda/3-2023.09-0
source activate nobrainer_satra
echo "Submitted job: $SLURM_JOB_ID"
python -u kwyk_train.py
