#!/bin/sh
#SBATCH --partition=stud-ewi
#SBATCH --qos=stud-ewi
#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu
#SBATCH --mem=1024
#SBATCH --mail-type=END

/usr/bin/scontrol show job -d "$SLURM_JOB_ID"

module use /opt/insy/modulefiles
module load cuda
module load cudnn
module load miniconda
conda activate env_thesis_ML
cd MLImplementation

srun python main.py
