#!/bin/sh
#
#SBATCH --account=education-eemcs-msc
#SBATCH --job-name="TStrain"
#SBATCH --partition=gpu
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=s.hoogeveen@student.tudelft.nl
#SBATCH --output=slurm-%x-%j.out

module load 2022r2
module load cuda/11.3
module load cudnn/11.3-8.2.1.32
module load python/3.8.12
module load py-pip

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate tf
cd /scratch/syllehoogeveen/MLImplementation

srun python main.py train TS AE_straight_OVERFIT_val001_E126_f16_z150