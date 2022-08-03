#!/bin/sh
#
#SBATCH --account=research-eemcs-diam
#SBATCH --job-name="hypertun"
#SBATCH --partition=gpu
#SBATCH --time=4:00:00
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
module load miniconda3/4.12.0

#module load py-pip

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate env_thesis_ML


cd /scratch/syllehoogeveen/MLImplementation

#python -m pip install --user -r requirements_DB.txt

#python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

srun python main.py train total