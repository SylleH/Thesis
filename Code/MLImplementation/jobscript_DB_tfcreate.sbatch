#!/bin/sh
#
#SBATCH --account=research-eemcs-diam
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s.hoogeveen@student.tudelft.nl
#SBATCH --output=slurm-%x-%j.out

module load 2022r2
module load openmpi
module load cuda/11.1.1
module load cudnn/8.0.5.39-11.1
module load miniconda3

conda create --name tf29 python

conda activate tf29
pip install tensorflow
pip install --user -r requirements_DB.txt
pip install -q -U keras-tuner

python3 -c "import tensorflow as tf; print(tf.version.VERSION); print(tf.config.list_physical_devices('GPU'))"
