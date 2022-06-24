#!/bin/sh
#
#SBATCH --account=education-eemcs-msc
#SBATCH --job-name="AEtrain"
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
module load miniconda3/4.12.0

conda activate env_ML_thesis_2
rsync -av --no-perms /tudelft.net/staff-umbrella/trasientcnn/MLImplementation /scratch/syllehoogeveen/
cd /scratch/syllehoogeveen/MLImplementation

srun python main.py train AE

rsync -av /scratch/syllehoogeveen/MLImplementation /tudelft.net/staff-umbrella/trasientcnn/
