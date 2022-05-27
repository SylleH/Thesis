#!/bin/sh
#SBATCH --partition=stud-ewi
#SBATCH --qos=stud-ewi
#SBATCH --time=0:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12288
#SBATCH --mail-type=END

/usr/bin/scontrol show job -d "$SLURM_JOB_ID"

module use /opt/insy/modulefiles
module load miniconda/3.8
cd /tudelft.net/staff-umbrella/trasientcnn/DataGeneration
conda create --file requirements.txt --name env_thesis_data

