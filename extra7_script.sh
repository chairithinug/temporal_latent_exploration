#!/bin/bash  
#SBATCH --job-name=cfd_thesis 
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=cqk769@alumni.ku.dk
#SBATCH --gres=gpu:a40:1

# Your commands here
module purge

module load pytorch/2.2.2
module load torchvision/0.17
module load gcc/10.2.0

export PHYSICSNEMO_DISTRIBUTED_INITIALIZATION_METHOD=''
source thesis_venv/bin/activate

# Run your Python script
output=$(python extrapolation_7_walk.py)
echo "$output"
