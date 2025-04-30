#!/bin/bash  
#SBATCH --job-name=cfd_thesis 
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=cqk769@alumni.ku.dk

# Your commands here
module purge

module load pytorch/2.2.2
module load torchvision/0.17
module load gcc/10.2.0

source thesis_venv/bin/activate

# Run your Python script
output=$(python pca_decode.py)
echo "$output"
