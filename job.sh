#!/bin/bash -l
#SBATCH --job-name=test_mem	        # Name of your job
#SBATCH --account=hyper-ep          # Your Slurm account
#SBATCH --partition=tier3           # Partition name (or scavenger if necessary)
#SBATCH --qos=qos_tier3             # QOS to match your account setup
#SBATCH -n 1                        # Number of tasks
#SBATCH -c 1                        # Number of CPU cores
#SBATCH --mem=50g                   # Memory allocation
#SBATCH --gres=gpu:a100:1           # Request 1 GPU
#SBATCH --mail-type=all             # Notifications for all job events
#SBATCH --mail-user=sv6234@g.rit.edu  # Your email for notifications

# Load the conda environment and run your script
conda activate miccai
python main.py
