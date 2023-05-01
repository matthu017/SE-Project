#!/bin/bash
#SBATCH --time=64:00:00
#SBATCH --nodes=1 --ntasks-per-node=28 --gpus-per-node=1
#SBATCH --job-name=final_project_AVSE_audio_only_stoiloss_speech_separation_version_Pat
# account for CSE 5539 Spring 2023
#SBATCH --account=PAS2400

echo job started at `date`

module load miniconda3
source activate avse

python train.py --exp_name $SLURM_JOB_NAME --log_dir ./logs --a_only True --gpu 1 --mode SE --max_epochs 100 --loss stoi
