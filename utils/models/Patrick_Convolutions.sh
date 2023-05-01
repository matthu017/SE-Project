#!/bin/bash
#SBATCH --time=64:00:00
#SBATCH --nodes=1 #--ntasks=28 --gpus-per-node=1
#SBATCH --job-name=5441_LAB_1_TEST 
# account for CSE 5441 Au'21
#SBATCH --account=PAS2400
export PGM=Patrick_Convolutions.py  # <--- CHANGE THIS
export SLURM_SUBMIT_DIR=/fs/scratch/PAS2400/ASR_groupofdeskclosesttothedoor/final_project/Intelligibility-Oriented-Audio-Visual-Speech-Enhancement/utils/models # <--- CHANGE THIS, everything else stay the same
export ARGUMENT=training_data/

echo job started at `date`
echo on compute node `cat $PBS_NODEFILE`

module load miniconda3
source activate tts

cd ${SLURM_SUBMIT_DIR} # change into directory with program to test on node

python ${SLURM_SUBMIT_DIR}/${PGM}    # <--- CHANGE THIS for arguments