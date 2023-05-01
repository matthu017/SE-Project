#!/bin/bash
#SBATCH --time=64:00:00
#SBATCH --nodes=1 --ntasks-per-node=28 --gpus-per-node=1
#SBATCH --job-name=final_project_AVSE_audio_only_stoiloss_speech_separation_version_Pat
# account for CSE 5539 Spring 2023
#SBATCH --account=PAS2400

#SBATCH --job-name /fs/scratch/PAS2400/ASR_groupofdeskclosesttothedoor/final_project/Intelligibility-Oriented-Audio-Visual-Speech-Enhancement_chengyu/logs/lightning_logs/version_24550817/checkpoints/epoch=11-step=110400.ckpt

echo job started at `date`

module load miniconda3
source activate avse



#python train.py --exp_name $SLURM_JOB_NAME --log_dir ./logs --a_only True --gpu 1 --mode SE --max_epochs 100 --loss stoi

# python test.py --a_only False --snrs -5 --ckpt_path ./checkpoints/final_project_AVSE_audio_visual_stoiloss_speech_enhancement_version01-epoch=69-val_loss=0.28.ckpt --save_root ./enhanced --loss stoi --mode SE
# python test_pat.py --a_only False --snrs -5 --ckpt_path ./checkpoints/final_project_AVSE_audio_visual_stoiloss_speech_enhancement_version_Pat-last.ckpt --save_root ./enhanced --loss stoi --mode SE
# python test.py --a_only True --snrs -5 --ckpt_path ./checkpoints/final_project_AVSE_audio_only_stoiloss_speech_enhancement_version01-epoch=24-val_loss=0.28.ckpt --save_root ./enhanced --loss stoi --mode SE
# python test.py --a_only False --snrs 0 --ckpt_path ./checkpoints/final_project_AVSE_audio_visual_stoiloss_speech_enhancement_version01-epoch=69-val_loss=0.28.ckpt --save_root ./enhanced --loss stoi --mode SE
# python test_pat.py --a_only False --snrs 0 --ckpt_path ./checkpoints/final_project_AVSE_audio_visual_stoiloss_speech_enhancement_version_Pat-last.ckpt --save_root ./enhanced --loss stoi --mode SE
# python test.py --a_only True --snrs 0 --ckpt_path ./checkpoints/final_project_AVSE_audio_only_stoiloss_speech_enhancement_version01-epoch=24-val_loss=0.28.ckpt --save_root ./enhanced --loss stoi --mode SE
# python test.py --a_only False --snrs 5 --ckpt_path ./checkpoints/final_project_AVSE_audio_visual_stoiloss_speech_enhancement_version01-epoch=69-val_loss=0.28.ckpt --save_root ./enhanced --loss stoi --mode SE
# python test_pat.py --a_only False --snrs 5 --ckpt_path ./checkpoints/final_project_AVSE_audio_visual_stoiloss_speech_enhancement_version_Pat-last.ckpt --save_root ./enhanced --loss stoi --mode SE
# python test.py --a_only True --snrs 5 --ckpt_path ./checkpoints/final_project_AVSE_audio_only_stoiloss_speech_enhancement_version01-epoch=24-val_loss=0.28.ckpt --save_root ./enhanced --loss stoi --mode SE

# python test.py --a_only False --snrs -5 --ckpt_path $SLURM_JOB_NAME --save_root ./enhanced_ssinse --loss stoi --mode SE
# python test.py --a_only False --snrs 0 --ckpt_path $SLURM_JOB_NAME --save_root ./enhanced_ssinse --loss stoi --mode SE
# python test.py --a_only False --snrs 5 --ckpt_path $SLURM_JOB_NAME --save_root ./enhanced_ssinse --loss stoi --mode SE

python test.py --a_only False --snrs 0 --ckpt_path ./checkpoints/final_project_AVSE_audio_visual_stoiloss_version01-epoch=29-val_loss=0.18.ckpt --save_root ./separated --loss stoi --mode SS
python test_pat.py --a_only False --snrs 0 --ckpt_path ./checkpoints/final_project_AVSE_audio_visual_stoiloss_speech_separation_version_Pat-last.ckpt --save_root ./separated --loss stoi --mode SS
python test.py --a_only True --snrs 0 --ckpt_path ./checkpoints/final_project_AVSE_audio_only_stoiloss_version01-epoch=99-val_loss=0.21.ckpt --save_root ./separated --loss stoi --mode SS
python test.py --a_only False --snrs 5 --ckpt_path ./checkpoints/final_project_AVSE_audio_visual_stoiloss_version01-epoch=29-val_loss=0.18.ckpt --save_root ./separated --loss stoi --mode SS
python test_pat.py --a_only False --snrs 5 --ckpt_path ./checkpoints/final_project_AVSE_audio_visual_stoiloss_speech_separation_version_Pat-last.ckpt --save_root ./separated --loss stoi --mode SS
python test.py --a_only True --snrs 5 --ckpt_path ./checkpoints/final_project_AVSE_audio_only_stoiloss_version01-epoch=99-val_loss=0.21.ckpt --save_root ./separated --loss stoi --mode SS
python test.py --a_only False --snrs 10 --ckpt_path ./checkpoints/final_project_AVSE_audio_visual_stoiloss_version01-epoch=29-val_loss=0.18.ckpt --save_root ./separated --loss stoi --mode SS
python test_pat.py --a_only False --snrs 10 --ckpt_path ./checkpoints/final_project_AVSE_audio_visual_stoiloss_speech_separation_version_Pat-last.ckpt --save_root ./separated --loss stoi --mode SS
python test.py --a_only True --snrs 10 --ckpt_path ./checkpoints/final_project_AVSE_audio_only_stoiloss_version01-epoch=99-val_loss=0.21.ckpt --save_root ./separated --loss stoi --mode SS