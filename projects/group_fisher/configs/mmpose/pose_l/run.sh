export CPUS_PER_TASK=20
# bash ./tools/slurm_test.sh mm_razor1 pose_l ./projects/group_fisher/configs/mmpose/pose_l/pose_l_pretrain.py ./work_dirs/pose_l_pretrain

bash ./tools/slurm_train.sh mm_razor1 pose_l ./projects/group_fisher/configs/mmpose/pose_l/pose_l_prune.py ./work_dirs/pose_l_prune
bash ./tools/slurm_train.sh mm_razor1 pose_l ./projects/group_fisher/configs/mmpose/pose_l/pose_l_finetune.py ./work_dirs/pose_l_finetune
