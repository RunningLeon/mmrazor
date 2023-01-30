export CPUS_PER_TASK=20
# bash ./tools/slurm_test.sh mm_razor1 pose_m ./projects/group_fisher/configs/mmpose/pose_m/pose_m_pretrain.py ./work_dirs/pretrained/rtmpose/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth

bash ./tools/slurm_train.sh mm_razor1 pose_m ./projects/group_fisher/configs/mmpose/pose_m/pose_m_prune.py ./work_dirs/pose_m_prune
bash ./tools/slurm_train.sh mm_razor1 pose_m ./projects/group_fisher/configs/mmpose/pose_m/pose_m_finetune.py ./work_dirs/pose_m_finetune
