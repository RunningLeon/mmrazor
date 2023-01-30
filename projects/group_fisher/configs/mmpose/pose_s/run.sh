export CPUS_PER_TASK=20
# bash ./tools/slurm_test.sh mm_razor1 pose_s ./projects/group_fisher/configs/mmpose/pose_s/pose_s_pretrain.py ./work_dirs/pretrained/rtmpose/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth

bash ./tools/slurm_train.sh mm_razor1 pose_s ./projects/group_fisher/configs/mmpose/pose_s/pose_s_prune.py ./work_dirs/pose_s_prune
bash ./tools/slurm_train.sh mm_razor1 pose_s ./projects/group_fisher/configs/mmpose/pose_s/pose_s_finetune.py ./work_dirs/pose_s_finetune
