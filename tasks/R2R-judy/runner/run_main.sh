# ========= Follower ========= 

python tasks/R2R-judy/main.py \
--config-file tasks/R2R-judy/configs/follower/follower_config.yaml \
TRAIN.DEVICE 0

python ./tasks/R2R-judy/main.py \
--config-file tasks/R2R-judy/configs/follower/follower_cl_config.yaml \
TRAIN.DEVICE 0 \
TRAIN.CLMODE NAIVE \

python ./tasks/R2R-judy/main.py \
--config-file tasks/R2R-judy/configs/follower/follower_cl_config.yaml \
TRAIN.DEVICE 0 \
TRAIN.SELF_PACE.WCTRL 0.0 \
TRAIN.SELF_PACE.MIU 3.0 \
TRAIN.SELF_PACE.FUNC linear


# ========= Self-Monitor ========= 

python tasks/R2R-judy/main.py \
--config-file tasks/R2R-judy/configs/monitor/selfmonitor_config.yaml \
TRAIN.DEVICE 0

python ./tasks/R2R-judy/main.py \
--config-file tasks/R2R-judy/configs/monitor/selfmonitor_cl_config.yaml \
OUTPUT.CKPT_DIR tasks/R2R-judy/snapshots/checkpoints/self-monitor/naive-curriculum \
TRAIN.DEVICE 0 \
TRAIN.CLMODE NAIVE

python ./tasks/R2R-judy/main.py \
--config-file tasks/R2R-judy/configs/monitor/selfmonitor_cl_config.yaml \
TRAIN.DEVICE 1 \
TRAIN.SELF_PACE.WCTRL 1.0 \
TRAIN.SELF_PACE.MIU 3.0 \
TRAIN.SELF_PACE.FUNC binary


# ========= EnvDrop ========= 

python tasks/R2R-judy/main.py \
--config-file tasks/R2R-judy/configs/envdrop/envdrop_config.yaml \
TRAIN.DEVICE 1 

python ./tasks/R2R-judy/main.py \
--config-file tasks/R2R-judy/configs/envdrop/envdrop_cl_config.yaml \
--seed $rnd_seed \
OUTPUT.CKPT_DIR tasks/R2R-judy/snapshots/checkpoints/envdrop/naive-curriculum \
TRAIN.DEVICE 1 \
TRAIN.CLMODE NAIVE

python ./tasks/R2R-judy/main.py \
--config-file tasks/R2R-judy/configs/envdrop/envdrop_cl_config.yaml \
TRAIN.CLMODE SELF-PACE \
TRAIN.DEVICE 0 \
TRAIN.SELF_PACE.WCTRL 0.5 \
TRAIN.SELF_PACE.MIU 2.0 \
TRAIN.SELF_PACE.FUNC linear
