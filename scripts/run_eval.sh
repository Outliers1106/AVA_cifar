echo "=============================================================================================================="
echo "Please run the scipt as: "
echo "bash run_eval.sh"
echo "=============================================================================================================="

PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)

python ${PROJECT_DIR}/../eval.py \
    --device_id 0 \
    --device_num 1 \
    --device_target Ascend \
    --network resnet18 \
    --train_data_dir /home/tuyanlun/code/mindspore_r1.0/cifar/train \
    --test_data_dir /home/tuyanlun/code/mindspore_r1.0/cifar/test \
    --load_ckpt_path /path_to_load_ckpt/somecheckpoint.ckpt