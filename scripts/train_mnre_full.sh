set -euo pipefail

GPU="${GPU:-0}"
N="${N:-5}"
export CUDA_VISIBLE_DEVICES="${GPU}"

python train_full.py \
  --dataset mnre --device cuda --cuda_visible_devices "${GPU}" \
  --N "${N}" --K 1 --Q 5 \
  --ckpt "ckpt/mnre_${N}w1s_full.pth.tar" \
  --metrics_path "logs/mnre_${N}w1s_full.json"
