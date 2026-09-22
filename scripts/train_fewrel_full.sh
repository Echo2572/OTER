set -euo pipefail

GPU="${GPU:-0}"
N="${N:-5}"
K="${K:-1}"
export CUDA_VISIBLE_DEVICES="${GPU}"

python train_full.py \
  --dataset fewrel --device cuda --cuda_visible_devices "${GPU}" \
  --N "${N}" --K "${K}" --Q 5 \
  --ckpt "ckpt/fewrel_${N}w${K}s_full.pth.tar" \
  --metrics_path "logs/fewrel_${N}w${K}s_full.json"

