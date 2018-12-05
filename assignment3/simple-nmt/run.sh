echo "bleu_rl"
python train.py \
    --n_epochs -1 \
    --model deen.iwslt_xnli_best.pth \
    --bimpm_pretrained_model_path BIMPM_multinli_iwslt_merged_vocab_best.pt \
    --train data/nli_split/nli.train \
    --valid data/iwslt/corpus.valid \
    --valid_nli data/nli_iwslt/nli.all \
    --lang deen \
    --print_every 40 \
    --rl_n_gram 4 \
    --rl_lr 0.01 \
    --rl_n_epochs 15 \
    --n_samples 1 \
    --temperature 1.0 \
    --reward_mode bleu \
    --pretrain \
    --batch_size 32 \
    --gpu_id 0 > train_rl_bleu_sgd_1e-2.log

echo "nli_rl"
python train.py \
    --n_epochs -1 \
    --model deen.iwslt_xnli_best.pth \
    --bimpm_pretrained_model_path BIMPM_multinli_iwslt_merged_vocab_best.pt \
    --train data/nli_split/nli.train \
    --valid data/iwslt/corpus.valid \
    --valid_nli data/nli_iwslt/nli.all \
    --lang deen \
    --print_every 40 \
    --rl_n_gram 4 \
    --rl_lr 0.01 \
    --rl_n_epochs 15 \
    --n_samples 1 \
    --temperature 1.0 \
    --reward_mode nli \
    --batch_size 32 \
    --gpu_id 0 > train_rl_nli_sgd_1e-2.log

echo "combined_rl"
python train.py \
    --n_epochs -1 \
    --model deen.iwslt_xnli_best.pth \
    --bimpm_pretrained_model_path BIMPM_multinli_iwslt_merged_vocab_best.pt \
    --train data/nli_split/nli.train \
    --valid data/iwslt/corpus.valid \
    --valid_nli data/nli_iwslt/nli.all \
    --lang deen \
    --print_every 40 \
    --rl_n_gram 4 \
    --rl_lr 0.01 \
    --rl_n_epochs 15 \
    --n_samples 1 \
    --temperature 1.0 \
    --reward_mode combined \
    --batch_size 32 \
    --gpu_id 0 > train_rl_combined_sgd_1e-2.log

echo "bleu_mrt"
python train.py \
    --n_epochs -1 \
    --model deen.iwslt_xnli_best.pth \
    --bimpm_pretrained_model_path BIMPM_multinli_iwslt_merged_vocab_best.pt \
    --train data/nli_split/nli.train \
    --valid data/iwslt/corpus.valid \
    --valid_nli data/nli_iwslt/nli.all \
    --lang deen \
    --print_every 40 \
    --rl_n_gram 4 \
    --rl_lr 0.05 \
    --rl_n_epochs 15 \
    --n_samples 5 \
    --use_minimum_risk \
    --temperature 1.0 \
    --reward_mode bleu \
    --batch_size 32 \
    --max_grad_norm 0.1 \
    --gpu_id 0 > train_mrt_bleu_sgd_5e-2_clip1e-1.log

echo "nli_mrt"
python train.py \
    --n_epochs -1 \
    --model deen.iwslt_xnli_best.pth \
    --bimpm_pretrained_model_path BIMPM_multinli_iwslt_merged_vocab_best.pt \
    --train data/nli_split/nli.train \
    --valid data/iwslt/corpus.valid \
    --valid_nli data/nli_iwslt/nli.all \
    --lang deen \
    --print_every 40 \
    --rl_n_gram 4 \
    --rl_lr 0.05 \
    --rl_n_epochs 15 \
    --n_samples 5 \
    --use_minimum_risk \
    --temperature 1.0 \
    --reward_mode nli \
    --batch_size 32 \
    --max_grad_norm 0.1 \
    --gpu_id 0 > train_mrt_nli_sgd_5e-2_clip1e-1.log

echo "combined_mrt"
python train.py \
    --n_epochs -1 \
    --model deen.iwslt_xnli_best.pth \
    --bimpm_pretrained_model_path BIMPM_multinli_iwslt_merged_vocab_best.pt \
    --train data/nli_split/nli.train \
    --valid data/iwslt/corpus.valid \
    --valid_nli data/nli_iwslt/nli.all \
    --lang deen \
    --print_every 40 \
    --rl_n_gram 4 \
    --rl_lr 0.05 \
    --rl_n_epochs 15 \
    --n_samples 5 \
    --use_minimum_risk \
    --temperature 1.0 \
    --reward_mode combined \
    --batch_size 32 \
    --max_grad_norm 0.1 \
    --gpu_id 0 > train_mrt_combined_sgd_5e-2_clip1e-1.log
