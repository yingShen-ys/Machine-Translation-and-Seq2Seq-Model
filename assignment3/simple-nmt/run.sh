echo "bleu_mrt"
python train.py \
    --n_epochs -1 \
    --model deen.iwslt_xnli_best.pth \
    --bimpm_pretrained_model_path BIMPM_multinli_iwslt_merged_vocab_best.pt \
    --train data/xnli/nli_no_neutral/nli.train.no_neutral \
    --valid data/iwslt/corpus.valid \
    --valid_nli data/xnli/nli_no_neutral/nli.valid.no_neutral \
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
    --max_length 50 \
    --gpu_id 0 > train_mrt_bleu_sgd_5e-2_clip1e-1.no_neutral.valid_iwslt.log

echo "nli_mrt"
python train.py \
    --n_epochs -1 \
    --model deen.iwslt_xnli_best.pth \
    --bimpm_pretrained_model_path BIMPM_multinli_iwslt_merged_vocab_best.pt \
    --train data/xnli/nli_no_neutral/nli.train.no_neutral \
    --valid data/iwslt/corpus.valid \
    --valid_nli data/xnli/nli_no_neutral/nli.valid.no_neutral \
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
    --max_length 50 \
    --gpu_id 0 > train_mrt_nli_sgd_5e-2_clip1e-1.no_neutral.valid_iwslt.log

echo "combined_mrt"
python train.py \
    --n_epochs -1 \
    --model deen.iwslt_xnli_best.pth \
    --bimpm_pretrained_model_path BIMPM_multinli_iwslt_merged_vocab_best.pt \
    --train data/xnli/nli_no_neutral/nli.train.no_neutral \
    --valid data/iwslt/corpus.valid \
    --valid_nli data/xnli/nli_no_neutral/nli.valid.no_neutral \
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
    --max_length 50 \
    --gpu_id 0 > train_mrt_combined_sgd_5e-2_clip1e-1_fixed_start_weight.no_neutral.valid_iwslt.log
