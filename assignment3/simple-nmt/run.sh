python train.py \
    --n_epochs -1 \
    --model deen.merged_vocab_best.pth \
    --bimpm_pretrained_model_path BIMPM_best_merged_vocab.pt \
    --train data/iwslt/corpus.valid \
    --valid data/test \
    --lang deen \
    --print_every 40 \
    --rl_n_gram 4 \
    --rl_lr 0.001 \
    --rl_n_epochs 8 \
    --n_samples 3 \
    --use_minimum_risk \
    --temperature 1.0 \
    --reward_mode bleu \
    --pretrain \
    --batch_size 32  > train_bleu_sgd_1e-3.log

echo "bleu_adam"

python train.py \
    --n_epochs -1 \
    --model deen.merged_vocab_best.pth \
    --bimpm_pretrained_model_path BIMPM_best_merged_vocab.pt \
    --train data/iwslt/corpus.valid \
    --valid data/test \
    --lang deen \
    --print_every 40 \
    --rl_n_gram 4 \
    --rl_lr 0.0001 \
    --rl_n_epochs 8 \
    --n_samples 3 \
    --use_minimum_risk \
    --temperature 1.0 \
    --reward_mode bleu \
    --pretrain \
    --adam \
    --batch_size 32  > train_bleu_adam_1e-4.log

echo "combined_sgd"

python train.py \
    --n_epochs -1 \
    --model deen.merged_vocab_best.pth \
    --bimpm_pretrained_model_path BIMPM_best_merged_vocab.pt \
    --train data/nli_iwslt/nli.all \
    --valid data/iwslt/corpus.valid \
    --lang deen \
    --print_every 40 \
    --rl_n_gram 4 \
    --rl_lr 0.001 \
    --rl_n_epochs 8 \
    --n_samples 3 \
    --use_minimum_risk \
    --temperature 1.0 \
    --reward_mode combined \
    --batch_size 32 > train_combined_sgd_1e-3.log

echo "combined_adam"

python train.py \
    --n_epochs -1 \
    --model deen.merged_vocab_best.pth \
    --bimpm_pretrained_model_path BIMPM_best_merged_vocab.pt \
    --train data/nli_iwslt/nli.all \
    --valid data/iwslt/corpus.valid \
    --lang deen \
    --print_every 40 \
    --rl_n_gram 4 \
    --rl_lr 0.0001 \
    --rl_n_epochs 8 \
    --n_samples 3 \
    --use_minimum_risk \
    --temperature 1.0 \
    --reward_mode combined \
    --adam \
    --batch_size 32 > train_combined_adam_1e-4.log