#!/bin/sh

source="gl"
auxiliary="pt"

original_src_emb="embeddings/prefix-vector-${source}.txt"
auxiliary_src_emb="embeddings/prefix-vector-${auxiliary}.txt"

train_original_src="prefixed_data/prefixed-train.en-${source}.${source}.txt"
train_original_tgt="data/train.en-${source}.en.txt"
train_auxiliary_src="prefixed_data/prefixed-train.en-${auxiliary}.${auxiliary}.txt"
train_auxiliary_tgt="data/train.en-${auxiliary}.en.txt"

dev_original_src="prefixed_data/prefixed-dev.en-${source}.${source}.txt"
dev_original_tgt="data/dev.en-${source}.en.txt"
dev_auxiliary_src="prefixed_data/prefixed-dev.en-${auxiliary}.${auxiliary}.txt"
dev_auxiliary_tgt="data/dev.en-${auxiliary}.en.txt"

vocab="vocab/${source}${auxiliary}-en.bin"
dev_src="combined_data/dev.${source}${auxiliary}-en.${source}${auxiliary}.txt"
dev_tgt="combined_data/dev.${source}${auxiliary}-en.en.txt"
test_src="prefixed_data/prefixed-test.en-$source.$source.txt"
test_tgt="data/test.en-$source.en.txt"

train_src="combined_data/train.${source}${auxiliary}-en.${source}${auxiliary}.txt"
train_tgt="combined_data/train.${source}${auxiliary}-en.en.txt"

tgt_emb="embeddings/vectors-en.txt"
src_emb="embeddings/prefix-vector-${source}${auxiliary}.txt"

work_dir="work_dir_basic_concatenate_${source}_${auxiliary}"

cat $train_original_src $train_auxiliary_src > $train_src
cat $dev_original_src $dev_auxiliary_src > $dev_src
cat $train_original_tgt $train_auxiliary_tgt > $train_tgt
cat $dev_original_tgt $dev_auxiliary_tgt > $dev_tgt
cat $original_src_emb $auxiliary_src_emb > $src_emb

python vocab.py --train-src $train_src --train-tgt $train_tgt $vocab

mkdir -p ${work_dir}
echo save results to ${work_dir}

python nmt.py \
    train \
    --model-type original_lstm \
    --cuda \
    --seed 233 \
    --src-embedding-path ${src_emb} \
    --tgt-embedding-path ${tgt_emb} \
    --vocab ${vocab} \
    --train-src ${train_src} \
    --train-tgt ${train_tgt} \
    --dev-src ${dev_src} \
    --dev-tgt ${dev_tgt} \
    --save-to ${work_dir}/model.bin \
    --valid-niter 1418 \
    --batch-size 64 \
    --hidden-size 512 \
    --embed-size 300 \
    --uniform-init 0.1 \
    --dropout 0.2 \
    --clip-grad 5.0 \
    --lr-decay 0.5 2>${work_dir}/err.log

python nmt.py \
    decode \
    --seed 233 \
    --beam-size 5 \
    --max-decoding-time-step 100 \
    ${vocab} \
    ${work_dir}/model.bin \
    ${test_src} \
    ${work_dir}/decode.txt

perl multi-bleu.perl ${test_tgt} < ${work_dir}/decode.txt