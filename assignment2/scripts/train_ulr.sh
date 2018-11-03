#!/bin/sh

source=$1
auxiliary=$2

echo "running experiments for language $1 and $2"

work_dir="experiments/work_dir_ulr_${source}_${auxiliary}"
mkdir -p ${work_dir}

original_src_emb="embeddings/prefix-vector-${source}.txt"
auxiliary_src_emb="embeddings/prefix-vector-${auxiliary}.txt"

original_src_mono_emb="embeddings/prefix-${source}-mono.txt"
auxiliary_src_mono_emb="embeddings/prefix-${auxiliary}-mono.txt"

train_original_src="prefixed_data/prefixed-train.en-${source}.${source}.txt"
train_original_tgt="data/train.en-${source}.en.txt"
train_auxiliary_src="prefixed_data/prefixed-train.en-${auxiliary}.${auxiliary}.txt"
train_auxiliary_tgt="data/train.en-${auxiliary}.en.txt"

dev_original_src="prefixed_data/prefixed-dev.en-${source}.${source}.txt"
dev_original_tgt="data/dev.en-${source}.en.txt"
dev_auxiliary_src="prefixed_data/prefixed-dev.en-${auxiliary}.${auxiliary}.txt"
dev_auxiliary_tgt="data/dev.en-${auxiliary}.en.txt"

vocab="${work_dir}/${source}${auxiliary}-en.bin"
original_src_vocab="${work_dir}/${source}-en.bin"
auxiliary_src_vocab="${work_dir}/${auxiliary}-en.bin"

dev_src="${work_dir}/dev.${source}${auxiliary}-en.${source}${auxiliary}.txt"
dev_tgt="${work_dir}/dev.${source}${auxiliary}-en.en.txt"
test_src="prefixed_data/prefixed-test.en-$source.$source.txt"
test_tgt="data/test.en-$source.en.txt"

train_src="${work_dir}/train.${source}${auxiliary}-en.${source}${auxiliary}.txt"
train_tgt="${work_dir}/train.${source}${auxiliary}-en.en.txt"

tgt_emb="embeddings/vectors-en.txt"
src_emb="${work_dir}/vector-${source}${auxiliary}.txt"
src_mono_emb="${work_dir}/mono-${source}${auxiliary}.txt"

cat $train_original_src $train_auxiliary_src > $train_src
cat $dev_original_src $dev_auxiliary_src > $dev_src
cat $train_original_tgt $train_auxiliary_tgt > $train_tgt
cat $dev_original_tgt $dev_auxiliary_tgt > $dev_tgt
cat $original_src_emb $auxiliary_src_emb > $src_emb
cat $original_src_mono_emb $auxiliary_src_mono_emb > $src_mono_emb

python vocab.py --train-src $train_src --train-tgt $train_tgt $vocab
python vocab.py --train-src $train_original_src --train-tgt $train_original_tgt $original_src_vocab
python vocab.py --train-src $train_auxiliary_src --train-tgt $train_auxiliary_tgt $auxiliary_src_vocab

echo save results to ${work_dir}

python nmt.py \
    train \
    --model-type ULR \
    --cuda \
    --seed 233 \
    --vocab ${vocab} \
    --src-embedding-path ${src_emb} \
    --tgt-embedding-path ${tgt_emb} \
    --src-mono-embedding-path ${src_mono_emb} \
    --original-src-vocab ${original_src_vocab} \
    --auxiliary-src-vocab ${auxiliary_src_vocab} \
    --train-src ${train_src} \
    --train-tgt ${train_tgt} \
    --language-code ${source}-${auxiliary} \
    --top-tokens 500 \
    --dev-src ${dev_src} \
    --dev-tgt ${dev_tgt} \
    --save-to ${work_dir}/model.bin \
    --valid-niter 3127 \
    --batch-size 29 \
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
    --max-decoding-time-step 70 \
    ${vocab} \
    ${work_dir}/model.bin \
    ${test_src} \
    ${work_dir}/decode.txt 2>${work_dir}/eval.log

perl multi-bleu.perl ${test_tgt} < ${work_dir}/decode.txt >> ${work_dir}/eval.log
cat ${work_dir}/eval.log
