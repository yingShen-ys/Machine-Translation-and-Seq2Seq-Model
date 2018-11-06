#!/bin/sh

source=$1
auxiliary=$2



# creating folders
echo "running experiments for language $1 and $2"
work_dir="experiments/work_dir_concat_bpe_ultimate_dec_${source}_${auxiliary}"
mkdir -p ${work_dir}

# load original and auxiliary source embeddings
original_src_emb="embeddings/vectors-${source}.txt"
auxiliary_src_emb="embeddings/vectors-${auxiliary}.txt"

# load train/dev source data for original and auxiliary language
train_original_src="data/train.en-${source}.${source}.txt"
train_auxiliary_src="data/train.en-${auxiliary}.${auxiliary}.txt"
dev_original_src="data/dev.en-${source}.${source}.txt"
dev_auxiliary_src="data/dev.en-${auxiliary}.${auxiliary}.txt"

# load train/dev target data for original and auxiliary language
train_original_tgt="data/train.en-${source}.en.txt"
train_auxiliary_tgt="data/train.en-${auxiliary}.en.txt"
dev_original_tgt="data/dev.en-${source}.en.txt"
dev_auxiliary_tgt="data/dev.en-${auxiliary}.en.txt"

# load test data, only test on original source language
original_test_src="data/test.en-$source.$source.txt"
test_src="${work_dir}/test.${source}${auxiliary}-en.${source}${auxiliary}.txt"
test_tgt="data/test.en-$source.en.txt"

# designate placeholders for preprocessed data
concat_dev_src="${work_dir}/concat.dev.${source}${auxiliary}-en.${source}${auxiliary}.txt"
concat_train_src="${work_dir}/concat.train.${source}${auxiliary}-en.${source}${auxiliary}.txt"

vocab="${work_dir}/${source}${auxiliary}-en.bin"
dev_src="${work_dir}/dev.${source}${auxiliary}-en.${source}${auxiliary}.txt"
train_src="${work_dir}/train.${source}${auxiliary}-en.${source}${auxiliary}.txt"
concat_dev_tgt="${work_dir}/concat.dev.${source}${auxiliary}-en.en.txt"
concat_train_tgt="${work_dir}/concat.train.${source}${auxiliary}-en.en.txt"
dev_tgt="${work_dir}/dev.${source}${auxiliary}-en.en.txt"
train_tgt="${work_dir}/train.${source}${auxiliary}-en.en.txt"

# echo $train_src
# echo $dev_src
# echo $train_tgt
# echo $dev_tgt

if [[ ! -e ${train_src} ]] && [[ ! -e ${dev_src} ]] && [[ ! -e ${train_tgt} ]] && [[ ! -e ${dev_tgt} ]]; then
    # simple concatenation of data
    cat $train_original_src $train_auxiliary_src > $concat_train_src
    cat $dev_original_src $dev_auxiliary_src > $concat_dev_src
    cat $train_original_tgt $train_auxiliary_tgt > $concat_train_tgt
    cat $dev_original_tgt $dev_auxiliary_tgt > $concat_dev_tgt
    # run BPE for src and tgt data
    if [[ ! -e "bpe_models/${source}${auxiliary}en.model" ]]; then
        python bpe.py train \
            --input ${concat_train_src} \
            --character-coverage 1.0 \
            --model-prefix ${source}${auxiliary}en \
            --model-type bpe \
            --vocab-size 15000
        mv "${source}${auxiliary}en.model" "bpe_models/"
        mv "${source}${auxiliary}en.vocab" "bpe_models/"
    fi

    # if [[ ! -e "bpe_models/${source}${auxiliary}-en.model" ]]; then
    #     python bpe.py train \
    #         --input ${concat_train_tgt} \
    #         --character-coverage 1.0 \
    #         --model-prefix ${source}${auxiliary}-en \
    #         --model-type bpe \
    #         --vocab-size 10000
    #     mv "${source}${auxiliary}-en.model" "bpe_models/"
    #     mv "${source}${auxiliary}-en.vocab" "bpe_models/"
    # fi
    # run BPE models on src and target
    python bpe.py encode --model "bpe_models/${source}${auxiliary}en.model" < ${concat_train_src} > ${train_src}
    python bpe.py encode --model "bpe_models/${source}${auxiliary}en.model" < ${concat_dev_src} > ${dev_src}
    python bpe.py encode --model "bpe_models/${source}${auxiliary}en.model" < ${original_test_src} > ${test_src}
    python bpe.py encode --model "bpe_models/${source}${auxiliary}en.model" < ${concat_train_tgt} > ${train_tgt}
    python bpe.py encode --model "bpe_models/${source}${auxiliary}en.model" < ${concat_dev_tgt} > ${dev_tgt}
    python vocab.py --train-src $train_src --train-tgt $train_tgt $vocab
else
    echo "Preprocessed files already exists"
fi

echo save results to ${work_dir}

python nmt.py \
    train \
    --model-type lstm \
    --cuda \
    --seed 233 \
    --vocab ${vocab} \
    --decoder-layers 2 \
    --num-mixtures 3 \
    --train-src ${train_src} \
    --train-tgt ${train_tgt} \
    --dev-src ${dev_src} \
    --dev-tgt ${dev_tgt} \
    --save-to ${work_dir}/model.bin \
    --valid-niter 5718 \
    --batch-size 35 \
    --hidden-size 256 \
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
    ${work_dir}/decode_bpe.txt

# run BPE decode
python bpe.py \
    decode \
    --model "bpe_models/${source}${auxiliary}-en.model" < ${work_dir}/decode_bpe.txt > ${work_dir}/decode.txt

perl multi-bleu.perl ${test_tgt} < ${work_dir}/decode.txt > ${work_dir}/eval.log
cat ${work_dir}/eval.log
