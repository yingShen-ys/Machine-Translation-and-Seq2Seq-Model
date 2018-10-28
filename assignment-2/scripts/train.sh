#!/bin/sh

source="az"
auxiliary="tr"

vocab="data/vocab_${source}_$auxiliary.bin"
train_original_src="data/train.en-$source.$source.txt"
train_original_tgt="data/train.en-$source.en.txt"
train_auxiliary_src="data/train.en-$auxiliary.$auxiliary.txt"
train_auxiliary_tgt="data/train.en-$auxiliary.en.txt"
dev_src="data/dev.en-$source.$source.txt"
dev_tgt="data/dev.en-$source.en.txt"
test_src="data/test.en-$source.$source.txt"
test_tgt="data/test.en-$source.en.txt"

train_src="data/train.${source}_${auxiliary}.txt"
train_tgt="data/train.${source}_${auxiliary}_en.txt"

if [ ! -f $vocab ]; then
    if [ ! -f $train_src ]; then
        python preprocess_file.py --file1=${train_original_src} --file2=${train_auxiliary_src} --output-file=${train_src}
        python preprocess_file.py --file1=${train_original_tgt} --file2=${train_auxiliary_tgt} --output-file=${train_tgt}
        echo "create concatenated training file ${train_src} ${train_tgt}"
    fi
    python vocab.py --train-src=${train_src} --train-tgt=${train_tgt} ${vocab}
    echo "save vocab file to ${vocab}"
fi

work_dir="work_dir_basic_concatenate_${source}_${auxiliary}"

mkdir -p ${work_dir}
echo save results to ${work_dir}

python nmt_mk_iii.py \
    train \
    --model-type original_lstm \
    --cuda \
    --seed 233 \
    --vocab ${vocab} \
    --train-src ${train_src} \
    --train-tgt ${train_tgt} \
    --dev-src ${dev_src} \
    --dev-tgt ${dev_tgt} \
    --save-to ${work_dir}/model.bin \
    --valid-niter 2400 \
    --batch-size 64 \
    --hidden-size 256 \
    --embed-size 256 \
    --uniform-init 0.1 \
    --dropout 0.2 \
    --clip-grad 5.0 \
    --lr-decay 0.5 2>${work_dir}/err.log

python nmt_mk_iii.py \
    decode \
    --seed 233 \
    --beam-size 5 \
    --max-decoding-time-step 100 \
    ${vocab} \
    ${work_dir}/model.bin \
    ${test_src} \
    ${work_dir}/decode.txt

perl multi-bleu.perl ${test_tgt} < ${work_dir}/decode.txt