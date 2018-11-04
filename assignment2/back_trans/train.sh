lan="tr"
~/mosesdecoder/scripts/training/train-model.perl -root-dir train-${lan} \
-cores 4 \
-corpus ~/data/processed_para/train.en-${lan}.clean \
-f en -e ${lan} -alignment grow-diag-final-and -reordering msd-bidirectional-fe \
-lm 0:3:$HOME/lm/wiki.${lan}.bin:8                          \
-external-bin-dir ~/mosesdecoder/tools
