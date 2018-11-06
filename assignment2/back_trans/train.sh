src=$1
aux=$2
~/mosesdecoder/scripts/training/train-model.perl -root-dir train-${src}${aux}-en \
-cores 4 \
-corpus ~/data/processed_para/train.en-${src}${aux}.clean \
-f ${src}${aux} -e en -alignment grow-diag-final-and -reordering msd-bidirectional-fe \
-lm 0:3:$HOME/lm/ted.en.bin:8                          \
-external-bin-dir ~/mosesdecoder/tools
