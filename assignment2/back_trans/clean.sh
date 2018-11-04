#!/bin/sh
lan="pt"
# truecaser & limit length
# para
#~/mosesdecoder/scripts/recaser/train-truecaser.perl --model ~/data/truecase-model.${lan} --corpus ~/data/parallel/dev.en-${lan}.${lan}.txt 
#~/mosesdecoder/scripts/recaser/truecase.perl --model ~/data/truecase-model.${lan} < ~/data/parallel/dev.en-${lan}.${lan}.txt > ~/data/processed_para/dev.en-${lan}.${lan}
#~/mosesdecoder/scripts/recaser/train-truecaser.perl --model ~/data/truecase-model.${lan}.en --corpus ~/data/parallel/dev.en-${lan}.en.txt
#~/mosesdecoder/scripts/recaser/truecase.perl --model ~/data/truecase-model.${lan}.en < ~/data/parallel/dev.en-${lan}.en.txt > ~/data/processed_para/dev.en-${lan}.en
#~/mosesdecoder/scripts/training/clean-corpus-n.perl ~/data/processed_para/dev.en-${lan} en ${lan} ~/data/processed_para/dev.en-${lan}.clean 1 80
# mono
~/mosesdecoder/scripts/recaser/train-truecaser.perl --model ~/data/truecase-model.mono.${lan} --corpus ~/${lan}.wiki.txt 
~/mosesdecoder/scripts/recaser/truecase.perl --model ~/data/truecase-model.mono.${lan} < ~/${lan}.wiki.txt > ~/data/wikis/${lan}.wiki.true
#~/mosesdecoder/scripts/recaser/train-truecaser.perl --model ~/data/truecase-model.${lan}.en --corpus ~/data/parallel/train.en-${lan}.en.txt
