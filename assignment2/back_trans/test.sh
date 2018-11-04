#-f $HOME/working/mert-work-${lan}/moses.ini   \
lan="az"
$HOME/mosesdecoder/bin/moses            \
-f $HOME/working/train-${lan}/model/moses.ini   \
< $HOME/data/parallel/test.en-${lan}.en.txt                \
> $HOME/working/test.pre.en-${lan}.${lan}.txt         \
2> $HOME/working/test.pre.en-${lan}.out
$HOME/mosesdecoder/scripts/generic/multi-bleu.perl \
-lc $HOME/data/parallel/test.en-${lan}.${lan}.txt              \
< $HOME/working/test.pre.en-${lan}.${lan}.txt
