src=$1
aux=$2
$HOME/mosesdecoder/bin/moses            \
-f $HOME/working/mert-work-${src}${aux}-en/moses.ini   \
< $HOME/data/parallel/test.en-${src}.${src}.txt                \
> $HOME/working/test.pre.${src}${aux}-en.en.txt         \
2> $HOME/working/test.pre.${src}${aux}-en.out
$HOME/mosesdecoder/scripts/generic/multi-bleu.perl \
-lc $HOME/data/parallel/test.en-${src}.en.txt              \
< $HOME/working/test.pre.${src}${aux}-en.en.txt
