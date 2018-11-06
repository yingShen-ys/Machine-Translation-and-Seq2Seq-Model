src=$1
aux=$2
$HOME/mosesdecoder/scripts/training/mert-moses.pl \
$HOME/data/processed_para/dev.en-${src}${aux}.clean.${src}${aux} $HOME/data/processed_para/dev.en-${src}${aux}.clean.en \
$HOME/mosesdecoder/bin/moses $HOME/working/train-${src}${aux}-en/model/moses.ini --mertdir $HOME/mosesdecoder/bin/ \
--rootdir $HOME/mosesdecoder/scripts \
--working-dir=$HOME/working/mert-work-${src}${aux}-en \
--decoder-flags="-threads 4"
