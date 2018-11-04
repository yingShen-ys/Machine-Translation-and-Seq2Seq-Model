lan="tr"
$HOME/mosesdecoder/scripts/training/mert-moses.pl \
$HOME/data/processed_para/dev.en-${lan}.en $HOME/data/processed_para/dev.en-${lan}.${lan} \
$HOME/mosesdecoder/bin/moses $HOME/working/train-${lan}/model/moses.ini --mertdir $HOME/mosesdecoder/bin/ \
--rootdir $HOME/mosesdecoder/scripts \
--working-dir=$HOME/working/mert-work-${lan} \
--decoder-flags="-threads 4"
