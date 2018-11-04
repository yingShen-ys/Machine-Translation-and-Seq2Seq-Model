#-f $HOME/working/mert-work-${lan}/moses.ini   \
lan="az"
$HOME/mosesdecoder/bin/moses            \
-f $HOME/working/train-${lan}/model/moses.ini   \
< $HOME/data/ted_en/dev.en-${lan}.en.txt                \
> $HOME/data/back-trans/dev.trans.en-${lan}.${lan}.txt         
