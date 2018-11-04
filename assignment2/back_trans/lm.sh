#!/bin/sh
lan="pt"
~/mosesdecoder/bin/lmplz -o 3 <~/data/wikis/${lan}.wiki.true > wiki.${lan}
~/mosesdecoder/bin/build_binary wiki.${lan} wiki.${lan}.bin
