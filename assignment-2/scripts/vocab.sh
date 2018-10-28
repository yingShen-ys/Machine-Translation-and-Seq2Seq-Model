#!/bin/sh

lan="en"
model_file=""
out_file="data/merged-${lan}.txt"

for file in data/*.${lan}.txt
do
    echo "Appending $file"
    cat $file >> ${out_file}
done


#./fasttext print-word-vectors ${model_file} < ${file}