deIndex = 'de'
enIndex = 'en'
count = 0
deCorpus = []
enCorpus = []

# with open('xnli.15way.orig.tsv', 'r') as f:
#     for line in f:
#         line = line.split('\t')
#         if count == 0:
#             deIndex = line.index(deIndex)
#             enIndex = line.index(enIndex)
#             count += 1
#         else:
#             deSentece = line[deIndex]
#             deCorpus.append(deSentece)

#             enSentece = line[enIndex]
#             enCorpus.append(enSentece)

enTranslatedCorpus = []
with open('xnli_raw.all.de', 'r') as src_file, open('xnli_raw.all.en', 'r') as trg_file, \
    open('xnli_raw.all.en.translated', 'r') as translated_trg_file:
    for src_line, tgt_line, translated_trg_line in zip(src_file, trg_file, translated_trg_file):
        deCorpus.append(src_line.rstrip('\n'))
        enCorpus.append(tgt_line.rstrip('\n'))
        enTranslatedCorpus.append(translated_trg_line.rstrip('\n'))

deEncoded = {s:i for i, s in enumerate(deCorpus)}

import json

output = []
with open('snli_1.0_test_de.jsonl', 'r') as f:
    for line in f:
        line = line.rstrip('\n')
        line = json.loads(line)
        deSentence1Index = deEncoded[line['sentence1_tokenized'].encode().decode('utf-8', 'ignore')]
        line['sentence1'] = enCorpus[deSentence1Index]

        deSentence2Index = deEncoded[line['sentence2_tokenized'].encode().decode('utf-8', 'ignore')]
        line['sentence2'] = enTranslatedCorpus[deSentence2Index]
        output.append(json.dumps(line))

with open('snli_1.0_test.jsonl', 'w') as f:
    for line in output:
        f.write(line + '\n')
