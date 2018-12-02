deIndex = 'de'
enIndex = 'en'
count = 0
deOutput = []
enOutput = []

with open('xnli.15way.tok.txt', 'r') as f:
    for line in f:
        line = line.split('\t')
        if count == 0:
            deIndex = line.index(deIndex)
            enIndex = line.index(enIndex)
            count += 1
        else:
            deSentece = line[deIndex]
            deOutput.append(deSentece)

            enSentece = line[enIndex]
            enOutput.append(enSentece)

with open('xnli_raw.all.de', 'w') as f:
    for sentence in deOutput:
        f.write(sentence + '\n')

with open('xnli_raw.all.en', 'w') as f:
    for sentence in enOutput:
        f.write(sentence + '\n')
