import os

deCorpus = []
enCorpus = []

with open('xnli_raw.all.de', 'r') as f:
    for line in f:
        deCorpus.append(line.rstrip('\n'))

with open('xnli_raw.all.en', 'r') as f:
    for line in f:
        enCorpus.append(line.rstrip('\n'))

deEncoded = {s:i for i, s in enumerate(deCorpus)}

def prepareData(files, outputPrefix, splitRatio = 0.9, skipLabels = set()):
    deOutput = []
    enOutput = []
    premises = []
    hypothesis = []
    isPremise = []
    labels = []

    def readXNLIFile(file):
        count = 0
        s1TokenizedIndex = 'sentence1_tokenized'
        s2TokenizedIndex = 'sentence2_tokenized'
        labelIndex = 'gold_label'
        with open(file, 'r') as f:
            for line in f:
                line = line.split('\t')
                if count == 0:
                    s1TokenizedIndex = line.index(s1TokenizedIndex)
                    s2TokenizedIndex = line.index(s2TokenizedIndex)
                    labelIndex = line.index(labelIndex)
                    print(s1TokenizedIndex, s2TokenizedIndex, labelIndex)
                    count += 1
                else:
                    if line[0] != 'de':
                        continue

                    label = line[labelIndex]
                    if label in skipLabels:
                        continue 

                    s1 = line[s1TokenizedIndex]
                    s2 = line[s2TokenizedIndex]

                    enIndex1 = deEncoded[s1]
                    enIndex2 = deEncoded[s2]

                    deOutput.append(s1)
                    enOutput.append(enCorpus[enIndex1])
                    premises.append(enCorpus[enIndex1])
                    hypothesis.append(enCorpus[enIndex2])
                    isPremise.append(str(1))
                    labels.append(label)

                    deOutput.append(s2)
                    enOutput.append(enCorpus[enIndex2])
                    premises.append(enCorpus[enIndex1])
                    hypothesis.append(enCorpus[enIndex2])
                    isPremise.append(str(0))
                    labels.append(label)
    
    def createTrainValidationSplit(ratio=0.9):
        size = len(labels)
        train_size = int(ratio * size)
        print(size, train_size)

        import numpy as np
        indices = np.arange(size)
        trainIndices = np.random.choice(size, train_size, replace=False)
        mask = np.zeros(size, dtype=bool)
        mask[trainIndices] = True

        return indices[mask], indices[~mask]

    def output(name, outputPrefix, indices, skipLabelTag):
        if not os.path.exists(outputPrefix):
            os.mkdir(outputPrefix)

        with open(os.path.join(outputPrefix, 'nli.' + name + skipLabelTag + '.de'), 'w') as f:
            for i in indices:
                f.write(deOutput[i] + '\n')
                
        with open(os.path.join(outputPrefix, 'nli.' + name + skipLabelTag + '.en'), 'w') as f:
            for i in indices:
                f.write(enOutput[i] + '\n')
                
        with open(os.path.join(outputPrefix, 'nli.' + name + skipLabelTag + '.premises'), 'w') as f:
            for i in indices:
                f.write(premises[i] + '\n')

        with open(os.path.join(outputPrefix, 'nli.' + name + skipLabelTag + '.hypothesis'), 'w') as f:
            for i in indices:
                f.write(hypothesis[i] + '\n')

        with open(os.path.join(outputPrefix, 'nli.' + name + skipLabelTag + '.is_premise'), 'w') as f:
            for i in indices:
                f.write(isPremise[i] + '\n')

        with open(os.path.join(outputPrefix, 'nli.' + name + skipLabelTag + '.label'), 'w') as f:
            for i in indices:
                f.write(labels[i] + '\n')

    for file in files:
        readXNLIFile(file)

    skipLabelTag = '_'.join([s for s in skipLabels])
    if skipLabelTag:
        skipLabelTag = '.no_' + skipLabelTag

    if int(splitRatio) != 1:
        trainIndices, testIndices = createTrainValidationSplit(splitRatio)
        output('train', outputPrefix, trainIndices, skipLabelTag)
        output('valid', outputPrefix, testIndices, skipLabelTag)
    else:
        output('all', outputPrefix, [i for i in range(len(labels))], skipLabelTag)

files = ['xnli.dev.tsv', 'xnli.test.tsv']
prepareData(files, 'nli_split', splitRatio = 0.9, skipLabels = set())
prepareData(files, 'nli_all', splitRatio = 1, skipLabels = set())
prepareData(files, 'nli_all_skip_neutral', splitRatio = 1, skipLabels = set(['neutral']))