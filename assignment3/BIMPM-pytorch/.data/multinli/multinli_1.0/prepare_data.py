import json

deIndex = 'de'
enIndex = 'en'
count = 0
deCorpus = []
enCorpus = []
enTranslatedCorpus = []

with open('xnli_raw.all.de', 'r') as src_file, open('xnli_raw.all.en', 'r') as trg_file, \
    open('nli.all.with_neutral.after_combined_finetune.en.translated', 'r') as translated_trg_file:
    for src_line, tgt_line, translated_trg_line in zip(src_file, trg_file, translated_trg_file):
        deCorpus.append(src_line.rstrip('\n'))
        enCorpus.append(tgt_line.rstrip('\n'))
        enTranslatedCorpus.append(translated_trg_line.rstrip('\n'))

deEncoded = {s:i for i, s in enumerate(deCorpus)}

def readValidationSet(premiseFile, hypothesisFile):
    result = set()
    with open(premiseFile, 'r') as premise, open(hypothesisFile, 'r') as hypothesis:
        for premise_line, hypothesis_line in zip(premise, hypothesis):
            premise_line, hypothesis_line = premise_line.rstrip('\n'), hypothesis_line.rstrip('\n')
            result.add((premise_line, hypothesis_line))
    
    return result

def readXNLIFile(files, premise_hypothesis_set):
    s1TokenizedIndex = 'sentence1_tokenized'
    s2TokenizedIndex = 'sentence2_tokenized'
    labelIndex = 'gold_label'
    language = 'language'
    result = []
    for file in files:
        with open(file, 'r') as f:
            for line in f:
                line = json.loads(line.rstrip('\n'))
                if line[language] != 'de':
                    continue

                s1 = line[s1TokenizedIndex]
                s2 = line[s2TokenizedIndex]

                premise = enCorpus[deEncoded[s1]]
                hypothesis = enCorpus[deEncoded[s2]]
                if (premise, hypothesis) in premise_hypothesis_set:
                    result.append(json.dumps(line))
    
    with open('multinli_1.0_dev_mismatched.raw.jsonl', 'w') as f:
        for r in result:
            f.write(r + '\n')

validationSet = readValidationSet('nli.valid.premises', 'nli.valid.hypothesis')
readXNLIFile(['xnli.dev.jsonl', 'xnli.test.jsonl'], validationSet)

output = []
with open('multinli_1.0_dev_mismatched.raw.jsonl', 'r') as f:
    for line in f:
        line = json.loads(line.rstrip('\n'))
        deSentence1Index = deEncoded[line['sentence1_tokenized'].encode().decode('utf-8', 'ignore')]
        line['sentence1'] = enCorpus[deSentence1Index]

        deSentence2Index = deEncoded[line['sentence2_tokenized'].encode().decode('utf-8', 'ignore')]
        line['sentence2'] = enTranslatedCorpus[deSentence2Index]
        output.append(json.dumps(line))

with open('multinli_1.0_dev_mismatched.jsonl', 'w') as f:
    for line in output:
        f.write(line + '\n')
