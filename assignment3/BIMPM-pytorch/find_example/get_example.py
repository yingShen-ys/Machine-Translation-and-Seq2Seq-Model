import json
import pdb

def read_xnli_file(filepath):
    xnli = []
    with open(filepath, 'r') as f:
        for line in f:
            xnli.append(json.loads(line.rstrip('\n')))
    
    return xnli

def read_ground_truth_files(de_ground_truth_file, en_ground_truth_file):
    deCorpus = []
    enCorpus = []
    enTranslatedCorpus = []

    with open(de_ground_truth_file, 'r') as src_file, open(en_ground_truth_file, 'r') as trg_file:
        for src_line, tgt_line in zip(src_file, trg_file):
            deCorpus.append(src_line.rstrip('\n'))
            enCorpus.append(tgt_line.rstrip('\n'))

    deEncoded = {s:i for i, s in enumerate(deCorpus)}
    return deEncoded, enCorpus

def read_translated_files(filepath):
    result = []

    with open(filepath, 'r') as f:
        for line in f:
            result.append(line.rstrip('\n'))

    return result

def read_error_file(filepath):
        indices = []
        with open(filepath, 'r') as f:
            for line in f:
                indices.append(int(line.rstrip('\n')))
        
        return indices

def find_wrong_example(
        pretrain_err_filepath, \
        mrt_bleu_err_filepath, mrt_nli_err_filepath, mrt_combined_err_filepath, \
        de_ground_truth_filepath, en_ground_truth_filepath, \
        pretrain_translation_filepath, \
        mrt_bleu_translation_filepath, mrt_nli_translation_filepath, mrt_combined_translation_filepath, \
        xnli_path):

    pretrain_error_indices = read_error_file(pretrain_err_filepath)
    mrt_bleu_error_indices = set(read_error_file(mrt_bleu_err_filepath))
    mrt_nli_error_indices = set(read_error_file(mrt_nli_err_filepath))
    mrt_combined_error_indices = set(read_error_file(mrt_combined_err_filepath))

    pretrain_translations = read_translated_files(pretrain_translation_filepath)
    mrt_bleu_translations = read_translated_files(mrt_bleu_translation_filepath)
    mrt_nli_translations = read_translated_files(mrt_nli_translation_filepath)
    mrt_combined_translations = read_translated_files(mrt_combined_translation_filepath)

    deSentenceToIndices, enSenteces = read_ground_truth_files(de_ground_truth_filepath, en_ground_truth_filepath)
    xnli = read_xnli_file(xnli_path)

    for i in range(len(pretrain_error_indices)):
        index = pretrain_error_indices[i]
        if index in mrt_nli_error_indices:
            continue

        xnli_example = xnli[index]
        deSentence1Index = deSentenceToIndices[xnli_example['sentence1_tokenized'].encode().decode('utf-8', 'ignore')]
        deSentence2Index = deSentenceToIndices[xnli_example['sentence2_tokenized'].encode().decode('utf-8', 'ignore')]

        hypothesis_index = deSentence2Index
        premise_ground_truth = enSenteces[deSentence1Index]
        hypothesis_ground_truth = enSenteces[hypothesis_index]
        label = xnli_example['gold_label']
        genre = xnli_example['genre']
        pretrain_translation = pretrain_translations[hypothesis_index]
        mrt_bleu_translation = mrt_bleu_translations[hypothesis_index]
        mrt_nli_translation = mrt_nli_translations[hypothesis_index]
        mrt_combined_translation = mrt_combined_translations[hypothesis_index]
        print("index:", index)
        print("premise_ground_truth:", premise_ground_truth)
        print("hypothesis_ground_truth:", hypothesis_ground_truth)
        print("ground truth label:", label)
        print("genre label:", genre)
        print("pretrain_translation:", pretrain_translation)
        print("mrt_bleu_translation:", mrt_bleu_translation)
        print("mrt_nli_translation:", mrt_nli_translation)
        print("mrt_combined_translation:", mrt_combined_translation)
        print()
    
if __name__ == "__main__":
    find_wrong_example('error_indices_before_tune.txt',\
        'error_indices_after_bleu_tune.txt', 'error_indices_after_nli_tune.txt', 'error_indices_after_combined_tune.txt', \
        'xnli_raw.all.de', 'xnli_raw.all.en', \
        'nli.all.before_finetune.en.translated', \
        'nli.all.with_neutral.after_bleu_finetune.en.translated', 'nli.all.with_neutral.after_nli_finetune.en.translated', \
        'nli.all.with_neutral.after_combined_finetune.en.translated', \
        'multinli_1.0_dev_mismatched.raw.jsonl')