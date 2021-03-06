"""
extract training, dev, test data for en
"""

import io
import csv
import sys
reload(sys)
sys.setdefaultencoding('utf8')

ftrain = io.open('all_talks_train.tsv','r',encoding='utf-8')
fdev = io.open('all_talks_dev.tsv','r',encoding='utf-8')
ftest = io.open('all_talks_test.tsv','r',encoding='utf-8')

def get_en_wo_ln(ln):
    def get_data(csv_f):
        data = []
        reader = csv.DictReader(csv_f, delimiter='\t')
        for row in reader:
            en, fr = row['en'].strip(), row[ln].strip()
            fr = fr.replace("__NULL__","").replace('_ _ NULL _ _','').strip()
            if len(fr) != 0: # remove en sentence that has existing paired language ln
                continue
            data.append((en,fr))
        csv_f.seek(0)
        return data
    tr, de, ts = get_data(ftrain), get_data(fdev), get_data(ftest)
    
    def write_data(data, fname1, fname2):
        f1 = io.open(fname1,'w',encoding='utf-8')
        f2 = io.open(fname2,'w',encoding='utf-8')
        print len(data)
        for i in data:
            f1.write(unicode(i[0])+"\n")
            f2.write(unicode(i[1])+"\n")
        f1.close()
        f2.close()

    write_data(tr, "train.{0}.{1}.txt".format("en-{0}".format(ln), "en"), "train.{0}.{1}.txt".format("en-{0}".format(ln), ln))
    write_data(de, "dev.{0}.{1}.txt".format("en-{0}".format(ln), "en"), "dev.{0}.{1}.txt".format("en-{0}".format(ln), ln))
    write_data(ts, "test.{0}.{1}.txt".format("en-{0}".format(ln), "en"), "test.{0}.{1}.txt".format("en-{0}".format(ln), ln))

get_en_wo_ln("gl")
get_en_wo_ln("az")
get_en_wo_ln("be")

get_en_wo_ln("pt")
get_en_wo_ln("tr")
get_en_wo_ln("ru")
