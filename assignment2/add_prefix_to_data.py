import os

def add_prefix_to_sent(sent, prefix):
    return prefix + sent.replace(' ', ' ' + prefix)

def add_prefix_to_file(file_path, output_path, prefix):
    with open(file_path, 'r') as f, open(output_path, 'w') as fout:
        for line in f:
            prefixed_line = add_prefix_to_sent(line, prefix)
            fout.write(prefixed_line)

if __name__ == "__main__":
    data_files = os.listdir('data')
    for data_file in data_files:
        lang_code = data_file.split('.')[-2]
        if lang_code == 'en':
            continue
        prefix = lang_code + '_'
        output_path = os.path.join('data', 'prefixed-' + data_file)
        input_path = os.path.join('data', data_file)
        add_prefix_to_file(input_path, output_path, prefix)
