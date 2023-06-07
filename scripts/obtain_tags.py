import argparse






def generate_tags(document_path, tag_path):
    tags = set()
    with open(document_path) as f:
        text = f.read()
        sentence_units = text.split('\n\n')
        for unit in sentence_units:
            unit = unit.split('\n')
            for item in unit:
                if item == '':
                    continue
                word, tag = item.split(' ')
                tags.add(tag)

    with open(tag_path, 'w')as f:
        for tag in tags:
            f.write(tag + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Extract tags'
    )
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--tag_path', type=str, required=True)
    args = parser.parse_args()
    generate_tags(args.data_path, args.tag_path)