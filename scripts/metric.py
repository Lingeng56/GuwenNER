import argparse
from sklearn.metrics import f1_score



def load_pred(path):
    pred = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line != '':
                pred.append(line)
    return pred



def load_true(path):
    true = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line != '':
                _, line = line.split(' ')
                true.append(line)
    return true




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--true', type=str, required=True)
    parser.add_argument('--pred', type=str, required=True)
    args = parser.parse_args()
    true = load_true(args.true)
    pred = load_pred(args.pred)
    f1 = f1_score(true, pred, average='micro')
    print('F1 Score: ', f1)
