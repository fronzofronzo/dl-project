import json
import torch
from pathlib import Path
from torchvision.datasets import CelebA

def load_data(data_dir, split, download=False):
    dataset = CelebA(root=data_dir, split=split, download=download)
    return dataset

def wire_attributes(dataset):
    attr_names = dataset.attr_names
    attr_dict = {name: idx for idx, name in enumerate(attr_names) if name}
    return attr_dict

def parse_query(query, attr_dict):
    """'+Eyeglasses, +Smiling' -> (pos_idx, neg_idx) liste di indici colonna."""
    pos, neg = [], []
    for tok in query.split(','):
        tok = tok.strip()
        if not tok:
            continue
        sign, name = tok[0], tok[1:]
        if name not in attr_dict:
            raise KeyError(f"Attributo sconosciuto: {name!r} in {query!r}")
        if sign == '+':
            pos.append(attr_dict[name])
        elif sign == '-':
            neg.append(attr_dict[name])
        else:
            raise ValueError(f"Segno mancante/non valido: {tok!r}")
    return pos, neg

def verify_eval_indices(dataset, eval_json):
    """Verifica che reference e target del JSON siano indici validi nel dataset.

    Reference = chiavi di ground_truth, target = valori.
    Ritorna dict con conteggi e indici fuori range."""
    with open(eval_json) as f:
        data = json.load(f)

    ref_idx, tgt_idx = set(), set()
    for entry in data:
        for k, v in entry['ground_truth'].items():
            ref_idx.add(int(k))
            tgt_idx.update(v)

    n = len(dataset)
    bad_ref = sorted(i for i in ref_idx if i < 0 or i >= n)
    bad_tgt = sorted(i for i in tgt_idx if i < 0 or i >= n)
    ok = not bad_ref and not bad_tgt
    return {
        'dataset_len': n,
        'num_queries': len(data),
        'num_ref': len(ref_idx),
        'num_tgt': len(tgt_idx),
        'bad_ref': bad_ref,
        'bad_tgt': bad_tgt,
        'ok': ok,
    }

if __name__ == "__main__":
    data_dir = Path('data')
    celeba = load_data(data_dir, split='train', download=False)
    print(f"Number of samples in the dataset: {len(celeba)}")

    img, attrs = celeba[0]
    # img.show()

    attr_dict = wire_attributes(celeba)
    print(f"Attribute names: {list(attr_dict.keys())}")
    print(f"Attribute for the first sample: {attrs}")

    pos, neg = parse_query("+Eyeglasses, +Smiling", attr_dict)
    print(f"pos={pos} neg={neg}")

    celeba_test = load_data(data_dir, split='test', download=False)
    report = verify_eval_indices(celeba_test, 'data/celeba_evaluation.json')
    print(f"Verifica indici eval (split=test): {report}")