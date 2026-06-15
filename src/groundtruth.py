import json
from dataclasses import dataclass, field


@dataclass
class QueryGT:
    """One evaluation query and its ground-truth.

    pos / neg are attribute *column indices* if an attr_dict was supplied to
    build_ground_truth, otherwise attribute *names* (strings). gt maps each source
    (reference) dataset index -> set of valid target dataset indices.
    """
    query: str
    pos: list = field(default_factory=list)
    neg: list = field(default_factory=list)
    gt: dict = field(default_factory=dict)  # ref_idx:int -> set[int]


def load_eval(json_path):
    """Read the raw evaluation JSON (list of entries)."""
    with open(json_path) as f:
        return json.load(f)


def _parse_names(query):
    """Split a query into (pos_names, neg_names) without needing an attr_dict."""
    pos, neg = [], []
    for tok in query.split(','):
        tok = tok.strip()
        if not tok:
            continue
        sign, name = tok[0], tok[1:]
        if sign == '+':
            pos.append(name)
        elif sign == '-':
            neg.append(name)
        else:
            raise ValueError(f"Segno mancante/non valido: {tok!r}")
    return pos, neg


def build_ground_truth(json_path, attr_dict=None, min_targets=5):
    """Parse the JSON into a list of QueryGT.

    - ref keys cast to int, target lists to sets.
    - sources with < min_targets valid targets dropped (matches the ">=5 valid targets"
      benchmark rule; the JSON appears pre-filtered, this is defensive).
    - pos/neg are column indices when attr_dict is given, else attribute names.
    """
    out = []
    for entry in load_eval(json_path):
        query = entry['query']
        if attr_dict is not None:
            from load_data import parse_query  # reuse utility; lazy so torch loads only when needed
            pos, neg = parse_query(query, attr_dict)
        else:
            pos, neg = _parse_names(query)

        gt = {}
        for ref, targets in entry['ground_truth'].items():
            tgt_set = set(targets)
            if len(tgt_set) >= min_targets:
                gt[int(ref)] = tgt_set

        out.append(QueryGT(query=query, pos=pos, neg=neg, gt=gt))
    return out


def valid_sources(query_gt):
    """Source (reference) indices to evaluate on: the post-filter JSON keys."""
    return sorted(query_gt.gt.keys())
