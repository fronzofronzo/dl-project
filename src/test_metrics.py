from pathlib import Path

from metrics import recall_at_k, precision_at_k, evaluate_query, evaluate_all
from groundtruth import build_ground_truth, valid_sources, QueryGT

EPS = 1e-9


def approx(a, b):
    return abs(a - b) < EPS


# ---------------------------------------------------------------- primitives
def test_recall_hit_rate():
    ranked = [10, 11, 12, 13, 14, 15]
    gt = {15}                       # hit only at position 6
    assert recall_at_k(ranked, gt, 1) == 0.0
    assert recall_at_k(ranked, gt, 5) == 0.0
    assert recall_at_k(ranked, gt, 10) == 1.0      # k > len(ranked) ok
    assert recall_at_k(ranked, {11}, 5) == 1.0     # hit inside k
    assert recall_at_k(ranked, set(), 5) == 0.0    # empty gt
    assert recall_at_k(ranked, {99}, 5) == 0.0     # no intersection


def test_precision_fraction():
    ranked = [1, 2, 3, 4, 5]
    gt = {2, 5}                      # 2 of top-5 are gt
    assert approx(precision_at_k(ranked, gt, 5), 0.4)
    assert approx(precision_at_k(ranked, gt, 1), 0.0)   # top-1 = {1}, miss
    assert approx(precision_at_k(ranked, {1, 2}, 1), 1.0)
    # gt larger than k: still divides by k
    assert approx(precision_at_k(ranked, {1, 2, 3, 4, 5}, 5), 1.0)
    try:
        precision_at_k(ranked, gt, 0)
        assert False, "k=0 should raise"
    except ValueError:
        pass


def test_exclude_self():
    ranked = [7, 1, 2, 3, 4]        # 7 = source's own index, should be dropped
    gt = {1}
    assert approx(precision_at_k(ranked, gt, 1, exclude={7}), 1.0)
    assert recall_at_k(ranked, gt, 1, exclude={7}) == 1.0
    assert recall_at_k(ranked, gt, 1) == 0.0        # without exclude, top-1 = 7, miss


# ---------------------------------------------------------------- aggregation
def test_evaluate_query_mean():
    # source A: hit@1 ; source B: miss@1, hit@5 ; source C: not in rankings -> skipped
    rankings = {
        100: [5, 9, 9, 9, 9],        # gt {5} -> recall@1 1, prec@1 1
        200: [9, 9, 8, 7, 5],        # gt {5} -> recall@1 0, recall@5 1, prec@5 0.2
    }
    qgt = QueryGT(query="+Test", gt={100: {5}, 200: {5}, 300: {5}})
    res = evaluate_query(rankings, qgt, ks=(1, 5))
    assert res["n_sources"] == 2                    # 300 skipped (no ranking)
    assert approx(res["recall@1"], 0.5)             # (1 + 0)/2
    assert approx(res["recall@5"], 1.0)             # (1 + 1)/2
    assert approx(res["precision@1"], 0.5)          # (1 + 0)/2
    assert approx(res["precision@5"], (0.2 + 0.2) / 2)


def test_evaluate_query_no_sources():
    qgt = QueryGT(query="+Test", gt={1: {2}})
    res = evaluate_query({}, qgt, ks=(1, 5))        # no overlapping sources
    assert res["n_sources"] == 0
    assert res["recall@1"] == 0.0 and res["precision@5"] == 0.0


def test_evaluate_all_macro():
    q1 = QueryGT(query="+A", gt={1: {2}})
    q2 = QueryGT(query="+B", gt={1: {2}})
    rankings_per_query = {
        "+A": {1: [2, 9, 9]},        # recall@1 1
        "+B": {1: [9, 2, 9]},        # recall@1 0
    }
    rows = evaluate_all(rankings_per_query, [q1, q2], ks=(1,))
    assert set(rows) == {"+A", "+B", "MACRO"}
    assert approx(rows["MACRO"]["recall@1"], 0.5)   # (1 + 0)/2
    assert rows["MACRO"]["n_sources"] == 2


# ---------------------------------------------------------------- real JSON
def test_real_json_parse():
    json_path = Path(__file__).resolve().parent.parent / "data" / "celeba_evaluation.json"
    gts = build_ground_truth(json_path)             # no attr_dict -> pos/neg are names
    assert len(gts) == 14, f"expected 14 entries, got {len(gts)}"

    by_query = {}
    for qgt in gts:
        # every kept source has >= 5 targets and an int key
        for ref, tgts in qgt.gt.items():
            assert isinstance(ref, int)
            assert len(tgts) >= 5
        # valid_sources matches the gt keys
        assert valid_sources(qgt) == sorted(qgt.gt.keys())
        by_query.setdefault(qgt.query, qgt)

    # pos/neg parsing matches the query strings
    composed = by_query["-Smiling, +Eyeglasses, +Wearing_Hat"]
    assert composed.neg == ["Smiling"]
    assert composed.pos == ["Eyeglasses", "Wearing_Hat"]
    assert by_query["+Smiling"].pos == ["Smiling"] and by_query["+Smiling"].neg == []


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
