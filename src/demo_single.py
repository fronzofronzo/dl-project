"""Demo qualitativa: una immagine sorgente + un prompt -> cosa risale in classifica.

CLIP non genera immagini: l'"output" sono le immagini del DB piu' simili a v_t.
Mostriamo tre righe: SORGENTE / top-K senza modifica / top-K con il prompt.
Sotto ogni immagine recuperata: ✓ se soddisfa il prompt, ✗ altrimenti.

Uso:
  .venv/bin/python src/demo_single.py "+Eyeglasses"
  .venv/bin/python src/demo_single.py "+Smiling" 13 4.0
  (args: query [ref_idx] [alpha])
"""
import sys
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from baselines import build_direction_axes, contrastive_query
from groundtruth import build_ground_truth
from load_data import load_data, wire_attributes
from retrieval import load_db, rank

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
K = 5
CELL_W, CELL_H = 128, 156
PAD, LABEL_H, MARK_H = 10, 24, 20


def parse_query(query):
    pos, neg = [], []
    for tok in query.split(','):
        tok = tok.strip()
        if not tok:
            continue
        (pos if tok[0] == '+' else neg).append(tok[1:])
    return pos, neg


def satisfies(attr_vec, pos_idx, neg_idx):
    return (all(int(attr_vec[i]) == 1 for i in pos_idx)
            and all(int(attr_vec[i]) == 0 for i in neg_idx))


def cell(celeba, idx, mark=None, draw_font=None):
    """Una cella: immagine ridimensionata + striscia testo sotto (mark)."""
    img = celeba[idx][0].convert("RGB").resize((CELL_W, CELL_H))
    canvas = Image.new("RGB", (CELL_W, CELL_H + MARK_H), "white")
    canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)
    label = f"#{idx}" if mark is None else f"#{idx} {mark}"
    d.text((3, CELL_H + 3), label, fill="black", font=draw_font)
    return canvas


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "+Eyeglasses"
    ref_idx = int(sys.argv[2]) if len(sys.argv) > 2 else None
    alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0

    pos, neg = parse_query(query)
    celeba = load_data(ROOT / "data", split="test", download=False)
    attr_dict = wire_attributes(celeba)
    pos_idx = [attr_dict[p] for p in pos]
    neg_idx = [attr_dict[n] for n in neg]
    male_idx = attr_dict["Male"]

    # sorgente: se non data, prima sorgente FEMMINA della gt di questa query
    if ref_idx is None:
        gts = {g.query: g for g in build_ground_truth(ROOT / "data" / "celeba_evaluation.json")}
        if query not in gts:
            raise SystemExit(f"Query {query!r} non nel JSON; passa un ref_idx esplicito.")
        for s in gts[query].gt:
            if int(celeba[s][1][male_idx]) == 0:      # femmina
                ref_idx = s
                break
        ref_idx = ref_idx if ref_idx is not None else next(iter(gts[query].gt))

    db = load_db(ROOT / "data" / "clip_features_test.pt")
    axes = build_direction_axes(set(pos) | set(neg))

    v_ref = db[ref_idx]
    no_edit = rank(v_ref, db, exclude={ref_idx}, k=K)
    v_t = contrastive_query(v_ref, pos, neg, axes, alpha=alpha)
    edited = rank(v_t, db, exclude={ref_idx}, k=K)

    # check numerico: quanti dei top-K soddisfano il prompt
    def hit_count(ids):
        return sum(satisfies(celeba[j][1], pos_idx, neg_idx) for j in ids)
    ref_ok = satisfies(celeba[ref_idx][1], pos_idx, neg_idx)
    print(f"query={query!r}  ref=#{ref_idx} (soddisfa gia'? {ref_ok})  alpha={alpha}  K={K}")
    print(f"  no-edit  : {hit_count(no_edit)}/{K} soddisfano il prompt  {no_edit}")
    print(f"  contrastive: {hit_count(edited)}/{K} soddisfano il prompt  {edited}")

    render(celeba, ref_idx, no_edit, edited, pos_idx, neg_idx, query, alpha)


def render(celeba, ref_idx, no_edit, edited, pos_idx, neg_idx, query, alpha, out_name=None):
    try:
        font = ImageFont.truetype("Arial.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    grid_w = PAD + K * (CELL_W + PAD)
    block_h = LABEL_H + CELL_H + MARK_H + PAD
    canvas = Image.new("RGB", (grid_w, 3 * block_h + PAD), "white")
    d = ImageDraw.Draw(canvas)

    def mark(j):
        ok = (all(int(celeba[j][1][i]) == 1 for i in pos_idx)
              and all(int(celeba[j][1][i]) == 0 for i in neg_idx))
        return "OK" if ok else "x"

    rows = [("SORGENTE", [ref_idx]),
            ("Simili SENZA modifica", no_edit),
            (f"Con  {query}  (alpha={alpha})", edited)]

    for r, (title, ids) in enumerate(rows):
        y0 = PAD + r * block_h
        d.text((PAD, y0), title, fill="black", font=font)
        for c, j in enumerate(ids):
            x = PAD + c * (CELL_W + PAD)
            m = None if (r == 0) else mark(j)
            canvas.paste(cell(celeba, j, mark=m, draw_font=font),
                         (x, y0 + LABEL_H))

    RESULTS.mkdir(exist_ok=True)
    if out_name is None:
        out_name = "demo_" + re.sub(r"[^A-Za-z0-9]+", "_", query).strip("_") + ".png"
    out = RESULTS / out_name
    canvas.save(out)
    print(f"salvato -> {out}")


if __name__ == "__main__":
    main()
