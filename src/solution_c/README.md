# Solution C — Φ-Flow: conditional flow matching sull'ipersfera CLIP

## Idea

T1 e T2 sono editor **one-shot**: qualunque sia il modulo, l'edit è un singolo Δ
sommato a `v_ref`. Vincoli antagonisti ma correlati (es. `+Wearing_Lipstick` e
`−Heavy_Makeup`) producono direzioni quasi antiparallele che **si cancellano
nella somma** — quella query misura 0.000 per *tutti* i metodi del notebook, e
l'architettura unificata T1T2 è un risultato negativo documentato.

Φ-Flow riformula la composizione come **traiettoria**: un campo di velocità
appreso `u_θ(v_t, t | condizioni)` su S^511, integrato da `v_ref` per N passi di
Eulero. Rivalutare il campo nel punto corrente permette di **sequenzializzare**
gli edit (prima `+Lipstick`, poi `−Heavy_Makeup` *dal nuovo punto*) invece di
sommarli una volta sola. Gli editor one-shot sono il caso particolare N=1.

- **Dynamic** conditioning: Φ è letteralmente un sistema dinamico condizionato.
- **Hybrid** conditioning: assi testuali CLIP frozen (bridge T2) + *classifier
  guidance* da 40 probe lineari sugli attributi (`λ·∇_v Σ log p(vincolo|v_t)`),
  trapiantata dai diffusion model alla costruzione della query di retrieval.

Vincoli rispettati: CLIP frozen (si allena solo Φ, ~1.6M parametri), DB visivo
frozen, fusione solo query-side, sampler self-supervised (nessuna label extra).

## Training (conditional flow matching)

Nessuna integrazione durante il training: per ogni ancora si campiona
`t ~ U(0,1)`, si mette `x_t` sulla geodetica slerp `v_ref → positivo` e si
regredisce `u_θ(x_t, t)` sulla velocità analitica del percorso (norma θ,
tangente per costruzione). Ricampionando il positivo dal pool a ogni batch, il
flusso marginale trasporta `v_ref` verso il **centro della regione dei target
validi**. Termine opzionale retrieval-aligned: InfoNCE + identity anchor
sull'endpoint integrato con gradiente (`--lam-nce`, 0 = pura CFM, fallback
stabile).

**Warm start (de-risk):** il ramo direzioni/pesi è nome-compatibile con `T2Phi`
e carica `results/phi_t2_hybrid.pt`; le teste nuove (correzione interazione,
residuo libero) partono a zero → al passo 0 il campo di velocità **è** il campo
di edit di T2-hybrid. Il training parte dal modello migliore, non dal rumore.

## File

| file | contenuto |
|---|---|
| `flow_phi.py` | `FlowPhi` (campo di velocità + integrazione), slerp/exp-map, smoke test |
| `probes.py` | 40 probe logistiche lineari su feature train frozen (guidance) |
| `train_flow.py` | training CFM (+ endpoint InfoNCE), validazione periodica, best ckpt |
| `run_flow.py` | eval checkpoint; `--sweep` = griglia N × λ × T (solo inferenza) |
| `train_bench.py` | pseudo-benchmark delle 13 query sul TRAIN split (tuning senza toccare test) |
| `rerank.py` | re-ranking product-of-experts: `cos(v_q,x) + η·Σ logσ(sᵢzᵢ(x))` sui logits probe precalcolati del DB frozen; `--tune` = scelta di η su train |
| `gated.py` | `GatedPhi`: router \|C\|≥2 → flow, \|C\|=1 → T2-hybrid; `--train-evidence` = conferma cieca del gate su train |

## Comandi (VM con GPU, dalla root del repo)

```bash
# 0. smoke test (CPU, nessun dato richiesto oltre a phi_t2_hybrid.pt per il 4°)
python -m src.solution_c.flow_phi
python -m src.solution_c.probes --smoke

# 1. probe lineari (~1 min)
python -m src.solution_c.probes

# 2. run principale (warm start da T2-hybrid, CFM + endpoint InfoNCE)
python -m src.solution_c.train_flow --name flow_main

# 3. sweep inferenza: passi di Eulero N × guidance λ (economico, solo forward)
python -m src.solution_c.run_flow --ckpt results/phi_flow_flow_main.pt --name flow_main --sweep

# 4. ablation (in ordine di priorità, ~1-2h l'una)
python -m src.solution_c.train_flow --name flow_purecfm --lam-nce 0
python -m src.solution_c.train_flow --name flow_scratch --no-warm-start
python -m src.solution_c.train_flow --name flow_nofree  --no-free-residual
python -m src.solution_c.train_flow --name flow_seed1 --seed 1

# check rapido end-to-end (2 minuti, qualità irrilevante)
python -m src.solution_c.train_flow --name smoke --steps 50 --eval-every 25
```

Prerequisiti su VM: `data/clip_features_train.pt`, `data/celeba_attrs_train.pt`,
`data/clip_features_test.pt`, `data/celeba_evaluation.json`, `data/celeba/`
(per i nomi attributi), `results/phi_t2_hybrid.pt` (warm start).

## PoE re-ranking + gated hybrid (inference-only, dopo il training)

```bash
# smoke (CPU, nessun dato)
python -m src.solution_c.train_bench --smoke
python -m src.solution_c.rerank --smoke
python -m src.solution_c.gated --smoke

# 5. sweep del ckpt finale -> fissa (N*, T*) del flow
python -m src.solution_c.run_flow --ckpt results/phi_flow_flow_v3.pt --name flow_v3 --sweep

# 6. scelta di η su TRAIN (mai su test), uno per esperto
python -m src.solution_c.rerank --ckpt results/phi_flow_flow_v3.pt --steps N --horizon T --tune
python -m src.solution_c.rerank --t2-ckpt results/phi_t2_hybrid.pt --tune
python -m src.solution_c.rerank --no-phi --tune

# 7. conferma cieca del gate |C| su TRAIN
python -m src.solution_c.gated --flow-ckpt results/phi_flow_flow_v3.pt --steps N --horizon T --train-evidence

# 8. tabelle test (una sola run per config, η congelato al punto 6)
python -m src.solution_c.rerank --ckpt results/phi_flow_flow_v3.pt --steps N --horizon T --eta H --name flow_v3_poe --eta-sweep
python -m src.solution_c.rerank --t2-ckpt results/phi_t2_hybrid.pt --eta H --name t2_hybrid_poe
python -m src.solution_c.rerank --no-phi --eta H --name poe_only
python -m src.solution_c.gated --flow-ckpt results/phi_flow_flow_v3.pt --steps N --horizon T
python -m src.solution_c.gated --flow-ckpt results/phi_flow_flow_v3.pt --steps N --horizon T --eta H
```

## Cosa riportare

- Sweep N: N=1 ≈ T2 (pavimento), la curva R@1(N) misura quanto compra la
  decomposizione temporale — è l'argomento centrale del metodo.
- Sweep λ: contributo della guidance ibrida (probe), on/off pulito.
- Ablation warm start / pure-CFM / free-residual.
- Per-query: le query in conflitto (`+Wearing_Lipstick, −Heavy_Makeup, +Smiling`,
  `−Male, −Mustache`) sono il target dichiarato del metodo.
- Interpretabilità: pesi `w_i(t)` per passo → "il modello applica gli edit in
  ordine" (figura qualitativa).
- PoE: ablation flow / PoE-only / t2+PoE / flow+PoE; delta per-query sulle due
  query a 0.000 (il posterior sul candidato non può cancellarsi, la guidance
  sulla query sì — λ>0 non vince mai, η è il probe messo dal lato giusto).
- Gated: tabella evidenza train (|C|=1 vs ≥2) + riga headline gated(+PoE);
  dichiarare che il pattern è stato notato su test e confermato alla cieca su
  train (router a zero parametri, proprietà della query).
