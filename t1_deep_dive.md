# T1 — Adapter a Cross-Attention condizionale: analisi in profondità e piano di implementazione

> Documento operativo per **Persona 1** (responsabile T1). Deriva da `next_steps.md`.
> Corrisponde alla **Soluzione B** dello spec dell'assignment.
> **Consegna: 5 luglio 2026.** Aggiornato allo stato reale del repo.

---

## 0. Stato reale del repo (verificato sul codice, non sul piano)

La **spina dorsale condivisa è già scritta** in `src/solution_b/`. Non va re-inventata —
va solo collegata. Inventario:

| Pezzo | File | Stato | Note |
|---|---|---|---|
| Contratto Φ + baseline MLP | `src/solution_b/phi.py` | ✅ fatto | `MLPPhi`, definisce l'interfaccia che T1 DEVE rispettare |
| Sampler self-supervised | `src/solution_b/sampler.py` | ✅ fatto | `TrainData`, FLIP-REF, hard-neg "viola esattamente 1 vincolo", smoke test incluso |
| Loss InfoNCE + ancora identità | `src/solution_b/losses.py` | ✅ fatto | `info_nce`, `identity_anchor`, `total_loss` |
| Loop di training | `src/solution_b/train.py` | ✅ fatto | Φ swappabile, valida ogni 500 step, salva best checkpoint |
| Harness di eval | `src/solution_b/run.py` | ✅ fatto | `eval_phi`, `write_results`, confronto vs naive/clay |
| Path | `src/common/paths.py` | ✅ fatto | `DB_TRAIN`, `ATTRS_TRAIN` già dichiarati |

**IL BLOCCO REALE — feature CLIP del train mancano.**
- `data/celeba_attrs_train.pt` (`ATTRS_TRAIN`) → **esiste** (52 MB, 40 attributi binari).
- `data/clip_features_train.pt` (`DB_TRAIN`) → **NON esiste**.
- `TrainData.__init__` fa `torch.load(DB_TRAIN)` → **crash immediato finché non lo generi**.

Quindi: tutta la pipeline T1 è collegabile in poche ore, MA niente gira finché non
estrai le feature CLIP del train. **È il prerequisito giorno-0, parte da qui.**

### Cosa NON è ancora stato verificato (controllare prima di fidarsi)
- Il sampler è girato davvero con dati reali? Lo smoke test (`python -m src.solution_b.sampler`)
  fallisce ora perché `DB_TRAIN` manca → **non sappiamo il positive-found rate vero**.
- `MLPPhi` è mai stato addestrato end-to-end? Nessun `phi_mlp.pt` in `results/` → **no**.
  Va fatto come smoke test della pipeline PRIMA di costruire T1.

---

## 1. Cosa deve fare T1 (e perché vince dove CLAY perde)

T1 è la carta **originalità/creatività**. Una piccola rete di attenzione guarda
**tutte le condizioni insieme** e in relazione a `v_ref`, produce una query modificata.

I 4 limiti CLAY che T1 deve chiudere, mappati su componenti concrete:

| Limite CLAY | Componente T1 | Meccanismo |
|---|---|---|
| **P1** segno/polarità | `sign_emb[{+1,−1}]` sommato al token attributo | segno esplicito nel token |
| **P2** pesatura | gate FiLM `g(v_ref, conds)` | dosa quanto applicare Δ |
| **P3** interazione/conflitto | cross-attention sull'insieme di condizioni | l'attenzione vede i token insieme, non li somma ciecamente |
| **P4** dinamico | query = `v_ref`, attention condizionata su `v_ref` | pesi dipendono dalla foto reale |

I 3 fallimenti misurati della Soluzione A che il training aggira automaticamente:
1. **Modality gap** → T1 lavora su `v_ref` (immagine) + token appresi (spazio proprio,
   NON prompt testuali) → niente coseno testo-immagine rotto.
2. **Direzioni entangled** → l'attenzione + gate imparano combinazioni non lineari →
   non eredita l'entanglement degli assi testuali CLIP.
3. **Polarità** → `sign_emb` + hard-negative mining (già nel sampler).

---

## 2. Architettura di Φ_T1 — dettaglio

### 2.1 Contratto da rispettare (NON negoziabile)
Da `phi.py`, ogni Φ implementa **esattamente** questa firma, così T1 entra nel loop
senza toccare `train.py`/`run.py`:

```python
forward(v_ref, cond_col, cond_sign, cond_mask) -> v_q
  v_ref     [B, 512]   reference image features (L2-norm, CLIP frozen)
  cond_col  [B, C]     long, indice colonna attributo per condizione (0 = pad)
  cond_sign [B, C]     float, +1 (additivo) / -1 (sottrattivo) / 0 (pad)
  cond_mask [B, C]     bool, True per condizioni reali
  -> v_q    [B, 512]   query composta, L2-normalizzata, solo lato query
```

⚠️ **Attenzione pad-vs-attributo-0:** `cond_col=0` è sia il padding sia l'attributo
con indice 0 (`5_o_Clock_Shadow`). La verità è `cond_mask`, NON `cond_col==0`.
Usa **sempre** `cond_mask` per distinguere; non usare mai `cond_col==0` come "è pad".

### 2.2 I componenti (in ordine di forward pass)

**(a) Token-condizione con segno.**
```
attr_emb   : nn.Embedding(40, d_model)      # tabella attributi appresa
sign_emb   : nn.Embedding(2, d_model)       # 0 -> negativo, 1 -> positivo
token_i = attr_emb[cond_col_i] + sign_emb[(cond_sign_i+1)/2]
```
Mappa il segno {−1,+1} → indice {0,1} per l'embedding. Risolve **P1**.

**(b) Token query da `v_ref`.**
`v_ref` è 512-dim (spazio CLIP). Se `d_model ≠ 512`, proietta:
```
q_proj : Linear(512, d_model)
q_token = q_proj(v_ref)                      # [B, d_model], 1 token query
```

**(c) Blocco cross-attention.**
- query = `q_token` (dalla reference)
- key/value = i token-condizione `[B, C, d_model]`
- `nn.MultiheadAttention(d_model, n_heads, batch_first=True)`, `key_padding_mask = ~cond_mask`
- l'attenzione vede tutte le condizioni insieme e rispetto a `v_ref` →
  **interazione/conflitto (P3)** + **dinamico (P4)**, permutation-invariant per costruzione.
```
attn_out, _ = mha(query=q_token[:,None,:], key=tokens, value=tokens,
                  key_padding_mask=~cond_mask)        # [B, 1, d_model]
```
- (opzionale, ablation): >1 layer, FFN+LayerNorm stile transformer encoder.

**(d) Proiezione Δ in spazio CLIP.**
```
delta_proj : Linear(d_model, 512)
delta = delta_proj(attn_out.squeeze(1))      # [B, 512] vettore di modifica
```

**(e) Gate FiLM residuo.**
```
gate_net : MLP([v_ref ; pooled_cond_summary]) -> sigmoid -> [B, 512]  (o scalare)
g = gate_net(...)                            # pesatura dinamica (P2)
v_q = v_ref + g * delta                      # residuo preserva identità
return F.normalize(v_q, dim=1)
```
Decisioni di design da fissare (e ablare):
- **g scalare vs vettoriale** (512-dim FiLM): vettoriale più espressivo, scalare più stabile.
- **residuo ambiente vs tangente**: lo spec chiede l'ablation tangente; partire ambiente
  (le findings A dicono tangente ≈ ambiente, ma per B va ri-testato).

### 2.3 Iperparametri di partenza (da sweepare dopo)
- `d_model = 256`, `n_heads = 4`, `n_layers = 1`.
- gate vettoriale, residuo ambiente.
- Parametri totali stimati: ~poche centinaia di k → gira su Colab in minuti.

---

## 3. Esempio concreto (per validare la logica)

Reference = **donna sorridente senza occhiali**. Query = `+Eyeglasses, −Smiling`.
1. `token_1 = attr_emb[Eyeglasses] + sign_emb[+]`, `token_2 = attr_emb[Smiling] + sign_emb[−]`.
2. `q_token = q_proj(v_ref)` guarda i 2 token. Rete addestrata:
   - `v_ref` non ha occhiali → attenzione alta su `+Eyeglasses`.
   - `v_ref` sorride → attenzione alta su `−Smiling`.
   - se la ref avesse già gli occhiali, l'attenzione su `+Eyeglasses` calerebbe da sola.
3. `g ⊙ Δ` dosa; `+ v_ref` tiene l'identità.
4. `normalize(v_q)`, coseno vs DB frozen → persone serie con occhiali, stessa identità.

**Conflitto:** con `+Eyeglasses & −Smiling` correlati nei dati, l'attenzione vede
entrambi i token insieme e impara a non far cancellare una richiesta dall'altra —
cosa che la singola SVD di CLAY non può fare. **È l'argomento headline da dimostrare**
sulle query: `+Eyeglasses & −Smiling`, `−Male & −Mustache`, `−Smiling & +Eyeglasses & +Wearing Hat`.

---

## 4. Loss (riusare `losses.py`, estendere se serve)

Già pronto `total_loss = InfoNCE + lam_id · identity_anchor`.
- **InfoNCE**: positivi vs hard-neg + in-batch negatives, `τ=0.07`.
- **ancora identità**: `1 − cos(v_q, v_ref)`, `lam_id=0.1`.

Aux opzionale per T1 (da aggiungere se la polarità non regge):
- **repulsione negativa**: spinge `v_q` via dalla direzione dell'attributo negato.
  Non c'è ancora in `losses.py` → aggiungerla come termine separato solo se serve
  (misurare prima senza).

---

## 5. PIANO DI IMPLEMENTAZIONE — passi chiari e ordinati

### Passo 0 — Sblocco prerequisito (giorno 0, BLOCCA TUTTO)
**Estrai le feature CLIP del train.** Senza, niente gira.
- [ ] Riusa l'estrattore esistente: guarda `src/common/features.py` (stesso procedimento
      del DB test) e come è stato creato `clip_features_test.pt`.
- [ ] Carica CelebA split **train** via `celeba[idx]` (MAI path a mano — pitfall dataset).
- [ ] Estrai CLIP ViT-B/32 frozen, L2-normalizza, salva → `data/clip_features_train.pt`.
- [ ] **Allinea le righe a `celeba_attrs_train.pt`** (riga i di feature ↔ riga i di attrs).
      Se gli attrs sono stati salvati su un certo ordine/subset, le feature DEVONO seguirlo.
- [ ] 162k immagini = pesante su Colab. Se la GPU è stretta → **subset 30–50k** reference
      (basta per un adapter). Salva anche il subset di attrs allineato.
- [ ] Verifica: `F.shape[0] == L.shape[0]`, entrambe L2-norm / bool.

### Passo 1 — Smoke test della pipeline esistente (giorno 0–1)
Prima di scrivere T1, dimostra che spina dorsale + baseline girano.
- [ ] `python -m src.solution_b.sampler` → deve passare gli assert e stampare
      **positive-found rate** decente (se è basso, il sampler va sistemato PRIMA di T1).
- [ ] `python -m src.solution_b.train` (MLPPhi) → deve addestrare e produrre
      `results/phi_mlp.pt` + `results/solution_b_mlp.md`. Questo è il **baseline interno**
      ("MLP vs attention", ablation richiesta dallo spec) e il check che loss/eval funzionano.
- [ ] Annota R@1/R@5 di MLPPhi → è il numero che T1 deve battere.

### Passo 2 — Implementa Φ_T1 (giorno 3–5)
- [ ] Nuovo file `src/solution_b/t1_attention.py`, classe `T1Phi(nn.Module)`.
- [ ] Rispetta il contratto `forward(v_ref, cond_col, cond_sign, cond_mask) -> v_q`.
- [ ] Implementa componenti §2.2 (a)→(e).
- [ ] **Test di forma + invarianti** (scrivi un mini smoke `_smoke()` come nel sampler):
  - [ ] output `[B,512]`, L2-norm (`v_q.norm(dim=1) ≈ 1`).
  - [ ] **permutation-invariance**: permuta l'ordine delle condizioni in `cond_col/sign/mask`
        → `v_q` invariato (entro tolleranza fp). È la prova che la cross-attention è set-input.
  - [ ] **mask honesty**: aggiungere colonne pad non cambia `v_q`.
  - [ ] **identità a vuoto**: con `cond_mask` tutto False, `v_q ≈ normalize(v_ref)`
        (Δ non deve esplodere su zero condizioni — controlla il gate).
  - [ ] backward: gradienti finiti e non nulli su tutti i parametri di Φ.

### Passo 3 — Aggancia al training (giorno 5)
- [ ] In `train.py`/`run.py`: parametrizza la scelta di Φ (`MLPPhi` vs `T1Phi`) —
      es. argomento `name="t1"` che istanzia `T1Phi`. Minima modifica, non duplicare il loop.
- [ ] `eval_phi` in `run.py` usa già il contratto → T1 ci entra senza modifiche.
- [ ] Train completo, salva `phi_t1.pt`, freeze `solution_b_t1.md`.

### Passo 4 — Tuning (giorno 5–9)
- [ ] Verifica che T1 batta MLPPhi e i baseline (naive, CLAY-SVD). Se non li batte →
      diagnosi: gate troppo aggressivo? identità persa? polarità ignorata (controlla
      le query negate)?
- [ ] Sweep ordinato (un fattore alla volta, registra ogni run):
      `n_heads/n_layers/d_model` · `τ` · `lam_id` · rapporto hard-neg (`k_hardneg` nel sampler).

### Passo 5 — Ablation per il report (giorno 9–11)
Tabella richiesta dallo spec, già pianificata:
- [ ] **sign-embedding sì/no** (rimuovi `sign_emb` → la polarità crolla? dimostra P1).
- [ ] **cross-attention vs MLP semplice** (T1 vs MLPPhi — già hai entrambi).
- [ ] **gate FiLM sì/no** (residuo nudo `v_ref + Δ` vs gated).
- [ ] **residuo tangente vs ambiente**.
- [ ] **τ** e **rapporto hard-negative**.
- [ ] Esempi qualitativi: successi + fallimenti, focus sulle 3 query headline composte.

### Passo 6 — Integrazione finale (giorno 12–13, insieme a P2)
- [ ] Flatten in unico notebook Colab, run end-to-end da zero.
- [ ] Sezione report: matematica dell'architettura, forward pass, loss, tabelle vs baseline.

---

## 6. Rischi specifici di T1 (e mitigazioni)

| Rischio | Sintomo | Mitigazione |
|---|---|---|
| **Feature train mancanti** | crash `torch.load(DB_TRAIN)` | Passo 0, parte ora. È IL blocco. |
| **Disallineamento feature↔attrs** | metriche a caso, sampler trova falsi positivi | verifica esplicita righe allineate al Passo 0 |
| **Overfit** (T1 più grande di T2) | train loss giù, R@K eval fermo | tieni Φ piccolo (`d_model=256`, 1 layer), early-stop su R@1 eval (il loop già salva il best) |
| **Gate collassa** (g→0 o g→1) | T1 = identità pura, o identità persa | gate vettoriale + monitor `g.mean()`; aux identità (`lam_id`) |
| **Pad trattato come attributo 0** | `5_o_Clock_Shadow` iniettato ovunque | usa SEMPRE `cond_mask`, MAI `cond_col==0` |
| **Polarità ignorata** | attributi negati ancora presenti nei risultati | sign-emb + hard-neg (già nel sampler); ablation sign-on/off lo prova |
| **Tetto basso ViT-B/32** | R@5 ~0.13–0.18, niente miracoli | obiettivo = battere CLAY-SVD + A + MLP **in modo pulito**, non il numero assoluto |

---

## 7. CLAY-SVD non ancora implementato — flag di rigore

`results/baseline_clay.md` esiste → potrebbe essere già fatto (verificare il contenuto:
`next_steps.md` lo dava come "non ancora fatto", ma il file risultati c'è).
È il **vero SOTA da battere**. Se non è implementato/valutato sul protocollo corrente,
va chiuso prima dell'ablation finale — senza, manca l'argomento di rigore.
**Da coordinare con P2** (è nella spina dorsale condivisa, non è lavoro T1 puro).

---

## 8. Definition of Done per T1

- [ ] `data/clip_features_train.pt` esiste, allineato agli attrs.
- [ ] `src/solution_b/t1_attention.py` con `T1Phi` che passa tutti i test di §Passo 2.
- [ ] `results/phi_t1.pt` + `results/solution_b_t1.md` congelati.
- [ ] T1 **batte** naive, CLAY-SVD, Soluzione A e MLPPhi su R@1/R@5 (almeno sulle composte).
- [ ] Tabella di ablation §Passo 5 completa.
- [ ] Esempi qualitativi (successo + fallimento) sulle 3 query headline.
- [ ] Sezione report con matematica + forward + loss.
</content>
</invoke>
