# T1T2Phi — Cross-attention Weighted Direction Adapter

## Problema: cosa manca a T1 e T2 da soli

### T2 (Solution A)
T2 impara un dizionario di direzioni per-attributo `D ∈ ℝ^{40×512}` nello spazio immagine CLIP.
Per ogni condizione `i` compone:

```
v_q = normalize( v_ref + scale · Σ_i  w_i · s_i · d̂_i(v_ref) )
```

dove `d̂_i = normalize(D[col_i] + U[col_i]ᵀ · dir_net(v_ref, attr_emb[col_i]))` (I1).

**Difetto specifico — P3 (interaction):** i pesi `w_i = softplus(MLP(v_ref, attr_emb[col_i]))`
sono calcolati **indipendentemente** per ogni condizione. `weight_net` non vede mai le altre
condizioni nel batch. Quando chiedi `+Chubby, -Young`, il peso di `-Young` non sa che stai
chiedendo anche `+Chubby`. Questo causa sotto-performance sulle query composte.

### T1 (Solution B)
T1 usa cross-attention per fondere le condizioni:

```
q_token = q_proj(v_ref)                         # [B, d_model]
summary = cross_attn(q=q_token, kv=cond_tokens) # [B, d_model]
delta   = delta_proj(summary)                   # [B, dim] — edit amorfa
gate    = sigmoid(gate_net(v_ref, summary))     # [B, dim]
v_q     = normalize(v_ref + gate * delta)
```

T1 chiude P3 (le condizioni si "vedono" vicendevolmente tramite attention) ma il delta è
**un vettore amorfo** nello spazio 512-dim senza struttura geometrica per-attributo.
Non c'è un concetto di "direzione del sorriso" vs "direzione degli occhiali" — è tutto
codificato implicitamente nei pesi della rete.

---

## Soluzione: T1T2Phi

**Idea chiave:** usare le **direzioni geometriche di T2** (con tutta la loro struttura:
warm-start, I1, hybrid text) ma rimpiazzare `weight_net` con un meccanismo di weighting
**interaction-aware** ispirato a T1.

### Difetto nella proposta originale

Un'implementazione ingenua usa `v_ref` come query e le condizioni come KV:

```python
attn_out, _ = self.attn(q, tokens, tokens)  # [B, 1, d_model]
w = softplus(w_proj(attn_out)).expand(-1, C) # [B, C] ← STESSO peso per tutti
```

Questo dà un **singolo scalare globale** applicato a tutte le condizioni — non weighting
per-condizione. È equivalente a una FiLM gate scalare, già meno espressivo di `weight_net`.

### Fix: condizioni come query (self-attention)

Per ottenere pesi **per-condizione** che vedono le altre condizioni:

```python
# ogni token condizione = attr_emb + sign_emb + q_proj(v_ref)
tokens = attr_emb_attn[col] + sign_emb[sign] + q_proj(v_ref).unsqueeze(1)
tokens = tokens * cond_mask                    # [B, C, d_model]

# self-attention: ogni condizione vede le altre
ctx, _ = cond_sa(tokens, tokens, tokens,
                  key_padding_mask=~cond_mask) # [B, C, d_model]

# peso scalare per-condizione
w = softplus(w_proj(ctx).squeeze(-1))         # [B, C]
```

Risultato:
- `w_i` è diverso per ogni condizione (non uguale per tutte)
- `w_i` dipende da `v_ref` (via iniezione nel token)
- `w_i` dipende da tutte le altre condizioni `j≠i` (via self-attention)

### Pipeline completa

```
(a) d̂_i(v_ref) = normalize( D[col_i] + U[col_i]ᵀ·dir_net(v_ref, attr_emb[col_i]) )
                  ↑ T2 I1: direzioni geometriche condizionate su v_ref

(b) t_i = attr_emb_attn[col_i] + sign_emb[sign_i] + q_proj(v_ref)
          ↑ T1: token condizione con polarity + reference

(c) ctx_i = self_attention(t_1, ..., t_C)[i]
            ↑ T1: ogni condizione vede le altre (P3)

(d) w_i = softplus(w_proj(ctx_i))
          ↑ peso scalare per-condizione, interaction-aware

(e) edit = log_scale.exp() · Σ_i  w_i · s_i · d̂_i(v_ref)
           ↑ T2: composizione geometrica firmata e mascherata

(f) v_q = normalize(v_ref + edit)
          ↑ residual identità
```

### Miglioramenti CLAY chiusi

| Limite | T2 | T1 | T1T2 |
|--------|----|----|------|
| P1 sign | ✓ (cond_sign) | ✓ (sign_emb) | ✓ (entrambi) |
| P2 weighting | ✓ (weight_net per-cond) | ✓ (FiLM gate) | ✓ (attention per-cond) |
| P3 interaction | ✗ (pesi indipendenti) | ✓ (cross-attn) | ✓ (self-attn su cond) |
| P4 dynamic | ✓ (dir_net usa v_ref) | ✓ (query=v_ref) | ✓ (v_ref iniettato in ogni token) |

### Invarianti di T2 preserved

- **Warm-start I3**: `load_directions(image_axes(F_train, L_train))` inizializza `D` con
  differenze statistiche di attributo — invariato
- **Input-conditioned directions I1**: `cond_dir=True` — invariato  
- **Correlation-aware ortho_reg I2**: `ortho_reg(target=corr_matrix)` su `D` — invariato
- **Text hybrid**: `text_proj` + `text_gate` che fonde assi CLIP text in `D` — invariato
- **Composizione geometrica**: le direzioni `d̂_i` sono unitarie, l'edit è una somma firmata
  nello spazio immagine — interpretabilità preservata

### Perché non un semplice ensemble T1+T2

Un ensemble `normalize(v_q_T1 + v_q_T2)` combina due vettori di query già normalizzati.
T1T2 invece **impara insieme** come pesare le direzioni geometriche di T2 usando l'attention
di T1 — è un'architettura unificata, non una post-hoc combinazione. Il training end-to-end
su questa singola rete ottimizza direttamente le due componenti in modo coordinato.

---

## Hyperparameters di training

Eredita da T2 (già ottimizzati):
- `LR = 3e-4` (T2 mostrò che 1e-3 oscillava con direzioni)
- `LAM_ID = 0.4`
- `LAM_ORTH = 0.1` con `ORTHO_MODE = "corr"`
- `WARM_START = True`, `HYBRID = True`
- Cosine LR decay (da T1, non usato in T2)
