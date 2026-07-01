# Roadmap → Notebook finale
_Deadline: 5 luglio 2026 — Budget VM residuo: ~25h_

---

## Requisiti assignment (da slide)

- **Single Jupyter Notebook su Google Colab**, self-contained
- Codice eseguibile, pulito, commentato
- **Output cells già salvati** (il grader verifica senza re-runnare)
- Report comprehensive in **Markdown cells tra le code cells**:
  - Descrizione metodologica dettagliata
  - Experimental setup con motivazione delle scelte
  - Results and discussion

---

## Stato attuale (1 luglio)

| modello | R@1 | R@5 | R@10 | checkpoint |
|---|---|---|---|---|
| **CLAY** (SOTA reimpl.) | **0.003** | **0.010** | **0.019** | — |
| Naive baseline | 0.022 | 0.072 | — | — |
| Sol A no-training | 0.037 | 0.110 | — | — |
| MLP (sol B) | ~0.060 | — | — | phi_mlp.pt |
| T1 cross-attention (sol B) | 0.082 | 0.240 | 0.355 | phi_t1.pt |
| T2 no hybrid (sol A) | 0.067 | 0.183 | 0.268 | phi_t2.pt |
| T2 hybrid **(best)** | **0.090** | **0.240** | **0.349** | phi_t2_hybrid.pt |
| Ensemble T1+T2hybrid | 0.090 | 0.240 | 0.352 | — |

**Narrativa centrale del report:** CLAY crolla a R@1=0.003 (peggio del naive) perché il subspace sign-agnostico distrugge il segnale compositivo. T2 hybrid raggiunge R@1=0.090 → **+30x rispetto a CLAY**. Il report deve tracciare esplicitamente ogni limite P1–P4 di CLAY alla soluzione che lo indirizza.

---

## Esperimenti — tutti completati ✓

### Risultati ablation study T2 (completati 1 luglio)

| ablation | cosa disabilita | R@1 | ΔR@1 | R@5 | R@10 |
|---|---|---|---|---|---|
| T2 hybrid (baseline) | — | **0.090** | — | 0.240 | 0.349 |
| I2 off (`ortho_mode=off`) | decorrelazione D | 0.083 | −0.007 (−8%) | 0.239 | 0.349 |
| I3 off (`warm_start=False`) | init da image probe | 0.080 | −0.010 (−11%) | 0.227 | 0.337 |
| I1 off (`cond_dir=False`) | input-conditioning | 0.065 | −0.025 (−28%) | 0.189 | 0.272 |

**Interpretazione:**
- **I1 è il contributo dominante** (−28%): senza conditioning su v_ref, D è statico come CLAY image-space. Il gate diventa sempre più negativo (−0.096 a fine training) — il modello "lotta" contro direzioni misallineate senza poterle adattare.
- **I3 è moderato** (−11%): warm-start porta D nel ballpark semantico corretto, accelera la convergenza. Senza, l'ortho_reg collassa a ≈0.001 (nulla su cui regolarizzare).
- **I2 è il contributo minore** (−8% R@1, −0.4% R@5): senza ortho_reg il gate diventa positivo (+0.030) — il modello compensa fidandosi di più del testo. Aiuta soprattutto la precision top-1.
- **Ordine di importanza:** I1 >> I3 > I2.
- **Il gate come diagnostico:** segno negativo = modello si fida delle direzioni immagine; positivo = si fida del testo. I1-off porta gate a −0.096 (immagine irrecuperabile), I2-off a +0.030 (testo come surrogate).

### Altri esperimenti completati
- ✓ T2 tau05+cosine LR: R@1=0.089 (non supera baseline)
- ✓ T2 tau05 fixed LR: R@1=0.089 (confermato τ non è il bottleneck)
- ✓ Ensemble T1+T2: R@1=0.090 (stesso macro, wins/losses si bilanciano)
- ✓ CLAY baseline: R@1=0.003 (confermato failure P1–P4)

---

## Fase 3 — Preparazione deliverable (~3h)

### 3a. Upload checkpoints su Google Drive
Caricare tutti i `.pt` finali su Drive in una cartella condivisa:
- `phi_mlp.pt`
- `phi_t1.pt`
- `phi_t2.pt`
- `phi_t2_hybrid.pt` (o best model finale)

### 3b. Esportare notebook dalla VM con output salvati
Eseguire tutto sulla VM, salvare il `.ipynb` con output intatti, caricare su Colab.
I grader verificano gli output senza re-runnare — non serve che il training riparta.

---

## Fase 4 — Stesura notebook (~8h)

### Struttura celle (Markdown + Code alternati)

```
[MD]   # 1. Setup e problema
       - Cos'è il compositional image retrieval e perché è difficile
       - CLIP ViT-B/32 frozen, CelebA test split, metriche R@K / P@K
[CODE] pip install / import / mount Drive / load CLIP / load DB
[OUT]  già salvato

[MD]   ## 2. Baselines
       - Naive arithmetic: v_ref + Σt⁺ − Σt⁻  (R@1=0.022)
       - CLAY reimplementato: subspace P=V_k V_kᵀ, no sign, no weighting (R@1=0.003)
       ⚠️ Punto narrativo chiave: spiegare PERCHÉ CLAY crolla
         → il subspace sign-agnostico è incapace di distinguere +attr da -attr
         → proietta v_ref in un sottospazio che mescola "con" e "senza" attributo
         → P non ha polarity: CLAY riordina il DB ma in modo indifferente al segno
         → risultato: peggio del naive su ogni query composta
       - Tabella P1–P4: limite CLAY → fix T1 → fix T2
[CODE] eval CLAY + eval naive + tabella comparativa
[OUT]  già popolata

[MD]   ## 3. Solution B — T1: cross-attention adapter
       - Architettura: signed condition tokens, cross-attention, FiLM gate, residual
       - Come risponde a P1 (sign embedding esplicito) e P3 (cross-attention = interazione)
       - Forward pass in matematica: v_q = v_ref + g(v_ref, conds) ⊙ Δ
       - Loss: InfoNCE + identity anchor, hard negatives (violare esattamente 1 constraint)
[CODE] class T1Phi  ← commenti inline su ogni componente
[OUT]  —
[MD]   ### 3.1 Training T1
       - Scelte: τ=0.07, LR=3e-4, hard-neg k=4, STEPS=6000
       - MLP come ablation implicita: cross-attn off → R@1=0.060 vs T1 R@1=0.082 (+37%)
[CODE] train loop con USE_PRETRAINED=True default
[OUT]  log training già salvato
[MD]   ### 3.2 Risultati T1
       - Tabella per-query, confronto vs naive (+4x) e vs CLAY (+27x)
       - Dove T1 è forte: query composte con interazione (es. +Eyeglasses,+Smiling)
[CODE] eval + tabella per-query
[OUT]  tabella già popolata ← R@1=0.082

[MD]   ## 4. Solution A — T2: dizionario direzioni appreso
       - Architettura: D [40×512] image-space, U input-conditioned, text hybrid
       - Miglioramenti I1–I4 rispetto a T1:
         I1 input-conditioned directions (cond_dir): pesi U da v_ref → P4 (dynamic)
         I2 ortho_reg correlation-aware: D decorrelato secondo struttura reale attributi → P3
         I3 warm-start da image probe: D parte semanticamente significativa → convergenza
         I4 text hybrid: text_gate fonde asse CLIP text per prior linguistico → P1 rinforzato
       - Forward pass in matematica: v_q = normalize(v_ref + U(v_ref) ⊙ (D + α·T))
       - PERCHÉ supera T1: direzioni image-space evitano il modality gap che affligge T1
         (T1 usa text embedding come query-key-value → sempre nel text cone)
[CODE] class T2Phi  ← commenti inline su D, U, dir_net, weight_net, text_gate
[OUT]  —
[MD]   ### 4.1 Training T2
       - Warm-start D da image probe (I3): d_img(j) = mean(F[L[:,j]=1]) − mean(F[L[:,j]=0])
       - Ortho_reg correlation-aware (I2): target = matrice correlazione attributi CelebA-train
       - τ=0.07, LR=3e-4 fisso (cosine LR testato ma non aiuta), STEPS=6000, EVAL_EVERY=100
[CODE] train loop con USE_PRETRAINED=True default
[OUT]  log training già salvato
[MD]   ### 4.2 Risultati T2 hybrid
       - Tabella per-query, confronto vs CLAY (+30x R@1), vs naive (+4x), vs T1
       - Query problematiche: -Male,-Mustache (0.000), +Lipstick,-Makeup,+Smiling (0.000)
         → analisi: troppo pochi source (27 e 34), ground truth sparsa
[CODE] eval + tabella per-query
[OUT]  tabella già popolata ← R@1=0.090

[MD]   ## 5. Ablation study
       - Tabella I1/I2/I3 con delta R@1 vs baseline 0.090 (numeri definitivi):
           I1 off: 0.065 (−28%) ← contributo dominante
           I3 off: 0.080 (−11%) ← moderato
           I2 off: 0.083 (−8%)  ← minore ma reale
       - Interpretazione: I1 (input-conditioning) è il cuore del modello; senza di esso
         T2 degenera a CLAY image-space con gate che va negativo
       - Diagnostico del gate: gate<0 = modello si fida di D (immagine), gate>0 = si fida del testo
         I1-off → gate −0.096 (D inutile), I2-off → gate +0.030 (testo come surrogate)
       - MLP vs T1: ablation cross-attention (R@1=0.060 → 0.082, +37%)
       - T2 no hybrid vs T2 hybrid: ablation text conditioning (R@1=0.067 → 0.090, +34%)
       - Ensemble T1+T2: R@1=0.090 — perché non migliora (wins e losses si bilanciano per query)
[CODE] tabella ablation comparativa
[OUT]  già popolata

[MD]   ## 6. Analisi per-query e discussione
       - Dove T2 batte T1: query negative (−Heavy_Makeup, −Young) e composte con negazione
       - Dove T1 tiene bene: query semplici con interazione positiva
       - CLAY crolla sulle composte perché P1 (no sign): +E,−S → proietta nella stessa direzione
         di −E,+S → il DB ordinato ignora la polarità → top-k random
       - Limiti del nostro approccio: query con pochissimi source (n<50), attributi rari

[MD]   ## 7. Conclusioni
       - Best model T2 hybrid: R@1=0.090, +30x vs CLAY, +4x vs naive
       - Contributo principale: I1 input-conditioned + I3 warm-start + I4 hybrid = generalizzazione
       - Limiti aperti: query con n_sources<50, attributi correlati (Male/No_Makeup)
```

### Punti chiave per il report (motivazioni da includere)

**Perché CLAY collassa (da scrivere in modo preciso):**
- Il subspace P=V_k V_kᵀ proietta sia "with attr" che "without attr" nello stesso spazio
- Non c'è segno: la query `+Smiling` e `-Smiling` producono lo stesso P
- Il Log-map al tangent plane non basta: il modality gap testo-immagine fa sì che le direzioni testuali puntino nel cono sbagliato rispetto alle features visive
- Risultato empirico: R@1=0.003, peggio del naive (0.022) su 12/13 query

**Perché le nostre soluzioni funzionano:**
- **InfoNCE**: loss contrastiva su spazio L2-normalizzato, τ controlla selettività delle direzioni
- **Identity anchor**: previene il collasso delle query verso un punto fisso (ablation: senza → R@1 scende)
- **Hard negatives**: immagini che violano esattamente 1 constraint → polarity più netta nel gradiente
- **Warm-start D (I3)**: probe immagine garantisce partenza semanticamente significativa senza leakage test
- **Ortho_reg correlation-aware (I2)**: evita che attributi correlati (es. Male↔No_Beard) si annullino vicendevolmente
- **Text hybrid (I4)**: assi CLIP text forniscono prior linguistico; gate scalare apprende quanto fidarsi del testo vs immagine
- **Cosine LR non aiuta T2**: fixed LR con eval frequente (ogni 100 steps) trova checkpoint migliori nell'oscillazione; cosine converge troppo presto

---

## Timeline

| giorno | attività |
|---|---|
| 30 giu | ✓ T2 tau05+cosine, T1T2 hybrid |
| 1 lug | ✓ T2 tau05 fixed, ✓ ablation I1/I2/I3 completate |
| **2 lug** | **← oggi: inizia stesura notebook sezioni 1-3 (Setup, Baselines, T1)** |
| 3 lug | Stesura sezioni 4-5 (T2, Ablation), upload Drive |
| 4 lug | Sezioni 6-7 (Analisi, Conclusioni) + revisione + output cells verificati |
| 5 lug | **Consegna** |

---

## Checklist finale notebook

- [ ] Checkpoints caricati su Google Drive e path aggiornato nel notebook
- [ ] `USE_PRETRAINED = True` come default in ogni cella di training
- [ ] Output cells popolate per tutte le celle di eval e training
- [ ] Commenti inline nel codice (architettura, loss, training loop)
- [ ] Tutte le tabelle risultati popolate con numeri finali
- [ ] Ablation table I1/I2/I3/I4 completa con delta R@1
- [ ] Analisi per-query (almeno T1 vs T2 hybrid)
- [ ] Motivazione di ogni scelta (loss, LR, τ, hard negatives, ortho_reg)
- [ ] Confronto vs naive baseline su ogni modello
- [ ] Notebook caricato su Colab e link condiviso
