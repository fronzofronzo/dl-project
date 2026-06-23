# Dynamic & Hybrid Conditioning for Compositional Image Retrieval
## Idee progettuali: superare i limiti di CLAY (soluzione *no-training* e soluzione *training-based*)

> Documento di lavoro. Sintetizza l'analisi dei tre paper (CLAY, GDE/Berasi, Project Assignment V1.2) e propone due famiglie di soluzioni per il modulo di fusione **Φ** richiesto dall'assignment.

---

## 0. Inquadramento del problema

### 0.1 Cosa chiede davvero l'assignment

Dobbiamo costruire un modulo di fusione **Φ** che, dati:

- un'**immagine di riferimento** `v_ref ∈ V` (embedding visivo CLIP),
- un insieme di **vincoli testuali positivi** `T⁺ = {t₁⁺, …, tₙ⁺}` (modifiche additive),
- un insieme di **vincoli testuali negativi** `T⁻ = {t₁⁻, …, tₘ⁻}` (modifiche sottrattive),

produca un **embedding di query composito**

```
v_query = Φ(v_ref, T⁺, T⁻)
```

tale che le immagini recuperate (per cosine similarity contro un database visivo **congelato**):

1. **preservino l'identità centrale** di `v_ref`;
2. **contengano** tutti gli attributi positivi richiesti;
3. **non contengano** nessuno degli attributi negati.

Concettualmente è un task di **Composed Image Retrieval (CIR)** la cui forma ingenua è
`v_target ≈ v_ref + t_pos − t_neg`,
ma fatta in modo geometricamente e semanticamente corretto.

**Vincoli operativi imposti dall'assignment:**

- Modello obbligatorio: **CLIP ViT-B/32** (HuggingFace), **congelato**.
- Dataset: **CelebA** (split di test per la valutazione), 40 attributi binari.
- Database visivo estratto **offline una sola volta** e mantenuto fisso (efficienza, come in CLAY).
- Metriche: **Recall@K** (metrica primaria, hit rate) e **Precision@K** per K = 1, 5, 10.
- Ground-truth: un'immagine è target valido per una query se (a) soddisfa **strettamente** i vincoli +/− e (b) ha **distanza di Hamming ≤ 2** dagli attributi residui di `v_ref` (preserva l'identità).
- **Attenzione tecnica critica:** le chiavi del `ground_truth` JSON sono **indici interi del dataset PyTorch** (`celeba[idx]`), **non** nomi di file. Usare sempre `celeba[idx]`, mai costruire path manualmente.

### 0.2 Cosa fa CLAY (il metodo da superare)

CLAY è un metodo **training-free** di *Conditional Visual Similarity Modulation*. Pipeline:

1. Per ogni condizione `c`, un LLM genera prompt testuali con template `"a photo of {c}"`.
2. I prompt vengono codificati dal text encoder → matrice delle feature testuali `Tᶜ = [t₁ᶜ, …, tₙᶜ]ᵀ ∈ ℝⁿˣᵈ`.
3. Poiché gli embedding CLIP vivono su un'**ipersfera unitaria** `Sᵈ⁻¹` (la norma non porta informazione), si mappano le feature nello **spazio tangente** al *mean* `μ` tramite la **mappa logaritmica** `Log_μ` (geometria manifold-aware).
4. Si esegue **una sola SVD** su queste feature mappate: `Log_μ(Tᶜ) = UΣVᵀ`.
5. Si tengono i **top-k vettori singolari destri** `Vₖ` → matrice di proiezione `Pᶜ = Vₖ Vₖᵀ`.
6. Si proietta ogni feature visiva sul sottospazio (con allineamento dei mean tramite una rotazione `H(·)` per gestire il *cono*/modality gap), poi cosine similarity.

**Multi-condizione in CLAY** = si impilano semplicemente i prompt di più condizioni nella stessa matrice `Tᶜ` *prima* della SVD. Questo è esattamente lo "stacking/concatenation ingenuo pre-SVD" che l'assignment critica come **rigido e non controllabile**.

### 0.3 I 4 limiti di CLAY (dalle note + assignment)

| # | Limite | Causa radice |
|---|--------|--------------|
| **P1** | **Nessun segno / polarità** | `span(Vₖ)` è agnostico alla direzione. `−red hair` e `+red hair` generano lo **stesso** sottospazio. Una persona con i capelli rossi ottiene score alto anche quando l'attributo è **negato**. |
| **P2** | **Nessuna pesatura** | Una singola SVD tiene la direzione col **singular value più grande**. Nessun controllo sull'importanza relativa delle condizioni. |
| **P3** | **Nessuna interazione / gestione conflitti** | Stack→SVD **non isola** i concetti. Condizioni quasi-parallele (`+blond` vs `−red`) diventano righe della stessa matrice, si fondono, si cancellano o una domina. Nessun ragionamento incrociato tra condizioni. |
| **P4** | **Statico, non dinamico** | Il sottospazio è calcolato **dal solo testo**, ignorando la `v_ref` reale. Stessa modifica per qualunque immagine di riferimento. |

**Entrambe** le soluzioni proposte sotto devono eliminare **tutti e quattro** i limiti.

### 0.4 Strumenti geometrici (comuni a CLAY e GDE/Berasi)

- Gli embedding CLIP stanno su `Sᵈ⁻¹`; la norma è priva di significato → si usa la **cosine similarity** e la **geometria sferica**.
- **Mappa logaritmica** `Log_μ(x)`: proietta un punto della sfera sullo **spazio tangente** `T_μ` nel punto di tangenza `μ` (il *mean intrinseco*). Linearizza localmente.
- **Mappa esponenziale** `Exp_μ(v)`: riporta un vettore tangente sulla sfera lungo la geodetica.
- **GDE (Geodesically Decomposable Embeddings, Berasi et al.)**: una composizione di concetti è la **somma di vettori tangenti** al mean intrinseco, seguita da `Exp_μ`. Berasi **dimostra** che l'aritmetica lineare nello spazio ambiente `ℝᵈ` **non basta** (i concetti non sono linearmente decomponibili): bisogna rispettare la geometria del manifold. Ogni direzione primitiva `v_z` è il *mean tangente* di tutte le composizioni che contengono quel concetto (con vincolo di centraggio `Σ v_zᵢ = 0`).
- **Modality gap**: testo e immagini vivono in **coni separati** del manifold. Sommare direttamente `v_ref + t` mischia le modalità in modo problematico → meglio usare **vettori-direzione** (differenze tra prompt) e/o allineare i mean (rotazione `H` come in CLAY).

---

## 1. SOLUZIONE 1 — *No-training*

### 1.1 Idea centrale

**Editing geometrico, consapevole del segno, nello spazio tangente, con pesi dinamici condizionati dall'input.**

Unisce la composizione tangente di **GDE** con la consapevolezza del manifold di **CLAY**, aggiungendo (i) gestione esplicita del **segno**, (ii) **pesi dipendenti dai dati** e (iii) **ortogonalizzazione** tra condizioni. Richiede solo forward pass di CLIP + algebra lineare (Log/Exp, Gram–Schmidt, coseni). È veloce, gira su Colab, mantiene il **DB congelato**.

### 1.2 Ricetta passo-passo

#### (a) Vettori-direzione contrastivi (non testo grezzo) → risolve **P1 (segno)** + modality gap

Per ogni attributo si costruisce un **asse polarizzato** dalla differenza tra prompt antonimi, invece di usare l'embedding di `t⁺` grezzo:

```
d_glasses = Log_μ( t["a photo of a person wearing glasses"] )
          − Log_μ( t["a photo of a person without glasses"] )
```

- Un vincolo **positivo** **somma** la direzione.
- Un vincolo **negativo** la **sottrae**.

La polarità ora vive **nel vettore** (direzione orientata), non viene persa in uno `span` di sottospazio. È l'idea degli *ideal words* / assi testuali (Trager et al. [67], citato in Berasi). L'uso del prompt neutro/antonimo come riferimento cancella anche le componenti non rilevanti (template, parole comuni).

#### (b) Composizione nello spazio tangente al mean μ → manifold-aware (GDE)

GDE dimostra che `v_ref + t⁺ − t⁻` nello spazio ambiente è errato sulla sfera. Si opera invece in `T_μ`:

```
v_edit   = Log_μ(v_ref)  +  Σᵢ wᵢ⁺ · dᵢ⁺  −  Σⱼ wⱼ⁻ · dⱼ⁻
v_target = Exp_μ(v_edit)
```

Il retrieval avviene per cosine similarity tra `v_target` e gli embedding del DB (eventualmente anch'essi proiettati/allineati al mean come in CLAY).

#### (c) Pesi dinamici condizionati dall'input → risolve **P2 (pesatura)** + **P4 (statico/dinamico)**

I pesi `wᵢ` si calcolano **dalla `v_ref` reale**, senza alcun training:

- Vincolo **positivo già soddisfatto** dalla reference → peso basso (aggiungere "occhiali" a chi li ha già spreca budget):
  ```
  wᵢ⁺ ∝ ( 1 − cos(v_ref, dᵢ⁺) )      # spingi di più dove la reference è LONTANA dall'attributo
  ```
- Vincolo **negativo realmente presente** nella reference → peso alto (serve rimuoverlo davvero):
  ```
  wⱼ⁻ ∝ cos(v_ref, dⱼ⁻)              # spingi di più dove l'attributo da negare È presente
  ```

Questo rende l'edit **adattivo per ogni immagine** = esattamente l'"adattamento all'input reale" che P4 richiede e che CLAY non ha.

**Riferimenti collegati:**

- **Dynamic Weighted Combiner for Mixed-Modal Image Retrieval** — Huang et al., AAAI 2024. Peso *non fisso* tra reference e modifica, deciso dall'input: modifica leggera → domina la reference, modifica forte → domina il testo. Stesso principio dei pesi `wᵢ` qui. Differenza: loro pesi **appresi** (training), il nostro punto (c) è **no-training** (coseni puri).
- **Learning with Multi-modal Gradient Attention for Explainable Composed Image Retrieval** — Chen et al., arXiv:2308.16649. Attenzione/pesi condizionati **sia su immagine sia su testo** → supporta il carattere dinamico dipendente da `v_ref`.
- **Heterogeneous Uncertainty-Guided CIR** — arXiv:2601.11393. Dynamic weighting per-campione guidato dall'incertezza.

#### (d) Ortogonalizzazione delle direzioni attive (Gram–Schmidt) → risolve **P3 (interazione/conflitti)**

Prima di sommare, si **ortogonalizza** ogni direzione rispetto alle altre attive, così ognuna contribuisce solo con la sua **componente unica**:

- Coppia in conflitto (`+blond`, `−red hair`): si proietta via l'**asse condiviso del colore dei capelli** → niente doppio conteggio, niente cancellazione silenziosa.
- È proprio l'"isolare i concetti" che una singola SVD su matrice impilata **non può** fare.

Pseudo:
```
D = [d₁, d₂, …]              # direzioni attive (con segno applicato)
D_orth = gram_schmidt(D)     # ciascuna ⟂ alle precedenti
v_edit = Log_μ(v_ref) + Σ wᵢ · D_orth[i]
```

#### (e) Ancora di identità → preserva `v_ref` (requisito assignment)

Si tiene il termine `v_ref` con peso 1 (ancora) e si **limita** `Σ|wᵢ|` (clip della norma dell'edit) → il passo geodetico da `v_ref` resta **limitato**, garantendo la preservazione dell'identità centrale.

#### (f) (Opzionale) Re-ranking negativo a test-time → layer di interazione economico

Dopo il retrieval, si **scarta/declassa** ogni candidato per cui `cos(candidate, d_neg)` supera una soglia (= contiene ancora l'attributo negato). Attacca direttamente il caso di fallimento di P1.

### 1.3 Ablation naturali (gratis)

- ambiente vs spazio tangente;
- testo grezzo vs asse contrastivo;
- pesi statici vs dinamici (punto c);
- con/senza ortogonalizzazione (punto d);
- con/senza re-ranking negativo (punto f).

### 1.4 Variante CLAY-fedele (cambio minimo)

Se si vuole restare vicini al paper: mantenere i **sottospazi per-condizione** ma aggiungere (1) **proiezione con segno** (verso il sottospazio per i positivi, **via dal** sottospazio per i negativi) e (2) **pesi del sottospazio dipendenti dall'input**. È un ponte CLAY→nostro metodo. **Raccomandazione:** preferire la versione ad aritmetica tangente (1.2) — più pulita e con segno nativo.

### 1.5 Rischi e mitigazioni

| Rischio | Mitigazione |
|---|---|
| Scelta del prompt neutro/antonimo per la direzione | testare più template, mediare su un piccolo set di prompt |
| Scala/normalizzazione dei pesi | normalizzare `w` e clippare `‖v_edit − Log_μ(v_ref)‖` |
| Modality gap residuo | rotazione di allineamento dei mean `H(·)` come in CLAY |

### 1.6 Spiegazione intuitiva (in parole semplici)

> Versione "senza matematica" della soluzione no-training. Stessa sostanza di §1.2, raccontata facile.

#### Idea in una frase

Hai una foto di partenza (`v_ref`). Vuoi una foto **simile ma modificata**: aggiungi cose (`+occhiali`), togli cose (`−trucco`). La soluzione no-training **sposta** la foto nello spazio CLIP nella direzione giusta, **senza addestrare niente** — solo matematica + CLIP così com'è.

Pensa agli embedding CLIP come **punti su un mappamondo** (una sfera). Ogni immagine è un punto. Spostare il punto nella direzione "occhiali" = avvicinarsi alle immagini con occhiali.

#### Problema 1 — il SEGNO (il più grave)

- **Problema:** CLAY non distingue `+rosso` da `−rosso`. Costruisce un "tema capelli-rossi" e cerca immagini vicine a quel tema. Ma "togliere rosso" e "mettere rosso" per lui sono lo **stesso tema**. Chiedi `−capelli rossi` e ti tornano comunque rosse.
- **Perché:** CLAY usa un *sottospazio* (tipo una linea). Una linea non ha verso: punta sia avanti che indietro. Perde la freccia.
- **Fix:** invece di un "tema", usiamo una **freccia orientata**:
  ```
  freccia_occhiali = punto("persona CON occhiali") − punto("persona SENZA occhiali")
  ```
  La freccia **punta** dal "senza" al "con".
  - `+occhiali` → **sommi** la freccia (vai verso "con").
  - `−occhiali` → **sottrai** la freccia (vai verso "senza").

  Il segno ora è dentro la freccia, non si perde più. Bonus: la sottrazione cancella la roba inutile ("a photo of a person") e resta solo il concetto puro "occhiali".

#### Problema 2 — il PESO

- **Problema:** CLAY tratta tutte le condizioni uguali. Non puoi dire "questa conta di più". Una condizione dominante schiaccia le altre.
- **Fix:** ogni freccia ha un **peso** `w`, calcolato **guardando la foto di partenza**:
  - La reference **ha già** gli occhiali e chiedi `+occhiali`? → peso basso (inutile spingere dove sei già).
  - La reference **ha** il trucco e chiedi `−trucco`? → peso alto (lì serve spingere forte).

  Misuri quanto la foto è già vicina all'attributo (un coseno) e regoli la spinta.

#### Problema 3 — i CONFLITTI tra condizioni

- **Problema:** due condizioni che si pestano i piedi. Es. `+biondo` e `−rosso`: concetti **quasi uguali** (entrambi = "colore capelli"). CLAY li mette nello stesso calderone, si **mischiano** e si **cancellano**, o una vince a caso.
- **Fix:** **ortogonalizzazione** (Gram–Schmidt). Prima di sommare le frecce, le rendiamo **indipendenti**: togliamo da ogni freccia la parte **in comune** con le altre, teniamo solo la parte **unica**.
  - *Analogia:* due persone spingono un carrello quasi nella stessa direzione → si annullano. Le mettiamo a spingere su **assi separati**, così ogni spinta conta.
  - Per `+biondo / −rosso`: togliamo l'asse condiviso "colore capelli" → biondo va su, rosso va giù, pulito.

#### Problema 4 — STATICO vs DINAMICO

- **Problema:** CLAY decide la modifica **guardando solo il testo**. Stessa modifica per qualunque foto, ignora chi hai davanti.
- **Fix:** già risolto dentro il Problema 2. I pesi dipendono dalla **foto reale** → ogni immagine riceve una modifica **su misura**. Niente training, solo confronti (coseni) tra foto e frecce.

#### Il calcolo completo (geometria)

Gli embedding stanno su una **sfera**, non su un foglio piatto. Sommare frecce a caso sulla sfera sbaglia (lo dimostra GDE). Quindi:

1. **Appiattisci** localmente: porti foto e frecce nello spazio tangente (mappa `Log`). Come spianare un pezzetto di mappamondo su un tavolo.
2. **Sommi** lì le frecce (con segno, peso, ortogonalizzate):
   ```
   v_edit = Log(v_ref) + Σ wᵢ⁺·frecciaᵢ⁺ − Σ wⱼ⁻·frecciaⱼ⁻
   ```
3. **Rimetti sulla sfera** (mappa `Exp`) → ottieni `v_target`.
4. Cerchi nel database le immagini più vicine a `v_target`.

**Ancora di identità:** tieni `v_ref` come base e **limiti** quanto ti allontani → resta "la stessa persona", solo modificata (requisito assignment).

**Pulizia finale (opzionale):** dopo la ricerca, se un risultato contiene ancora l'attributo negato (controllo col coseno sulla freccia negativa), lo **scarti**. Rete di sicurezza per il segno.

#### Riassunto semplice

| Problema CLAY | In parole povere | Fix no-training |
|---|---|---|
| P1 segno | confonde "metti" e "togli" | freccia orientata (prompt CON − prompt SENZA) |
| P2 peso | tutte le condizioni uguali | peso da quanto la foto è già vicina |
| P3 conflitto | condizioni simili si cancellano | ortogonalizzazione (spinte indipendenti) |
| P4 statico | ignora la foto reale | pesi calcolati sulla foto stessa |

**Perché è bello:** zero training, solo CLIP + algebra. Veloce, gira su Colab, prototipabile in 1–2 giorni. Diventa anche il baseline per confrontare la soluzione con training.

---

## 2. SOLUZIONE 2 — *Training-based*

### 2.1 Idea centrale

**Adapter di fusione leggero a cross-attention condizionale, addestrato con loss contrastiva di retrieval.**

CLIP resta **congelato**; si addestra **solo** Φ (modulo piccolo). È il percorso "cross-attention / gating / proiezione non-lineare" suggerito dall'assignment. Risolve tutti e 4 i limiti *by design*.

### 2.2 Architettura di Φ

- **Input:** `v_ref` + un **insieme** di token-condizione **con segno** `(tᵢ, sᵢ)`, con `sᵢ ∈ {+1, −1}` codificato come **embedding di polarità appreso** → **segno esplicito (P1)**.
- **Blocco di cross-attention:**
  - *query* = `v_ref` (o un token query apprendibile),
  - *key/value* = le condizioni con segno.
  - L'attention guarda **tutte le condizioni insieme e in relazione alla reference** → gestisce **interazione/conflitti (P3)** ed è **dinamica/condizionata dall'input (P4)** per costruzione (i pesi dipendono sia da `v_ref` sia dall'insieme di condizioni).
  - Input a **insieme** → numero variabile di condizioni gestito naturalmente, **permutation-invariant**.
- **Gating (residuo stile FiLM):**
  ```
  v_q = v_ref + g(v_ref, conds) ⊙ Δ
  ```
  dove `Δ` è la combinazione attesa delle condizioni e `g` è un gate appreso → **pesatura dinamica (P2)**. Il **residuo su `v_ref`** preserva l'identità.
- **Geometria:** eseguire il residuo nello **spazio tangente** (`Log_μ` prima, `Exp_μ` dopo) per restare coerenti col manifold (si porta dentro l'insight GDE/CLAY).
- **Output:** `v_q` L2-normalizzato, confrontato con il **DB congelato** via coseno. **Fusione solo lato query → nessun re-encoding del DB** (si conserva il vantaggio di efficienza di CLAY).

### 2.3 Addestramento (senza etichette umane)

- **Auto-supervisione** sugli attributi binari di CelebA (split di **train**). Si campiona una reference + un insieme casuale di vincoli +/−.
  - **Positivi** = immagini che soddisfano i vincoli con Hamming ≤ 2 dalla reference (la **regola di GT dell'assignment stesso**).
  - **Negativi** = immagini che violano i vincoli.
- **Loss contrastiva (InfoNCE):** avvicina `v_q` ai target positivi, allontana dai negativi.
- **Hard negatives:** negativi che violano **solo** l'attributo negato (es. per `−red hair`: il negativo più difficile = persona coi capelli rossi ma per il resto coincidente). Forza il rispetto della **polarità** → elimina il fallimento di **P1**.
- **Loss ausiliarie** (anche materiale per ablation):
  - *identity anchor:* `‖v_q − v_ref‖` (o coseno) per tenere l'identità centrale;
  - *negative repulsion:* spinge `v_q` **via** dalla direzione testuale dell'attributo negato;
  - *decomposability reg (GDE):* incoraggia i `Δ` attesi a essere geodeticamente decomponibili → disentangle dei concetti.

### 2.4 Perché è leggero e compatibile coi vincoli

- Si addestra solo Φ (pochi layer di attention + MLP) → CLIP congelato, iterazione rapida, sta su GPU Colab.
- DB visivo congelato preservato (fusione lato query) → coerente con l'efficienza richiesta.

### 2.5 Ablation (richieste dalla Fase 5 del piano)

- capacità della rete (numero di head / layer / dimensione);
- scelta dell'optimizer;
- iperparametri (temperatura, pesi delle loss);
- strategia di sampling (numero di condizioni per campione, rapporto hard-negative);
- con/senza embedding di segno;
- cross-attention vs gating semplice vs MLP;
- spazio tangente vs ambiente.

---

## 3. Mappa Problema → Soluzione

| Limite CLAY | Fix *no-training* | Fix *training* |
|---|---|---|
| **P1** segno | vettori-direzione contrastivi (±) + re-rank negativo | embedding di token con segno + hard-negative mining |
| **P2** pesatura | pesi `wᵢ` da `cos(v_ref, dᵢ)` (relevance gating) | gate appreso (FiLM) |
| **P3** interazione | ortogonalizzazione Gram–Schmidt delle direzioni | cross-attention sull'insieme di condizioni |
| **P4** dinamico | pesi calcolati dalla `v_ref` reale | attention condizionata su `v_ref` |

---

## 4. Raccomandazione e integrazione col piano

Eseguire **entrambe** le soluzioni in sequenza — coerente con `pianificazione_progetto.md`:

- **Fase 3 — baseline:** aritmetica ingenua nello spazio ambiente (`v_target ≈ v_ref + t_pos − t_neg`), già pianificata → *lower bound*.
- **Fase 3.5 — secondo baseline (aggiunta consigliata):** reimplementare la **SVD impilata di CLAY** come secondo baseline → permette di dimostrare di battere il **vero SOTA**, non solo la versione ingenua. Forte per il criterio di **rigore metodologico**.
- **Fase 4 — metodo:**
  1. prima **Soluzione 1** (1–2 giorni, niente training, valida l'idea a basso costo);
  2. poi **Soluzione 2** (i punti di **originalità/creatività**).
  La Soluzione 1 funge anche da riferimento di ablation per la Soluzione 2.
- **Query da vincere** (dove P1/P3 rompono CLAY e il nostro metodo brilla): le **composte e conflittuali** —
  `+Eyeglasses & −Smiling`, `−Male & −Mustache`, `−Smiling & +Eyeglasses & +Wearing Hat`, `+Wearing Lipstick & −Heavy Makeup & +Smiling`.

---

## 5. Riferimenti rapidi

- **CLAY** — Lim et al., *Conditional Visual Similarity Modulation in VLM Embedding Space*, CVPR 2026. (Metodo training-free da superare; SVD su sottospazio testuale manifold-aware.)
- **GDE** — Berasi et al., *Not Only Text: Exploring Compositionality of Visual Representations in VLMs*, CVPR 2025. (Composizione geodetica nello spazio tangente; l'aritmetica lineare ambiente è insufficiente.)
- **Project Assignment V1.2** — *Dynamic and Hybrid Conditioning for Compositional Image Retrieval*, Deep Learning 2026. (Specifica del task, GT a Hamming ≤ 2, metriche R@K / P@K, vincolo CLIP ViT-B/32 + CelebA.)
- Trager et al. [67] — *Linear spaces of meanings* (ideal words / assi testuali).
