# Prossimi passi — Due soluzioni con training per la fusione Φ

> Documento operativo per la divisione del lavoro tra i 2 membri del team.
> **Consegna: 5 luglio 2026.** Oggi: 22 giugno. Abbiamo ~13 giorni.

---

## 0. Dove siamo adesso (riassunto onesto)

La **Soluzione A (no-training)** è stata sviluppata e diagnosticata a fondo
(`results/findings_2026-06-19.md`). Cosa abbiamo imparato:

- **Il metodo migliore attuale** è l'**asse contrastivo in spazio ambiente**, con
  `α ≈ 4`. Risultati su CLIP ViT-B/32 frozen, CelebA test (DB = 19.962 immagini):

  | metodo | R@1 | R@5 | R@10 |
  |---|---|---|---|
  | no edit (solo v_ref) | 0.019 | 0.053 | 0.076 |
  | naive ambient | 0.022 | 0.072 | — |
  | **contrastive ambient, α=4** | **0.034** | **0.103** | **0.156** |

  L'editing funziona: quasi raddoppia R@1 rispetto al non fare niente.

- **Tre fallimenti dimostrati** (sono questi che il training deve attaccare):
  1. **Modality gap.** `cos(immagine, direzione_testo) ≈ 0` per tutte le immagini.
     Testo e immagini vivono in coni separati → i pesi dinamici calcolati col coseno
     testo-immagine sono **rotti** (costanti, niente dinamicità).
  2. **Direzioni CLIP non disentangled.** La direzione "Heavy_Makeup" è
     fortissimamente correlata col genere femminile. Il re-rank negativo declassa
     i target buoni quanto i cattivi → funziona solo su attributi separabili (Young),
     danneggia quelli entangled (Heavy_Makeup, Wavy_Hair).
  3. **Polarità (segno).** Senza una direzione orientata, "metti" e "togli" si
     confondono.

- **Spazio tangente ≈ spazio ambiente:** niente guadagno misurabile. Da riportare
  come ablation negativa (è rigore, non sconfitta).

- **Tetto basso.** Con ViT-B/32 frozen il massimo ottenibile è limitato. L'obiettivo
  realistico **non** è un numero enorme, ma **battere in modo pulito** i due baseline
  (naive + CLAY-SVD) e la Soluzione A. Aspettativa onesta: R@5 ≈ 0.13–0.18.

---

## 1. Vincoli non negoziabili (dall'assignment)

- **CLIP ViT-B/32 frozen.** Si addestra **solo** il modulo Φ. Altri modelli solo come
  ablation extra, e bisogna comunque riportare i risultati ViT-B/32.
- **Database visivo frozen**, estratto offline una volta sola. La fusione è
  **solo lato query** — mai ri-codificare il DB quando cambiano le condizioni.
- **Eval solo sugli indici source presenti nel JSON** per quella query.
- **`celeba[idx]`**, mai costruire i path dei file a mano.
- L2-normalizzare sempre prima del coseno.

---

## 2. Strategia: due soluzioni con training, complementari

Entrambe tengono CLIP frozen e addestrano solo Φ (modulo piccolo, lato query).
Sono **architetture diverse** così i due membri lavorano in parallelo senza
toccare gli stessi file. Entrambe devono risolvere i 4 limiti di CLAY
(segno P1, pesatura P2, interazione P3, dinamico P4) e i 3 fallimenti misurati sopra.

| | **T1 — Cross-Attention** | **T2 — Direzioni apprese** |
|---|---|---|
| paradigma | set-transformer / attenzione | dizionario di direzioni + gate |
| punto di forza | modella interazioni/conflitti | disentanglement + interpretabilità |
| parametri | più grande | piccolissimo |
| si abbina a | Soluzione B dello spec | Soluzione A no-training (ablation pulita) |
| rischio | overfit, più lento | meno espressiva su query a 3 condizioni |

Il report finale racconta la progressione:
`naive → CLAY-SVD → no-train A → T1 & T2 → confronto diretto`. Forte sul **rigore**.

---

## 3. SPINA DORSALE CONDIVISA (costruire INSIEME, giorni 0–2)

⚠️ **Da fare prima di dividersi.** Entrambe le soluzioni usano questi pezzi.
Costruirli una volta sola, niente lavoro doppio.

### 3.1 Feature CLIP dello split TRAIN
Abbiamo solo `data/clip_features_test.pt` (il DB di valutazione). Il training ha
bisogno delle feature delle immagini di **train**, estratte offline e congelate
(stesso identico procedimento del DB).
- 162k immagini = pesante su Colab. Se la GPU è stretta, usare un **subset di
  30–50k** reference: per addestrare un adapter è più che sufficiente.
- Output: `data/clip_features_train.pt`.
- **È il prerequisito che blocca tutto: iniziare subito (giorno 0).**

### 3.2 Sampler self-supervised (il motore dati, riusato da entrambe)
Genera gli esempi di training dalle label binarie di CelebA train (nessuna etichetta
umana in più):
- campiona una reference `v_ref` dal train;
- campiona un set casuale di vincoli ± (1–3 condizioni, mix di semplici e composte);
- **positivi** = immagini train che soddisfano **strettamente** i vincoli **E** hanno
  Hamming ≤ 2 sugli attributi residui (è esattamente la regola di GT dell'assignment);
- **hard negative** = immagini che violano **solo** l'attributo negato (es. per
  `−red hair`: persona coi capelli rossi ma per il resto uguale) → forza la polarità,
  risolve P1;
- **easy negative** = il resto del batch (in-batch negatives).

### 3.3 Loss InfoNCE
Modulo contrastivo condiviso con temperatura `τ`. Avvicina `v_q` ai positivi,
allontana da hard + in-batch negatives.

### 3.4 Harness di valutazione
Già pronto: `src/metrics.py`, `src/groundtruth.py`, `src/retrieval.py`. **Riusarlo
intatto.** Niente leakage: Φ si addestra sul TRAIN, si valuta sui source del JSON di
TEST.

### 3.5 Baseline CLAY-SVD (ancora da implementare)
È il "vero SOTA" da battere e nelle findings risulta **non ancora fatto**. Serve per
l'argomento di rigore. Assegnarlo a chi finisce prima la spina dorsale, oppure
inserirlo nei giorni 0–2.
- Pipeline: impila i prompt delle condizioni → Log map → **una sola SVD** →
  top-k vettori singolari (`k=50`) → matrice di proiezione `Pᶜ = VₖVₖᵀ` → proietta le
  feature visive → coseno.

---

## 4. SOLUZIONE T1 — Adapter a Cross-Attention condizionale
**Responsabile: Persona 1.** È la carta "originalità/creatività". Corrisponde alla
Soluzione B dello spec.

### Idea in una frase
Una piccola rete di attenzione guarda **tutte le condizioni insieme** e in relazione
alla foto di partenza, e produce una query modificata. Gestisce interazioni e pesi
in modo automatico.

### Architettura di Φ
- **Token-condizione con segno**: una tabella di embedding appresi per i 40 attributi
  (40 × d) **più** un embedding di polarità appreso per {+1, −1}.
  `token_i = attr_emb[i] + sign_emb[s_i]`. Il segno è esplicito → risolve **P1**.
- **Blocco di cross-attention**: la *query* è `v_ref` (proiettata in un token query
  apprendibile), *key/value* sono il set di condizioni con segno. L'attenzione vede
  tutte le condizioni insieme e rispetto alla reference → gestisce
  **interazione/conflitto (P3)** ed è **dinamica/condizionata dall'input (P4)** per
  costruzione. Input a insieme → numero di condizioni variabile,
  **permutation-invariant**.
- **Gate residuo stile FiLM**: `v_q = v_ref + g(v_ref, conds) ⊙ Δ`, dove `Δ` è la
  combinazione attesa delle condizioni e `g` è un gate appreso → **pesatura dinamica
  (P2)**. Il residuo su `v_ref` preserva l'identità.
- **Output**: `v_q` L2-normalizzato, coseno contro il DB frozen.

### Esempio pratico passo-passo
**Scenario:** reference = foto di una **donna sorridente senza occhiali**.
Query = `+Eyeglasses, −Smiling` (voglio la stessa identità ma con occhiali e seria).

1. **Input.** `v_ref` = embedding immagine CLIP (vettore 512-dim). Le 2 condizioni
   diventano token:
   - `token_1 = attr_emb["Eyeglasses"] + sign_emb[+1]`
   - `token_2 = attr_emb["Smiling"]   + sign_emb[−1]`
2. **Cross-attention.** La query (`v_ref`) "guarda" i 2 token. La rete, addestrata,
   ha imparato a pesarli **in base alla foto**:
   - `v_ref` NON ha occhiali → presta molta attenzione a `+Eyeglasses` (modifica forte).
   - `v_ref` STA sorridendo → presta molta attenzione a `−Smiling` (va davvero rimosso).
   - Se invece la ref avesse già gli occhiali, l'attenzione su `+Eyeglasses` sarebbe
     bassa **da sola** — nessuna regola scritta a mano.
   Risultato: un vettore di modifica `Δ` che combina le due richieste.
3. **Gate FiLM.** `g(v_ref, conds)` decide quanto applicare: `v_q = v_ref + g ⊙ Δ`.
   Il `+ v_ref` tiene l'identità (stessa persona), `g` dosa la spinta.
4. **Retrieval.** Normalizzo `v_q`, coseno contro il DB congelato → tornano foto di
   **persone serie con occhiali** simili all'identità di partenza.

**Conflitto (perché l'attention vince):** query `+Eyeglasses & −Smiling` su una foto
dove i due tratti sono correlati nei dati. L'attention vede entrambi i token
**insieme** e impara a non far cancellare una richiesta dall'altra — cosa che la
singola SVD di CLAY non può fare.

### Loss
- InfoNCE (positivi vs hard + in-batch negatives).
- ausiliarie: ancora di identità `‖v_q − v_ref‖`; repulsione negativa (spinge `v_q`
  via dalla direzione dell'attributo negato).

### Ablation (già pronte per il report)
numero di head/layer/dimensione · sign-embedding sì/no · cross-attention vs MLP
semplice · temperatura `τ` · rapporto hard-negative · residuo tangente vs ambiente.

---

## 5. SOLUZIONE T2 — Direzioni apprese in spazio immagine + gate dinamico
**Responsabile: Persona 2.** È la carta "rigore + sistemiamo i bug documentati".
Più leggera, interpretabile, ed è il **partner di ablation diretto della Soluzione A
no-training**.

### Idea in una frase
La Soluzione A no-training falliva per 3 motivi misurati (direzioni testo nel cono
sbagliato, pesi col coseno quasi costanti, direzioni CLIP entangled). T2 **impara**
direttamente le direzioni e i pesi giusti, eliminando quei 3 problemi.

### Architettura di Φ
- **Dizionario di direzioni apprese** `D ∈ ℝ^{40×512}` — le direzioni vivono in
  **spazio immagine** e si imparano dal self-supervision su CelebA (non dai prompt
  testuali). Risolve modality gap + entanglement per costruzione: le direzioni sono
  ottimizzate per il retrieval, e si può aggiungere un **regolarizzatore di
  ortogonalità** per forzare il disentanglement (è ciò che la Gram–Schmidt provava a
  fare a mano).
- **Segno**: un vincolo positivo somma `d_i`, uno negativo lo sottrae. La polarità è
  nel vettore (**P1**).
- **Testa dei pesi dinamici**: un piccolo MLP `w = f(v_ref, attr_id)` predice il peso
  di ogni condizione dalla `v_ref` reale (**P2 + P4**) — è la versione **appresa** del
  `cos(v_ref, d)` che in spazio immagine era rotto.
- **Composizione**: `v_q = normalize(v_ref + Σ w_i · s_i · d_i)`. Variante in spazio
  tangente disponibile come ablation.
- (opzionale) **mappa di gap-bridge** `W` se si vuole inizializzare `D` dagli assi
  testuali e poi rifinirla.

### Esempio pratico passo-passo
**Stessa scena:** reference = **donna sorridente senza occhiali**, query
`+Eyeglasses, −Smiling`.

1. **Direzioni dal dizionario.** Pesco 2 righe della tabella appresa:
   `D["Eyeglasses"]` e `D["Smiling"]` — due vettori 512-dim **in spazio immagine**
   (imparati, non costruiti dai prompt → niente modality gap).
2. **Pesi dinamici (MLP).** `f(v_ref, "Eyeglasses") → w_glasses` alto (la ref non ha
   occhiali, serve spingere). `f(v_ref, "Smiling") → w_smile` alto (la ref sorride,
   serve rimuovere). Se la ref avesse già gli occhiali, `w_glasses` uscirebbe basso.
3. **Composizione (con segno).**
   ```
   v_q = normalize( v_ref + w_glasses · D["Eyeglasses"] − w_smile · D["Smiling"] )
   ```
   `+` per il positivo, `−` per il negativo: la polarità è esplicita.
4. **Retrieval.** Coseno di `v_q` contro il DB congelato.

**Numeri-giocattolo (per capire, dimensione 3 invece di 512):**
```
v_ref              = [0.9, 0.1, 0.2]
D["Eyeglasses"]    = [0.0, 1.0, 0.0]   w_glasses = 0.8
D["Smiling"]       = [0.0, 0.0, 1.0]   w_smile   = 0.7
v_q (pre-norm) = v_ref + 0.8·[0,1,0] − 0.7·[0,0,1] = [0.9, 0.9, −0.5]
v_q = normalize([0.9, 0.9, −0.5])
```
La componente "occhiali" sale, quella "sorriso" scende → il vettore punta verso
foto serie+occhiali, restando vicino all'identità di `v_ref`.

**Ortogonalità (perché serve):** `D["Blond_Hair"]` e `D["Black_Hair"]` sono lo stesso
asse "colore capelli" con verso opposto. Il regolarizzatore le tiene separate così
`+Blond & −Black` non si cancellano (è la Gram–Schmidt della Soluzione A, ma
**appresa**).

**Differenza chiave vs Soluzione A no-training:** lì `D` veniva dal testo (cono
sbagliato) e i pesi da `cos(v_ref, d)` (≈ costanti, rotti). Qui **entrambi appresi in
spazio immagine** → i 2 bug misurati spariscono.

### Loss
InfoNCE + regolarizzatore di ortogonalità su `D` (decomponibilità) + ancora di
identità.

### Trucco pratico
Inizializzare `D` dagli assi contrastivi già esistenti (`build_direction_axes` in
`src/baselines.py`) → warm start, convergenza veloce.

### Ablation
direzioni apprese vs assi testuali (**vs Soluzione A no-training!**) · pesi statici vs
appresi · regolarizzatore di ortogonalità sì/no · tangente vs ambiente · numero di
parametri.

---

## 5.5 Come funziona il training (esempio di un batch)

Vale per **entrambe** T1 e T2 — cambia solo il modulo Φ in mezzo. Nessuna etichetta
umana extra: si usano le 40 label binarie di CelebA train.

**Costruzione di un esempio:**
1. Pesco una reference dal train, es. immagine #X = `donna, sorride, niente occhiali,
   capelli scuri, …`.
2. Genero una query casuale di vincoli, es. `+Eyeglasses, −Smiling`.
3. **Positivi** = immagini del train che (a) soddisfano i vincoli (occhiali sì,
   sorriso no) **e** (b) differiscono da #X per **≤ 2** altri attributi (Hamming ≤ 2 →
   stessa identità di base). Es. `donna, seria, occhiali, capelli scuri`.
4. **Hard negative** = viola **solo** il vincolo negato: `donna, occhiali, …, ma
   ANCORA sorride`. Serve a insegnare la polarità (se Φ ignora il segno, sbaglia qui).
5. **Easy negative** = le altre immagini del batch.

**Passo di training:**
```
v_q = Φ(v_ref, [(+Eyeglasses), (−Smiling)])
# InfoNCE: avvicina v_q ai POSITIVI, allontana da hard + easy negatives
loss = InfoNCE(v_q, positives, negatives) / τ
      + λ_id · ‖v_q − v_ref‖        # ancora identità
      ( + λ_orth · ortogonalità(D)   solo T2 )
loss.backward();  optimizer.step()   # aggiorna SOLO Φ, CLIP resta frozen
```

Ripetuto su molti (ref, query) campionati a caso → Φ impara a produrre la query
giusta per **qualunque** combinazione di vincoli, dosata sulla reference.

**Cosa NON si addestra:** CLIP (frozen) e il DB visivo (frozen). Si tocca solo Φ
(poche centinaia di migliaia di parametri) → gira su GPU Colab in minuti/poche ore.

---

## 6. Perché due e non una

- **Indipendenza**: dopo la spina dorsale i due non toccano mai gli stessi file.
- **Complementarità**: T1 eccelle sulle query composte/conflittuali (attenzione),
  T2 eccelle sul disentanglement ed è interpretabile.
- **Coppia di ablation**: T2 è il gemello-con-training della Soluzione A → racconto
  perfetto "cosa cambia quando addestriamo le stesse idee".

---

## 7. Calendario (22 giugno → 5 luglio)

| Giorni | Chi | Cosa |
|---|---|---|
| **0–2** | insieme | feature TRAIN + sampler + InfoNCE + (CLAY-SVD baseline) ← **blocca tutto, partire ora** |
| **3–9** | parallelo | P1 costruisce e addestra T1 · P2 costruisce e addestra T2 (sync giornaliero sul sampler) |
| **9–11** | ciascuno | ablation + esempi qualitativi (successi e fallimenti) |
| **12–13** | insieme | flatten in unico notebook Colab, run end-to-end da zero, report (matematica + tabelle), buffer |

---

## 8. Rischi e flag onesti

- **Il tetto è reale.** ViT-B/32 frozen + GT a Hamming ≤ 2 → aspettarsi R@5 ≈ 0.13–0.18,
  non miracoli. La vittoria è **battere CLAY-SVD + contrastive in modo pulito**, non il
  numero assoluto. Inquadrarlo così nel report.
- **L'estrazione delle feature TRAIN è il rischio principale di calendario.** Se 162k
  immagini sono troppe per Colab, subset a ~40k reference: va benissimo per addestrare
  un adapter.
- **CLAY-SVD non è ancora implementato** ed è il vero SOTA da battere: serve per
  l'argomento di rigore.
- **Fix leakage probe** (segnalato nelle findings): le direzioni immagine vanno
  ricostruite dallo split **train**, non test.
- Altri modelli (ViT-L/14, OpenCLIP) solo come **ablation extra**: ViT-B/32 resta il
  risultato primario da riportare.

---

## 9. Mappa Problema CLAY → Fix con training

| Limite CLAY | Fix in T1 (cross-attn) | Fix in T2 (direzioni apprese) |
|---|---|---|
| **P1** segno | embedding di polarità appreso + hard-negative mining | direzione orientata (somma/sottrae) + hard-negative |
| **P2** pesatura | gate FiLM appreso | MLP dei pesi condizionato su `v_ref` |
| **P3** interazione | cross-attention sull'insieme di condizioni | regolarizzatore di ortogonalità su `D` |
| **P4** dinamico | attention condizionata su `v_ref` | pesi calcolati dalla `v_ref` reale |
</content>
</invoke>
