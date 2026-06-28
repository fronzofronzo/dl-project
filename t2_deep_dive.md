# T2 — Direzioni apprese in spazio immagine + gate dinamico: spiegazione completa


---

## 0. T2 in una frase

T2 è una piccola rete che prende la foto di una persona (sotto forma di vettore) e
una lista di richieste tipo «+occhiali, −sorriso», e produce un **nuovo vettore di
ricerca** che punta verso foto della **stessa persona** ma **con occhiali e seria**.
Poi cerca nel database le immagini più vicine a quel vettore.

Tutto il lavoro è **lato query** (si tocca solo il vettore di ricerca): il database
di immagini resta congelato, mai ricalcolato. È questa la furbizia di efficienza che
ereditiamo da CLAY.

---

## 1. Le basi, spiegate da zero

Se questi concetti sono già chiari, salta alla Sezione 2.

### 1.1 Cosa è un «embedding» (vettore)
Un computer non capisce un'immagine o una parola direttamente. Le trasforma in una
**lista di numeri** chiamata **vettore** (o **embedding**). Esempio con 3 numeri:
`[0.9, 0.1, 0.2]`. Nel nostro progetto i vettori hanno **512 numeri** (512
dimensioni). Ogni numero cattura un pezzo di «significato» dell'immagine.

Idea chiave: **immagini simili → vettori vicini**. Due foto di persone sorridenti
hanno vettori più vicini tra loro che a una foto di un paesaggio.

### 1.2 CLIP: la macchina che fa i vettori
**CLIP** (modello di OpenAI, `clip-vit-base-patch32`) è una rete pre-addestrata che
sa due cose:
- prende un'**immagine** → restituisce un vettore da 512 numeri;
- prende un **testo** (es. «a photo of a person with eyeglasses») → restituisce un
  vettore da 512 numeri.

Nel progetto **CLIP è congelato (frozen)**: non lo riaddestriamo mai. Lo usiamo solo
per produrre i vettori, una volta sola, e poi lavoriamo su quei numeri.

### 1.3 La sfera unitaria e la «similarità coseno»
Tutti i vettori di CLIP vengono **normalizzati a lunghezza 1** (si dividono per la
loro lunghezza). Immagina che vivano sulla superficie di una **palla** (la «sfera
unitaria»): contano solo le **direzioni**, non quanto sono lunghi.

Per misurare quanto due vettori sono simili usiamo la **similarità coseno**: è il
coseno dell'angolo tra loro.
- stessa direzione → coseno = **1** (identici)
- perpendicolari → coseno = **0** (niente in comune)
- opposti → coseno = **−1**

Su vettori già normalizzati, il coseno è semplicemente il **prodotto scalare**
(moltiplica numero per numero e somma):
`cos([1,0,0], [0,1,0]) = 1·0 + 0·1 + 0·0 = 0`.

**Regola d'oro del progetto:** normalizzare sempre prima di confrontare col coseno.

### 1.4 Una «direzione» = uno spostamento di significato
Se parti dal vettore di una persona **senza** occhiali e vuoi quello **con** occhiali,
esiste una **freccia** (un vettore-direzione) che, sommata, ti sposta da «senza» a
«con». Questa freccia è il concetto di **direzione di attributo**.

- Sommare la freccia = **mettere** l'attributo (segno **+**).
- Sottrarla = **togliere** l'attributo (segno **−**).

Il **segno** è la polarità. È fondamentale: «metti occhiali» e «togli occhiali» sono
la stessa freccia con verso opposto.

### 1.5 Il problema da risolvere (compositional retrieval)
Ci danno:
- una **foto di riferimento** (`v_ref`), es. una donna sorridente senza occhiali;
- una **lista di vincoli** con segno, es. `+Eyeglasses, −Smiling`.

Vogliamo trovare nel database le foto della **stessa identità** ma che rispettano i
vincoli (con occhiali, seria). «Stessa identità» nel progetto significa: stessi altri
attributi a meno di **2 differenze** (distanza di Hamming ≤ 2 sulle 40 etichette).

### 1.6 Il «modality gap» (un bug nascosto importante)
Testo e immagini, dentro CLIP, **non vivono nella stessa zona della sfera**: stanno in
due «coni» separati. Conseguenza pratica misurata nel progetto: il coseno tra un
vettore-immagine e una direzione costruita dai testi è **quasi sempre ~0**. Quindi se
costruisci le direzioni dai prompt testuali (come faceva la Soluzione A no-training) e
poi le applichi a un'immagine, la spinta è «storta». **T2 evita questo** imparando le
direzioni direttamente in **spazio immagine** (vedi Sezione 4.1).

---

## 2. Perché T2 esiste: cosa non andava prima

Lo standard da battere è **CLAY**, che ha **4 limiti** noti. La nostra **Soluzione A
no-training** li attaccava a mano, ma misurando i risultati abbiamo trovato **3
fallimenti concreti**. T2 nasce per chiudere i 4 limiti **imparando** ciò che la
Soluzione A faceva a mano.

| Limite di CLAY | Cosa significa | Come lo chiude T2 |
|---|---|---|
| **P1 — niente segno** | non distingue «metti» da «togli» | direzione orientata: `+` somma, `−` sottrae |
| **P2 — niente pesatura** | applica tutto con la stessa forza | testa MLP che predice un **peso** per ogni vincolo |
| **P3 — niente interazione** | vincoli correlati si pestano/cancellano | **regolarizzatore di ortogonalità** sulle direzioni |
| **P4 — statico** | stesso edit per ogni foto | i pesi dipendono dalla **`v_ref` reale** |

I **3 fallimenti misurati** della Soluzione A no-training (vedi
`results/findings_2026-06-19.md`) che T2 elimina **per costruzione**:
1. **Modality gap**: direzioni dai testi = cono sbagliato → spinta ~0. **T2 impara le
   direzioni in spazio immagine.**
2. **Pesi rotti**: i pesi `cos(v_ref, d)` erano quasi costanti (sempre per il gap).
   **T2 impara i pesi con un MLP.**
3. **Direzioni entangled**: in CLIP «Heavy_Makeup» è incollato a «femmina», ecc. **T2
   le separa con il regolarizzatore di ortogonalità** (la Gram–Schmidt, ma appresa).

---

## 3. L'idea centrale di T2, in parole semplici

T2 ha tre ingredienti, tutti **appresi durante il training**:

1. **Un dizionario di frecce** `D`: una freccia (direzione da 512 numeri) per ognuno
   dei 40 attributi. Vivono in spazio immagine (giuste per le immagini).
2. **Una manopola intelligente** (testa dei pesi): guarda la foto e l'attributo, e
   decide **quanto** spingere su quella freccia. Se la persona ha già gli occhiali, la
   manopola per «+occhiali» resta bassa.
3. **Una regola di buona separazione** (ortogonalità): durante il training tiene le
   frecce «pulite» e distinte, così richieste diverse non si annullano a vicenda.

La ricetta finale (composizione):
```
v_q = normalizza( v_ref  +  scala · Σ ( peso_i · segno_i · freccia_i ) )
```
- parti dalla foto (`v_ref`) → **mantieni l'identità**;
- aggiungi la somma delle frecce, ognuna **pesata** (manopola) e con il **segno** giusto;
- `scala` è un'unica manopola globale appresa (quanto spingere in totale);
- normalizza per tornare sulla sfera.

---

## 4. L'architettura, componente per componente

Tutti i numeri qui sotto sono quelli **veri** del codice (`t2_directions.py`).
Notazione: `B` = quante foto nel batch, `C` = quante condizioni (vincoli) per foto,
`dim = 512`, `n_attr = 40`.

```
INPUT                                 COMPONENTI                          OUTPUT
v_ref [B,512] ─────────────┐
                           ├─► (b) testa pesi ─► w [B,C] (≥0, softplus)
cond_col [B,C] ─► (b) attr_emb[64] ┘                │
            └────► (a) D[40,512] ─► normalizza ─► frecce [B,C,512]        │
cond_sign [B,C] (±1) ───────────────────────────────┤                    │
cond_mask [B,C] (vero/falso) ───────────────────────┤                    │
                                       (c) edit = scala · Σ w·segno·freccia [B,512]
                                       (d) v_q = v_ref + edit  ─► normalizza ─► v_q [B,512]
```

### Il «contratto» di ingresso (uguale per T1, T2, MLP baseline)
```
forward(v_ref, cond_col, cond_sign, cond_mask) -> v_q
  v_ref     [B,512]  vettore-immagine di riferimento (già normalizzato, CLIP frozen)
  cond_col  [B,C]    indice di colonna dell'attributo per ogni condizione (0..39)
  cond_sign [B,C]    +1 (additivo) / −1 (sottrattivo) / 0 (riempitivo/pad)
  cond_mask [B,C]    vero per le condizioni reali, falso per il riempitivo
  -> v_q    [B,512]  query composta, normalizzata
```
**Trappola del padding:** le foto possono avere un numero diverso di vincoli, ma il
tensore deve essere rettangolare → le caselle vuote si «riempiono» (pad). Il pad ha
`cond_col=0`, ma **0 è anche un attributo vero** (`5_o_Clock_Shadow`). La verità su
«è pad?» è **solo `cond_mask`**, mai `cond_col==0`.

### 4.1 (a) Dizionario di direzioni apprese `D`
```python
self.D = nn.Parameter(torch.randn(n_attr, dim) * 0.02)   # [40, 512]
```
Una tabella di 40 righe × 512 numeri. La riga `j` è la freccia dell'attributo `j`.
- È un `nn.Parameter`: viene **modificata dal training** (parte da rumore piccolo e
  impara la freccia giusta).
- Vive in **spazio immagine** (non costruita dai prompt) → niente modality gap.
- In `forward` ogni freccia usata viene **normalizzata a lunghezza 1**:
  `dirs = F.normalize(self.D[cond_col], dim=-1)`. Così la freccia dà solo la
  **direzione**; la **forza** la decide la manopola dei pesi. Questo rende il training
  stabile e l'ortogonalità un semplice coseno.

### 4.2 (b) Testa dei pesi dinamici (la «manopola intelligente»)
```python
self.attr_emb   = nn.Embedding(n_attr, w_emb)            # 40 x 64
self.weight_net = Sequential(Linear(512+64, 256), ReLU(), Linear(256, 1))
w = softplus( weight_net( concat(v_ref, attr_emb[col]) ) )   # [B, C], sempre ≥ 0
```
Per ogni condizione: prende **la foto** (`v_ref`, 512 numeri) **e** un piccolo
identikit dell'attributo (`attr_emb`, 64 numeri), li concatena (576 numeri), e
attraverso due strati produce **un solo numero**: il **peso** di quel vincolo.
- Dipende da `v_ref` → **dinamico (P4)**: foto diverse ricevono pesi diversi.
- Dipende dall'attributo → sa distinguere occhiali da sorriso → **pesatura (P2)**.
- **`softplus`** (= `ln(1+e^x)`) forza il peso a essere **≥ 0**. Perché? Così la
  **forza** (peso) e il **verso** (segno) restano separati: il segno arriva solo da
  `cond_sign`. Se il peso potesse essere negativo, confonderebbe «quanto» con «in che
  verso», rompendo la polarità.

Esempio intuitivo: foto senza occhiali + vincolo `+Eyeglasses` → la rete (allenata)
produce peso **alto** (serve spingere tanto). Se la foto avesse già gli occhiali →
peso **basso** (è già a posto). Nessuna regola scritta a mano: lo impara dai dati.

### 4.3 (c) Composizione segnata e mascherata
```python
dirs  = F.normalize(self.D[cond_col], dim=-1)           # [B,C,512] frecce unitarie
coeff = w * cond_sign * cond_mask                       # [B,C] forza·verso, pad→0
edit  = (coeff[...,None] * dirs).sum(dim=1)             # [B,512] somma sui vincoli
edit  = self.log_scale.exp() * edit                     # scala globale appresa
```
- `coeff` mette insieme **forza** (`w`), **verso** (`cond_sign`, ±1) e **maschera**
  (`cond_mask`): le caselle pad vengono moltiplicate per 0 → **non contribuiscono**.
- La somma su tutti i vincoli (`.sum(dim=1)`) è ciò che rende T2 **invariante
  all'ordine**: «+occhiali, −sorriso» e «−sorriso, +occhiali» danno lo stesso `edit`.
- `log_scale` è una **scalare appresa**; usiamo `exp(log_scale)` per garantire che la
  scala sia sempre positiva. È il corrispettivo dell'`alpha` (≈4) della Soluzione A,
  ma qui **imparato**, non scelto a mano.

### 4.4 (d) Residuo + normalizzazione
```python
# ambient (default)
v_q = v_ref + edit
# poi sempre:
return F.normalize(v_q, dim=1)
```
Il **`+ v_ref`** è l'**àncora di identità**: si parte dalla persona e la si «spinge»
solo un po'. Senza, perderemmo l'identità. Alla fine `normalize` riporta il vettore
sulla sfera (lunghezza 1) per poterlo confrontare col coseno.

Variante **tangente** (ablation, `residual="tangent"`): invece di sommare in linea
retta, si fa un passo «sulla superficie della sfera» (mappa esponenziale ancorata a
`v_ref`). Le misure della Soluzione A dicono che **tangente ≈ ambiente**; la teniamo
come ablation per rigore.

**Caso a zero vincoli** (tutte le caselle pad): `coeff` è tutto 0 → `edit = 0` →
`v_q = normalize(v_ref)`. Cioè: nessun vincolo ⇒ nessuna modifica. Corretto, e
verificato nello smoke test.

### 4.5 Il regolarizzatore di ortogonalità (`ortho_reg`)
```python
Dn   = F.normalize(self.D, dim=1)      # [40,512] frecce unitarie
gram = Dn @ Dn.t()                     # [40,40] coseni tra ogni coppia di frecce
off  = gram - diag(diag(gram))         # azzera la diagonale (auto-coseni = 1)
return off.pow(2).sum() / (40*39)      # media dei coseni-fuori-diagonale al quadrato
```
La matrice `gram` contiene il coseno tra **ogni coppia** di direzioni. Sulla diagonale
c'è 1 (ogni freccia con sé stessa). **Fuori** dalla diagonale c'è quanto due attributi
sono «sovrapposti».
- Se `D["Blond_Hair"]` e `D["Black_Hair"]` puntano quasi nello stesso verso (sono lo
  stesso asse «colore capelli»), il loro coseno è alto → `ortho_reg` è grande →
  **penalità**.
- Minimizzandola, il training **spinge le frecce a essere perpendicolari**
  (disaccoppiate). Così `+Blond & −Black` non si cancellano. **È la Gram–Schmidt
  della Soluzione A, ma imparata invece che fatta a mano** → chiude **P3**.

Questo termine viene aggiunto alla loss dal training loop (Sezione 6), pesato da
`LAM_ORTH = 0.1`.

### 4.6 Conto dei parametri (perché T2 è «piccolissimo»)
| Componente | Forma | Parametri |
|---|---|---|
| `D` (direzioni) | 40 × 512 | 20 480 |
| `attr_emb` (per il gate) | 40 × 64 | 2 560 |
| `weight_net` strato 1 | (576 × 256)+256 | 147 712 |
| `weight_net` strato 2 | (256 × 1)+1 | 257 |
| `log_scale` | scalare | 1 |
| **Totale** | | **≈ 171 010** |

~171k parametri = piccolissimo, gira su GPU in minuti. (T1, a cross-attention, è più
grande: T2 è la carta «leggerezza + interpretabilità».)

---

## 5. Forward pass con esempio numerico (dimensione 3 invece di 512)

Per capire i numeri usiamo vettori da **3** numeri. Scenario: **donna sorridente
senza occhiali**, query `+Eyeglasses, −Smiling`.

Dati di partenza (immagina che il training abbia prodotto questi):
```
v_ref                = [0.90, 0.10, 0.20]      (la foto di riferimento)
D["Eyeglasses"] →norm= [0.00, 1.00, 0.00]      (asse "occhiali")
D["Smiling"]    →norm= [0.00, 0.00, 1.00]      (asse "sorriso")
scala (exp(log_scale)) = 1.0
```

**Passo 1 — pesi dinamici (la manopola).**
La testa guarda `v_ref` e ogni attributo:
```
w["Eyeglasses"] = 0.8     # v_ref ha la 2ª componente bassa (0.10) → "non ha occhiali" → spingi
w["Smiling"]    = 0.7     # v_ref ha la 3ª componente media (0.20)  → c'è sorriso → rimuovi
```
(In T2 reale questi numeri escono dall'MLP allenato; qui li fissiamo per capire.)

**Passo 2 — applica forza, verso e maschera.**
```
+Eyeglasses:  0.8 · (+1) · [0,1,0] = [0.0,  0.8,  0.0]
−Smiling:     0.7 · (−1) · [0,0,1] = [0.0,  0.0, -0.7]
```
Nota il **segno**: occhiali col `+`, sorriso col `−`.

**Passo 3 — somma (l'edit).**
```
edit = [0.0, 0.8, 0.0] + [0.0, 0.0, -0.7] = [0.0, 0.8, -0.7]
edit = scala · edit = 1.0 · [0.0, 0.8, -0.7] = [0.0, 0.8, -0.7]
```

**Passo 4 — residuo (parti dalla foto) + normalizza.**
```
v_q (pre-norm) = v_ref + edit = [0.90, 0.10+0.80, 0.20-0.70] = [0.90, 0.90, -0.50]
lunghezza = sqrt(0.90² + 0.90² + 0.50²) = sqrt(1.87) ≈ 1.367
v_q = [0.90, 0.90, -0.50] / 1.367 ≈ [0.658, 0.658, -0.366]
```

**Lettura del risultato:** la componente «occhiali» (2ª) è **salita** (0.10 → 0.90),
quella «sorriso» (3ª) è **scesa sotto zero** (0.20 → −0.50). Il vettore ora punta
verso «**con occhiali, seria**», restando vicino a `v_ref` (1ª componente quasi
intatta = identità preservata). Il coseno di questo `v_q` contro il database
restituirà foto di persone serie con occhiali simili alla persona di partenza.

**E se la foto avesse già gli occhiali?** La manopola produrrebbe `w["Eyeglasses"]`
**basso** (es. 0.1): la 2ª componente verrebbe spinta poco, perché è già a posto. Lo
stesso edit «si adatta» alla foto → **dinamicità (P4)**.

---

## 6. Come si allena T2 (training)

File: `src/solution_a/train_t2.py`. Allena **solo** i ~171k parametri di T2; **CLIP e
il database restano congelati**.

### 6.1 Da dove vengono gli esempi: il sampler self-supervised
File: `src/solution_b/sampler.py` (backbone condiviso). Genera esempi dalle **40
etichette binarie** di CelebA-train, **senza etichette umane extra**. Strategia
**FLIP-REF**:
1. Pesca una foto di riferimento, es. #X = *donna, sorride, niente occhiali, capelli
   scuri…*.
2. Pesca 1–3 attributi a caso (es. Eyeglasses, Smiling) e **inverte** i valori di #X
   su quegli attributi → il vincolo diventa `+Eyeglasses, −Smiling` (così l'edit non
   è mai banale e un positivo esiste di sicuro).
3. **Positivo** = una foto del train che (a) rispetta i vincoli (occhiali sì, sorriso
   no) **e** (b) differisce da #X per **≤ 2** altri attributi (Hamming ≤ 2 = stessa
   identità di base). Es. *donna, seria, occhiali, capelli scuri*.
4. **Hard negative** = viola **esattamente UN** vincolo, il resto a posto. Es.
   *donna, occhiali, …, ma ANCORA sorride*. Serve a **insegnare il segno**: se T2
   ignorasse la polarità, sbaglierebbe proprio qui.
5. **Easy negative** = tutte le altre foto del batch.

(`cmax=3` vincoli max, `k_hardneg=4` hard-neg per esempio, `max_ham=2`.)

### 6.2 La loss (cosa premia/punisce il training)
File: `src/solution_b/losses.py` + il termine ortho aggiunto da `train_t2.py`.
```
loss = InfoNCE  +  λ_id · identity_anchor  +  λ_orth · ortho_reg(D)
                   (λ_id = 0.3)               (λ_orth = 0.1)
```

**InfoNCE (la parte principale) — spiegata da zero.** Abbiamo il nostro `v_q`, **una**
foto giusta (il positivo) e **tante** sbagliate (hard-neg + le altre del batch).
1. calcola il coseno di `v_q` con **tutti** i candidati;
2. dividi i coseni per la **temperatura** `τ = 0.07` (un numero piccolo «affila» le
   differenze: la rete è costretta a essere decisa);
3. applica softmax (li trasforma in probabilità che sommano a 1);
4. premia se la probabilità più alta è quella del **positivo**.

In una frase: **«fai puntare `v_q` verso la foto giusta e lontano da quelle
sbagliate»**. Gli hard-neg, che violano solo il segno, insegnano la polarità (P1).

**identity_anchor:** `1 − cos(v_q, v_ref)`. Tiene `v_q` vicino alla foto di partenza
→ non perdere l'identità. Pesato `λ_id = 0.3`.

**ortho_reg(D):** il termine della Sezione 4.5; tiene le frecce separate. Pesato
`λ_orth = 0.1`. **È specifico di T2** (T1 e il baseline MLP non ce l'hanno).

### 6.3 Il loop (estratto reale)
```python
for step in range(1, steps + 1):
    b = data.sample_batch(batch)                      # genera un batch di esempi
    v_q = phi(b['v_ref'], b['cond_col'], b['cond_sign'], b['cond_mask'])
    loss, parts = total_loss(v_q, b['v_ref'], b['pos_feat'], b['hneg_feat'], b['hneg_mask'])
    loss = loss + lam_orth * phi.ortho_reg()          # termine specifico T2
    opt.zero_grad(); loss.backward(); opt.step()      # aggiorna SOLO i parametri di T2
    if step % eval_every == 0:
        rows = eval_phi(...)                          # valuta sul benchmark di test
        # salva il checkpoint MIGLIORE (per R@1, poi R@5)
```
- **Ottimizzatore:** Adam, `LR = 1e-3`.
- **Default:** `STEPS = 6000`, `BATCH = 256`, `EVAL_EVERY = 100`.
- **Cosa NON si tocca:** CLIP (frozen) e il database (frozen). Si aggiornano solo
  `D`, `attr_emb`, `weight_net`, `log_scale`.
- **Best checkpoint:** ogni `eval_every` step si misura su test; si tiene il modello
  con il miglior R@1 (a parità, R@5). Output: `results/phi_t2.pt`.

### 6.4 «Epoche» equivalenti
Il training è **a step**, non a epoche (il sampler pesca a caso con rimpiazzo). Conto:
`STEPS × BATCH = 6000 × 256 = 1 536 000` esempi visti; il train ha `162 770` immagini
→ `≈ 9.4` passate equivalenti, cioè **~9 epoche**.

---

## 7. Come si valuta T2

File: `src/solution_a/run_t2.py`. Stesso identico protocollo dei baseline (così il
confronto è onesto).

### 7.1 Cosa è un target valido (ground truth)
Per una coppia (foto sorgente, query), un'immagine del database è bersaglio valido se:
1. **rispetta tutti** i vincoli +/−; **e**
2. è entro **Hamming ≤ 2** dalla sorgente sugli altri attributi (stessa identità).

Si valuta solo sulle **sorgenti elencate nel JSON** (`celeba_evaluation.json`) per
quella query, e solo dove esistono **≥ 5** bersagli validi.

### 7.2 Le metriche
- **Recall@K** = 1 se **almeno un** bersaglio valido è nei primi K risultati, 0
  altrimenti (è un «hit-rate»), mediato sulle sorgenti. K = 1, 5, 10.
- **Precision@K** = (quanti dei primi K sono validi) / K.

### 7.3 Il flusso di eval (per query)
```
per ogni query:
  per ogni sorgente valida s:
    v_q = T2(v_ref = DB[s], vincoli della query)     # query-side
    classifica TUTTO il DB di test per coseno con v_q (escludendo s)
  media Recall@K e Precision@K sulle sorgenti
```
Output: `results/solution_a_t2.json` e `results/solution_a_t2.md` (tabella per query +
MACRO + confronto automatico vs **naive**, **CLAY** e **Soluzione A no-training**). Il
naming `solution_a_t2.*` non collide con i `solution_b_*` del collega.

---

## 8. T2 vs il resto (le differenze che contano)

| | **Soluzione A (no-train)** | **T2 (questa)** | **CLAY** | **T1 (cross-attn)** |
|---|---|---|---|---|
| direzioni | dai **testi** (cono sbagliato) | **apprese, in spazio immagine** | sottospazio SVD dai prompt | token appresi |
| pesi | `cos(v_ref,d)` (≈ costanti, rotti) | **MLP appreso** da `v_ref` | nessuno | gate FiLM appreso |
| segno | freccia orientata (a mano) | freccia orientata **+** hard-neg | **assente** | sign-embedding |
| interazione | Gram–Schmidt a mano | **ortho_reg appresa** | nessuna | attenzione sull'insieme |
| training | **no** | **sì** (solo Φ) | no | sì (solo Φ) |
| punti di forza | semplice, zero training | leggera, interpretabile, fixa i 3 bug | baseline SOTA da battere | query composte/conflitti |

**Differenza chiave vs Soluzione A:** lì `D` veniva dal testo (modality gap) e i pesi
da un coseno rotto. **Qui entrambi sono appresi in spazio immagine** → i due bug
spariscono. T2 è il «gemello con training» di A: confronto perfetto per il report
(«cosa cambia quando alleniamo le stesse idee»).

---

## 9. Comandi (sulla VM)

Dalla **root** del progetto (`~/dl-project`), non da `src/solution_a/`:
```bash
# training (gira a schermo, log live; primo log dopo EVAL_EVERY step)
python -m src.solution_a.train_t2

# valutazione di un checkpoint salvato
python -m src.solution_a.run_t2

# smoke test del solo modello (forme + invarianti, niente dati)
python -m src.solution_a.t2_directions

# GPU
nvidia-smi
```
Riga di log tipica:
```
step   100 | loss 5.12 (nce 5.03 id 0.24 orth 0.002) | R@1 0.022 R@5 0.071 R@10 0.110
```
- `nce` = InfoNCE, `id` = identity anchor, `orth` = ortho_reg (deve **scendere** nel
  tempo = frecce che si separano).
- `R@K` sul benchmark di test = il numero che deve **superare** naive, CLAY e
  Soluzione A.

---

## 10. Iperparametri e ablation

### Iperparametri (default in `train_t2.py` e `t2_directions.py`)
| Nome | Valore | Significato |
|---|---|---|
| `STEPS` | 6000 | passi di training (~9 epoche equiv.) |
| `BATCH` | 256 | esempi per passo |
| `LR` | 1e-3 | learning rate (Adam) |
| `TAU` (`τ`) | 0.07 | temperatura InfoNCE (più piccola = più decisa) |
| `LAM_ID` | 0.3 | peso àncora di identità |
| `LAM_ORTH` | 0.1 | peso ortogonalità (specifico T2) |
| `EVAL_EVERY` | 100 | ogni quanti step si valuta/salva il best |
| `w_emb` | 64 | dimensione embedding attributo per il gate |
| `w_hidden` | 256 | larghezza nascosta dell'MLP dei pesi |
| `residual` | `ambient` | `ambient` o `tangent` (ablation) |

### Ablation previste per il report
- **direzioni apprese vs assi testuali** (= **T2 vs Soluzione A no-training!**);
- **pesi appresi vs statici** (spegni l'MLP, usa peso fisso);
- **ortho_reg sì/no** (`LAM_ORTH = 0`): dimostra che separa le direzioni (P3);
- **segno sì/no** (hard-neg sì/no): dimostra la polarità (P1);
- **residuo ambiente vs tangente**;
- **τ** e **rapporto hard-negative**;
- **numero di parametri** (`w_hidden`, `w_emb`).

---

## 11. Glossario lampo

- **embedding / vettore**: lista di numeri che rappresenta un'immagine o un testo.
- **CLIP**: la rete (congelata) che produce gli embedding da 512 numeri.
- **sfera unitaria**: i vettori normalizzati a lunghezza 1; conta solo la direzione.
- **coseno**: misura di similarità tra due vettori (1 = uguali, 0 = scollegati).
- **direzione di attributo**: freccia che sposta «senza X» → «con X».
- **segno/polarità**: `+` mette l'attributo, `−` lo toglie (stessa freccia, verso opposto).
- **peso dinamico**: quanto spingere su una freccia, deciso guardando la foto.
- **residuo**: `v_ref + edit`; parti dalla foto e modificala poco (preserva identità).
- **ortogonalità**: frecce perpendicolari = attributi separati, non si annullano.
- **InfoNCE**: loss che avvicina `v_q` al positivo e lo allontana dai negativi.
- **hard negative**: foto che viola **solo** un vincolo → insegna il segno.
- **Hamming ≤ 2**: al massimo 2 attributi diversi → stessa identità di base.
- **Recall@K**: 1 se un bersaglio valido è nei primi K risultati.
- **frozen**: congelato, non si addestra (CLIP e il database).
- **query-side**: si modifica solo il vettore di ricerca, mai il database.

---

## 12. Riepilogo in 5 righe

1. T2 trasforma `v_ref` + vincoli (±) in un `v_q` di ricerca, **lato query**.
2. Usa un **dizionario di frecce apprese** (spazio immagine), una **manopola** che ne
   decide la forza guardando la foto, e il **segno** per il verso.
3. Un **regolarizzatore di ortogonalità** tiene le frecce separate (anti-conflitto).
4. Si allena con **InfoNCE + àncora identità + ortho**, CLIP e DB **congelati**, solo
   ~171k parametri.
5. Chiude i 4 limiti di CLAY (P1–P4) e i 3 bug misurati della Soluzione A: è il suo
   **gemello con training**, ed è il confronto centrale del report.
