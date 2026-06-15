# Pianificazione progetto — Dynamic & Hybrid Conditioning for Compositional Image Retrieval

**Team:** 2 persone (A / B) · **Durata:** 21 giorni · **Deliverable finale:** singolo notebook Colab self-contained (codice + report)

**Modello obbligatorio:** CLIP ViT-B/32 (HuggingFace) · **Dataset:** CelebA (test split) · **Metriche:** Recall@K e Precision@K per K = 1, 5, 10

**Query obbligatorie (da `celeba_evaluation.json`):** +Smiling · +Eyeglasses · −Heavy Makeup · +Male · −Young · +Blond Hair · +Mustache · (+Eyeglasses & −Smiling)

---

## Setup iniziale (giorno 0–1)

- [x] Creare repo Git condiviso (codice in `.py`, notebook come orchestratore)
- [x] Impostare struttura cartelle (`data/`, `features/`, `src/`, `notebook/`, `results/`)
- [ ] Configurare `nbstripout` o Jupytext per diff puliti del notebook
- [x] Registrare il gruppo tramite il Google Form
- [x] Verificare accesso a Colab + GPU e al JSON delle query / link Moodle

---

## Fase 1 — Fondamenta condivise (giorni 1–4) · in coppia

- [x] Lettura approfondita dei tre documenti (assignment, GDE/Berasi, CLAY)
- [x] Allineamento sul significato del modulo Φ e su come positivi/negativi devono interagire
- [x] Caricamento corretto di CelebA con la classe PyTorch (usare `celeba[idx]`, **non** i nomi file)
- [x] Esplorazione delle 40 annotazioni di attributi
- [x] Mapping fra le 8 query del JSON e gli attributi del dataset
- [x] Script di validazione: verificare che esistano immagini target valide per ogni query nel corpus

---

## Fase 2 — Feature offline + pipeline di valutazione (giorni 5–8) · A e B in parallelo

**Persona A — Estrazione feature**
- [ ] Caricare CLIP ViT-B/32 da HuggingFace
- [ ] Estrarre le feature visive di tutto il corpus
- [ ] Salvare il database visivo "congelato" (riuso in tutti gli esperimenti)
- [ ] Funzione di encoding testuale per le condizioni

**Persona B — Pipeline di metriche**
- [ ] Costruzione del ground-truth a partire dal JSON
- [ ] Implementare Recall@K (K = 1, 5, 10)
- [ ] Implementare Precision@K (K = 1, 5, 10)
- [ ] Media sulle source image valide + test della pipeline su dati fittizi

---

## Fase 3 — Baseline zero-shot (giorni 8–10)

- [ ] Implementare l'aritmetica latente naïve (v_target ≈ v_ref + t_pos − t_neg)
- [ ] Eseguire la baseline su tutte le query (semplici + composta)
- [ ] Congelare i numeri di riferimento (lower bound)
- [ ] Sanity check della pipeline di valutazione sui risultati baseline

---

## Fase 4 — Sviluppo del metodo (giorni 10–16) · cuore del progetto, sync frequenti

- [ ] Scegliere l'approccio per Φ (cross-attention / gating / proiezione non-lineare / variante dinamica di SVD)
- [ ] Definire formalmente la combinazione positivi/negativi (pesi dinamici condizionati dal testo)
- [ ] Implementare il modulo di fusione
- [ ] (Se training-based) Implementare il training loop
- [ ] Primo giro su subset ridotto per validare l'idea
- [ ] Scaling sull'intero corpus
- [ ] Prima valutazione vs baseline + decisione go / iterate
- [ ] Iterazioni di miglioramento (margine per il fatto che il primo approccio raramente funziona)

---

## Fase 5 — Esperimenti, ablation e confronti (giorni 16–18)

- [ ] Confronto sistematico metodo vs baseline su tutte le query
- [ ] Ablation: capacità della rete
- [ ] Ablation: scelta dell'optimizer
- [ ] Ablation: tuning degli iperparametri
- [ ] Ablation: strategia di sampling dei dati
- [ ] Raccolta di esempi qualitativi (successi)
- [ ] Raccolta di esempi qualitativi (fallimenti)
- [ ] (Opzionale) Test su modelli/dataset aggiuntivi da riportare

---

## Fase 6 — Report e rifinitura (giorni 19–21)

- [ ] Descrizione metodologica formale (matematica dell'architettura, forward pass, loss)
- [ ] Sezione setup sperimentale con motivazione delle scelte
- [ ] Tabelle comparative dei risultati R@K
- [ ] Curve di apprendimento (se applicabile)
- [ ] Esempi qualitativi nel report
- [ ] Citazioni della letteratura rilevante
- [ ] Pulizia e modularizzazione del codice
- [ ] "Appiattire" il codice nelle celle per rendere il notebook self-contained
- [ ] Verifica run end-to-end da zero su Colab
- [ ] Buffer finale per imprevisti

---

## Criteri di valutazione (da tenere d'occhio per tutto il progetto)

- [ ] Originalità e creatività del meccanismo di fusione
- [ ] Rigore metodologico (design sperimentale + validazione)
- [ ] Chiarezza e formalità del report
- [ ] Performance empirica (R@1, R@5) rispetto alla baseline
- [ ] Qualità del codice (leggibilità, modularità, efficienza)

## Promemoria policy

- [ ] Codice costruito da zero / sopra lo starter code del laboratorio (no repo di terze parti)
- [ ] Eventuali snippet esterni commentati e citati
- [ ] Nessuna condivisione di codice tra gruppi
