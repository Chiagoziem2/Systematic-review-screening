# Active-learning abstract screening for systematic reviews

Systematic reviews require a human to read thousands of abstracts to find the
handful that are actually relevant — typically 1–2% of the corpus. This tool
prioritises the reading order using active learning, so most relevant papers
surface early and the rest of the corpus can be left unread.

Benchmarked across all 26 datasets in the
[SYNERGY](https://github.com/asreview/synergy-dataset) collection.

## Headline result

**Mean WSS@95 of 64.1 across 26 datasets** (range 13.3–93.0, 3 seeds per dataset).

WSS@95 = the percentage of the corpus you avoid reading, relative to random
screening, while still finding 95% of the relevant papers. On
`van_de_Schoot_2018`, 95% recall is reached after screening 6.1% of 4,544 records.

![WSS@95 against prevalence](data/wss_vs_prevalence.png)

The figure shows the finding I think is most worth stating: **WSS@95 is not a
property of the method alone — it is strongly conditioned on prevalence**
(Spearman ρ = −0.62, p = 0.0007). Reporting a single dataset's WSS@95 without its
prevalence is close to meaningless, which is why the result above is a distribution
rather than a number.

Prevalence is not the whole story either. It explains under 40% of the variance,
and several datasets defy it badly — `Moran_2021` (2.13% prevalence) scores 13.3
while `Hall_2012` (1.18%) scores 92.0. Why some reviews resist prioritisation is
an open question in this repo, not a solved one.

## Method

- **Data:** SYNERGY, 26 labelled systematic reviews, 258–48,375 records,
  0.16%–21.9% prevalence.
- **Features:** TF-IDF over title + abstract, fitted once on the full corpus.
- **Model:** logistic regression with balanced class weights, refitted as labels
  arrive.
- **Query strategy:** relevance sampling — screen the record with the highest
  predicted P(include).
- **Baseline:** random screening order.
- **Metrics:** WSS@95 and recall-at-effort. Not accuracy or AUC, which are
  misleading at 0.16% prevalence — a classifier predicting "exclude" for every
  record scores 99.84% accuracy and is useless.

### Why relevance sampling, not uncertainty sampling

Standard active learning queries the *least confident* record, because the goal is
to improve the classifier using as few labels as possible. That is the wrong
objective here. A reviewer does not care how good the model becomes; they care how
few abstracts they have to read. The objective is recall and ranking, so the loop
queries the *most likely positive* instead.

## Limitations

Quantified rather than hedged:

1. **Warm-start assumption.** The seed set is forced to contain at least one known
   positive, because a classifier cannot fit on a single class. This hides the cost
   of finding that first positive — expected at (N+1)/(P+1) draws under random
   search. On `Chou_2004` that is 10.0% of the corpus, on `Bos_2018` 9.1%, against
   reported efforts of 62.4% and 9.8% respectively. On `Bos_2018` this roughly
   doubles the true cost. Per-dataset figures: `python3 -m src.sweep --coldstart`.

2. **Batch size is not like-for-like across datasets.** Larger corpora refit the
   model less often (batch 100 at 48k records vs batch 1 under 1k), so large
   datasets are mildly penalised. Measured cost on `Nelson_2002`: effort@95 rises
   from 58.7% at batch 1 to 71.0% at batch 50. Refitting after every record on
   48,375 records was not tractable.

3. **Single untuned classifier.** Bigram features and naive Bayes were tested on
   `Nelson_2002`; neither beat unigram logistic regression by more than the
   seed-to-seed standard deviation, so no variant was adopted. Absence of a
   demonstrated difference, not a demonstrated absence.

4. **Simulated screening.** Labels are revealed instantly and perfectly. Real
   reviewers are slower, disagree with each other, and make mistakes.

5. **Three seeds per dataset in the sweep.** Enough to show the ranking is stable
   on low-prevalence datasets (SD ≈ 0.5 on `van_de_Schoot_2018`), not enough to
   resolve small differences between model variants.

## Reproducing

```bash
pip install -r requirements.txt

python3 -m src.audit                            # dataset audit: size, prevalence, abstract coverage
python3 -m src.run_experiment Nelson_2002       # single dataset, recall-vs-effort curve
python3 -m src.experiments seeds Nelson_2002    # seed sensitivity
python3 -m src.experiments variants Nelson_2002 # model variants
python3 -m src.experiments batch Nelson_2002    # batch size cost
python3 -m src.sweep                            # all 26 datasets + headline figure
python3 -m src.sweep --coldstart                # warm-start cost per dataset
```

SYNERGY downloads automatically on first run. All results above reproduced
identically on two machines running different Python versions.

## Repo layout

```
src/data.py           SYNERGY loading
src/audit.py          dataset audit
src/screening.py      active-learning loop, query strategies, metrics
src/experiments.py    multi-seed sweeps, model variants, batch-size study
src/sweep.py          all-26-dataset benchmark and headline figure
data/                 generated results and figures
```

## Note on AI assistance

The data loader, evaluation harness, and experiment runners were scaffolded with
AI assistance. The active-learning query strategy, the choice of evaluation metrics,
and the interpretation of results are my own. Design decisions I can defend — the
transductive TF-IDF fit, the warm-start assumption, relevance over uncertainty
sampling — are documented inline in `src/screening.py`.

## Next

- Why do `Moran_2021` and `Chou_2003` resist prioritisation despite low prevalence?
  Hypothesis to test: textual heterogeneity among included studies, measurable as
  mean pairwise cosine similarity between positives.
- External validation on CLEF TAR, an independently constructed collection.
- Transformer embeddings (SPECTER/SciBERT) in place of TF-IDF.
