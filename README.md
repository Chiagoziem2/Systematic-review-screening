# Systematic-review abstract screening

Active-learning tool to prioritise which citations a human screens first in a
systematic review, so that most of the relevant papers are found after reading a
small fraction of the corpus. Benchmarked against random screening on the SYNERGY
collection.

**Status:** Week 1 — data audit complete. Modelling not yet started.

## Scope (v1)

- **Data:** SYNERGY (26 labelled systematic-review datasets).
- **Method:** active-learning screening loop (TF-IDF + linear classifier, certainty
  sampling) vs a random-screening baseline. *(Week 2 — not yet written.)*
- **Metrics:** WSS@95, recall-at-effort curve, time-to-discovery. **Not** accuracy/AUC
  — see prevalence below for why. *(Week 3.)*
- **Out of scope for v1:** transformer embeddings (v2); CLEF TAR external validation
  (stretch goal — that collection ships IDs only and needs a real PubMed fetch).

## Week-1 audit findings

Run `python -m src.audit` to reproduce. Three numbers per dataset decide the plan:

- **Prevalence is brutal and variable: 0.16%–21.86%.** Most datasets sit at 1–2% or
  below (Brouwer_2019 is 62 positives in 38,114 records). This is the whole reason
  accuracy/AUC are the wrong headline metrics.
- **Abstract coverage is high: 76.5%–100% (median ~98%).** Missing-abstract handling
  is a minor engineering detail, not a headline feature. Only Appenzeller-Herzog_2019
  (76.5%) is notably sparse.
- **Size range: 258–48,375 records.**
- **No live network fetch.** The packaged datasets reconstruct abstracts offline from
  locally-shipped works. The OpenAlex/PubMed "fetch rabbit hole" applies to CLEF, not
  to the SYNERGY path used here.

## Setup

```bash
pip install -r requirements.txt
python -m src.audit          # downloads SYNERGY (once) and prints the audit
```

If `dataverse.nl` is blocked on your network, use the GitHub mirror:
`python -c "import synergy_dataset as sd; sd.download_raw_dataset(source='github')"`
then run the audit.
