# Synthetic robustness checks

The [case brief](https://anyfin.github.io/kth-case-night/) ranks unseen customers using +1 correct, −1 wrong, and 0 for abstention. This suite tests behavior beyond the 696 official examples without using validation answers. It does not estimate hidden-test accuracy.

## Run

```sh
python3 synthetic.py
python3 test_synthetic.py
uv run test_classify.py
uv run --offline --with anthropic==2.14.0 synthetic.py --llm
```

The first command regenerates `data/synthetic.json` and `data/synthetic-rules-report.json` without an API. The last runs the full classifier with the existing API key/cache and writes `data/synthetic-llm-report.json`; uncached items incur API cost. Reports include per-scenario results and wrong predictions. Exit code 1 means at least one wrong prediction or an API failure; abstentions alone do not fail. The current full-model run exits 1 for the documented Swish disagreement below.

Use `--seed 42 --repeats 24` to vary merchant amounts, capitalization, row order, and the number of structural cases. Increasing repeats adds correlated stress cases, not independent evidence or new merchant coverage. Defaults are seed 20261125 and 12 repeats. Alternate settings overwrite the same generated files; rerun defaults to restore the checked-in corpus.

## Corpus and provenance

876 rows, 780 isolated fictional customer groups, 32 scenario families, every official category plus `unknown`. Each record includes its expected label, rationale, scenario, and a nested input row matching the official schema. These are programming fixtures in JSON, not a spreadsheet or submission file. Expected labels and rationales never enter `predict()`; the full evaluation uses only the original official training rows as examples.

Merchant examples are manually authored business-type assumptions. The generator uses 21 merchant descriptions across 10 bank/provider formats and both money directions (420 rows). It adds 456 structural/adversarial rows covering savings accounts typed as checking, asymmetric credit-card bill labels, five-day pairing boundaries, equally close candidates, unrelated equal amounts, private/company Swish, salary, benefits, capital, marketplace sales, childcare, invoices, mortgages, collections, withdrawals, and insufficient evidence.

The synthetic names and formats are invented stress inputs, not verified statements about how a bank or merchant books transactions. Refund expectations follow the case taxonomy. Generic `unknown` cases reflect a conservative evidence policy. In particular, “Swish ANNA B tack” is deliberately expected to abstain: the message supplies no financial purpose. The model instead chooses `transfer.person`, which is also a plausible interpretation of the taxonomy. Treat those 12 rows as a policy disagreement requiring clarification, rather than confirmed model errors.

The data is deliberately biased toward boundary cases, with repeated templates and familiar merchants. It does not model real transaction frequencies, bank coverage, merchant novelty, privacy guarantees, or production error rates. It is a regression corpus used to develop the fixes below, not an independent held-out benchmark. Generated examples are not added to the classifier's training prompt or persistent merchant table.

## Measured results

| Evaluation | Correct | Wrong | Abstained | Precision on labelled rows | Case score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original rules on synthetic corpus | 228 | 60 | 588 | 79.17% | 19.18 |
| Current rules on synthetic corpus | 228 | 0 | 648 | 100% | 26.03 |
| Current full pipeline on synthetic corpus | 744 | 12 | 120 | 98.41% | 83.56 |

`data/synthetic-baseline-report.json` preserves the run before the rule changes. The full pipeline used Claude Opus 5.5 with the official training examples. Its first run after the rule changes was cold (296 uncached items in five calls, about $0.19); the checked-in report is a warm rerun with the same results (0 calls). The 12 wrong rows are the "Swish … tack" policy disagreement below. The 120 abstentions are 36 ambiguous-pair rows, 24 transfers outside the five-day window, 12 unowned transfers, 12 bare references, and 36 merchant traps (card, cash and provider words inside purchases), which this suite accepts as safer than unsupported labels. Unknown never earns a correct-label point, including where the authored expectation is unknown. `expectation_match` separately counts agreement with abstention expectations.

Leave-one-customer-out on the official 696 training rows still scores 100.0 on a cold run, including all 636 rows with unseen normalised text. All 29 edge-case checks in `test_classify.py` and all 8 corpus checks pass. The validation predictions (`adrian-sohrabi-validation.csv`) and final-set predictions (`adrian-sohrabi.csv`) come from the current code.

## Changes driven by the corpus

Explicit card-purchase and payment-provider markers now take precedence over cash, collector, card-bill, and provider-invoice keyword rules, and are excluded from transfer pairing. Cash rules require an outgoing withdrawal prefix. Collector rules require Kronofogden, or a bill payment to a named collector. A card bill must be a bill payment (BG/Betalning/Autogiro, or the word KREDITKORT), and it pairs only with a receipt on the card. Own-account pairs need texts that name nobody except the customer's own accounts. Merchant ambiguity goes to the model. The scorer also now handles empty datasets and gives unknown-vs-unknown zero points.

The new tests check deterministic generation, input schema and all-category coverage, unique IDs disjoint from the official datasets, zero wrong rule assertions, row-order invariance, isolation across customers, correctness of assigned official training rules, scoring semantics, and the actual `predict()` path with an offline abstaining model. The latter checks that no synthetic answer keys enter the model items and that the corpus's private Swish names are removed.
