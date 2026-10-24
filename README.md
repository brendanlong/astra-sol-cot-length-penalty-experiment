# astra-sol-cot-length-penalty-experiment

What the missing spaces in GPT‑6.1 Sol and GPT‑6 Astra chain-of-thought are
optimizing.

OpenAI's [misalignment report][report] of 20 October 2026 is the first public
source with chain-of-thought from either model, and much of it has the spaces
between words missing:

> Need asknotifications ifsendno-- since noSlack withwrite no_notifytrue
> preventsalertsbut disabling?

**→ Read the [report](https://gpt6-cot-spaces-cd652c60.surge.sh)** (or
[WRITEUP.md](WRITEUP.md), the same thing in Markdown).
Discussed [on LessWrong](https://www.lesswrong.com/posts/igEogGD9TAgAeAM7u/jimrandomh-s-shortform?commentId=su76ee7QNov6NqhBw). [RESULTS.md](RESULTS.md) is the full run
log — every command, every number, the four claims two review rounds withdrew,
and the known gaps. [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) has the questions
and predictions.

## Findings

1. **Gluing costs tokens and saves characters.** Over the 7 glued excerpts,
   against OpenAI's own space-restored reading: characters **−7.6%**, canonical
   o200k tokens **+7.9%**, word-at-a-time tokens **+5.0%**. 6 of 7 excerpts are
   individually positive in tokens. Compatible with a length penalty counting
   characters, hard to square with one counting tokens — but it cannot identify the
   training objective, since GPT‑6's tokenizer and emitted tokens are both
   unobservable.

   **But shorter words went unused.** 12 everyday swaps for words GPT‑6 wrote
   (`msgs`, `auto`, `ok`, `pref`…, each among the ~0.25% of words nearest the
   original in gpt-oss's embeddings, checked by hand) would have saved 83
   characters — 30% of what gluing saved — at no cost in o200k tokens. GPT‑6
   wrote the full form 20 times. That leans toward a penalty on word count
   rather than characters, though not decisively.

   **Is gluing just easier to learn?** [STEERING.md](STEERING.md) tests that on
   gpt-oss-20b. Gluing is a sturdier steering direction and, learned on some
   excerpts, transfers to others about 15× better per edit than short forms.
   But unsteered gpt-oss is about 12× likelier to write a short form than to
   drop a space, so a character penalty would start by pushing short forms.
   The word-penalty reading survives, weakened.

2. **The stored transcript is probably not the sequence that ran.** Scored on
   **gpt-oss-120b** — an OpenAI model whose tokenizer gives the same ids as the
   o200k used above, so the same positions — the word-respecting token is the
   likelier of the two at **15 of 21** points where the tokenizations disagree
   at a glued boundary (median rank **2,365** canonical against **596**). Both are deep in the
   tail, so this says which of two improbable options is less improbable, not
   that either is likely. Four smaller models agree. Re-tokenizing inflates the
   transcript **+2.3%** over word-at-a-time emission.

3. **86% of glued words are canonically chunked exactly as if emitted alone**,
   so most of this text needs no unusual token. On a matched context sweep, a
   glued word is about as context-dependent in its chunking as a GPT‑2 integer
   (2.09 against 2.10 distinct chunkings per item) — the same order as the
   standard example of tokenization forcing memorization, not worse.

4. **The longest Sol chain-of-thought in the report is perfectly spaced**, 143
   boundaries, in the same conversation as one that glued 80 of 109.

5. **The model glues the token-cheaper boundaries.** Mid-sentence, gluing costs
   +0.18 canonical tokens where it happened against +0.36 where it did not
   (within-excerpt permutation p = 0.0024). Pre-registered as a null; it is
   not one. Probably lexical rather than token-aware.

6. **A poor candidate for a side channel.** The space choice is visible in the
   transcript, and re-segmenting the same characters — which leaves no trace —
   offers 4.45 bits per reference token against at most 0.77. Prior RL work
   found tokenization is never used as a channel even when it pays.

![Two grids of chunk patterns side by side, each 25 right items by 248
left-context items, shaded light where an item chunks exactly as it does alone
and darker for its 2nd, 3rd and rarer chunkings. Left, GPT-2 on an integer
glued after an integer: mean 1.60 chunkings per item, 13.7% of pairs chunked
differently from standalone. Right, o200k on a word glued after a word: mean
2.09 chunkings per item, 12.3% of pairs. Both panels are mostly light with a
scattered dark speckle in comparable density, and in each the rows range from
items that never move to items that move under most left
contexts.](cot_spacing/figures/chunking.png)

## The dataset

All 22 model-generated and user excerpts from the report, labelled by model
(Sol / Astra), speaker (parent / subagent / other agent / user) and kind
(chain-of-thought / agent message / user-facing / tool call).

The report's HTML ships OpenAI's **own "Spaces added" view** for each
chain-of-thought excerpt, and it is an exact space-insertion of the original —
a test asserts this — so the word segmentation is theirs, not a judgement call
made here. (The PDF export of the report drops those panels.) The two glued
agent messages have no such view and are segmented by hand; every other model
output is recorded in `cot_spacing/reference.py` as an explicit "not glued"
judgement rather than a silent default.

Editorial insertions — the report's `[redacted]`-style paraphrases and its
`[…]` elisions — break the text rather than being measured, since their
characters, internal spaces and adjacent boundaries are the report authors'
writing.

## Setup

```bash
uv sync
uv run pytest
```

Python 3.12+. No GPU and no accounts are needed. The one step that downloads
anything is the open-model emission scoring (~10 GB of weights, CPU, ~15 min).

## Reproducing

```bash
bash scripts/reproduce.sh            # everything
SKIP_MODELS=1 bash scripts/reproduce.sh   # skip the model downloads
```

Or step by step:

```bash
# rebuild cot_spacing/corpus/excerpts.json from the saved report HTML
uv run python -m cot_spacing.extract

# every tokenizer number -> cot_spacing/results/{analysis.json,report.md}
uv run python -m cot_spacing.analysis

# the two figures -> cot_spacing/figures/*.png
uv run python -m cot_spacing.figures

# which tokenization open models would emit (CPU, ~15 min)
uv run python -m cot_spacing.emission_logprob
```

## Layout

```
.
├── WRITEUP.md              # the write-up
├── scripts/build_report.py # builds the published HTML report from results/
├── RESULTS.md              # run log: commands, numbers, withdrawn claims, gaps
├── EXPERIMENT_PLAN.md      # questions and predictions
├── scripts/reproduce.sh
└── cot_spacing/
    ├── corpus/             # the report as saved + the extracted dataset
    ├── extract.py          # report HTML -> corpus/excerpts.json
    ├── data.py             # pydantic models, loader, editorial-insertion handling
    ├── reference.py        # the space-restored reference per excerpt
    ├── segment.py          # aligns a glued excerpt to its spaced version
    ├── tokenize_cost.py    # the length measures and per-boundary costs
    ├── chunking.py         # chunk patterns, GPT-2 integer baseline, channel capacity
    ├── emission_logprob.py # canonical vs word-at-a-time under open models
    ├── analysis.py         # driver; writes results/
    ├── figures.py
    ├── results/            # committed outputs
    └── tests/
```

## Caveats

Seven glued excerpts from two incidents, and OpenAI chose which excerpts to
publish — the concentration in chain-of-thought may be a selection effect.
GPT‑6's tokenizer is unpublished and its token IDs are unobservable, so every
tokenization claim here is about o200k or about open models standing in for the
mechanism. [RESULTS.md](RESULTS.md) lists the rest, including the four claims
two review rounds withdrew from the first draft.

## Provenance

Extracted from a private monorepo where this lived at
`experiments/cot_spacing/`; `RESULTS.md` is the original run log, with a header
note mapping the historical commands onto this layout. No checkpoints, no
training runs, no private infrastructure — everything here runs from the
committed corpus.

Built with [Claude Code](https://claude.com/claude-code) under direction, with
the analysis reviewed by separate adversarial review passes whose findings are
recorded in RESULTS.md.

Report text quoted under fair use for research commentary; it remains
© OpenAI. Code in this repo is MIT ([LICENSE](LICENSE)).

[report]: https://alignment.openai.com/misalignment-reports/suppressing-misalignment-monitoring-alerts-and-disabling-misalignment-monitoring/
