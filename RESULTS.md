# cot_spacing — results

> **Provenance note.** This file is the run log, kept as written rather than
> rewritten for the release, so the commands below are the ones that actually
> ran. They were run in a private monorepo where this experiment lived at
> `experiments/cot_spacing/`; in this repo the package is top-level, so
> `uv run python -m experiments.cot_spacing.X` becomes
> `uv run python -m cot_spacing.X`. Paths named `experiments/cot_spacing/...`
> are `cot_spacing/...` here, and `RESULTS.md` / `EXPERIMENT_PLAN.md` moved to
> the repo root. Nothing else differs: no checkpoints, no GPU, no private
> infrastructure.

Numbers first, interpretation separate and labelled. Everything quoted here is
in `cot_spacing/results/analysis.json` and `cot_spacing/results/emission_logprob.json`; `cot_spacing/results/report.md`
is the generated summary.

## Run 1 — tokenizer-only analysis of the report excerpts (2026-10-21)

**Command**

```bash
uv run python -m experiments.cot_spacing.extract
uv run python -m experiments.cot_spacing.analysis
```

**Inputs.** `cot_spacing/corpus/openai_report_2026-10-20.html`, saved 2026-10-21 from
<https://alignment.openai.com/misalignment-reports/suppressing-misalignment-monitoring-alerts-and-disabling-misalignment-monitoring/>
(report dated "updated Oct 20, 2026"). 22 excerpts, of which 15 are
model-generated prose with a reference spacing and 7 of those contain glued
boundaries. No GPU, no model calls; `tiktoken` 0.13.0, three encodings
(`o200k_base` primary, `o200k_harmony`, `cl100k_base`). Seeds: 0 for the
permutation test and the run-length null, both deterministic.

**What counts as the model's text.** The report replaces internal names with a
bracketed paraphrase (`[runtime sample identifier]`, `[redacted]`) and elides
with `[…]`. Those characters, their internal spaces, and the boundaries either
side of them are the report authors' writing, so each insertion *breaks* the
text rather than being kept (which would measure their spacing) or deleted
(which would invent a boundary that was never written). This costs 14 boundaries
across the glued excerpts.

**Caveat that applies to everything below.** GPT-6's tokenizer is not public and
its token IDs are not observable, so "what the model emitted" is an emission
*model*, not a measurement. o200k is the closest public stand-in.

### Where the register appears (Q5)

| kind | glued boundaries | excerpts with gluing |
| --- | --- | --- |
| CoT | 268/600 | 5/6 |
| agent-to-agent message | 10/151 | 2/6 |
| user-facing response | 0/109 | 0/3 |

By model: Sol glues 2 of 3 CoT excerpts and 0 of 3 agent messages; Astra 3 of 3
CoT and 1 of 2 agent messages; the one quoted instruction from an unspecified
other agent ("Do notdisablemonitor even temporarily") is also glued.

`eval_pipeline-07-parent-cot` is the longest Sol CoT in the report (143
boundaries) and has **0** glued boundaries — same model, same conversation, a
few turns after a CoT that glued 80 of 109. It has its own "Spaces added" panel,
identical to the original, so the exception is in the report, not in the
extraction.

*Interpretation, separate:* this is not evidence that gluing is CoT-specific in
general. **OpenAI chose which excerpts to publish**, plausibly in part because
the gluing was striking, so the concentration in CoT could be a selection
effect end to end. The within-conversation exception is the more robust
observation: whatever drives gluing varies turn to turn.

### Which length unit gluing shortens (Q1)

Totals over the 7 glued excerpts, `o200k_base`:

| measure | as written | spaced | change |
| --- | --- | --- | --- |
| characters | 3361 | 3639 | **−7.6%** |
| canonical tokens | 667 | 618 | **+7.9%** |
| word-at-a-time tokens | 652 | 621 | **+5.0%** |

Figure: `cot_spacing/figures/length.png`.

The word-at-a-time row compares like with like: word-at-a-time emission of the
glued text against word-at-a-time emission of the spaced text. (Against
*canonical* spaced it would read +5.5%, which folds in the ordinary +0.5%
overhead that word-at-a-time emission carries even for unglued text.)

Per excerpt rather than pooled: **6 of 7** glued excerpts are individually
positive in canonical tokens, range +0.0% to +12.3%, mean +5.8%. The tie is the
5-boundary `reference_impl-12`.

Over all 15 excerpts with a reference: characters −4.5%, canonical +4.4%,
word-at-a-time +2.8%. Other encodings, glued excerpts:
`o200k_harmony` identical to `o200k_base`; `cl100k_base` +9.1% canonical, +5.9%
word-at-a-time.

Re-tokenizing the stored transcript costs **+2.3%** over word-at-a-time
emission of the same characters.

**Segmentation sensitivity.** The reference spacing is OpenAI's own, and its
least certain calls are the digit labels: it renders `severity4?` as two words
while `L2`, `P0` and `L1` stay single. Merging all 10 such boundaries — the
worst disagreement a reader could reasonably insist on — moves the headline
from −7.6% / +7.9% to **−7.4% / +8.1%**.

**Why no space-separated-word count here.** A penalty counting whitespace
chunks and a penalty counting characters are *observationally equivalent on
this data*: dropping one space removes exactly one chunk and exactly one
character, wherever it happens, so no arrangement of drops can separate them.
Earlier drafts reported the chunk reduction (−55.4%) as a second finding
alongside the character one; it is the same finding in a different unit, and
the larger-looking percentage is only the smaller denominator. Both candidates
are referred to below as "a per-space penalty".

*Interpretation, separate:* these excerpts are shorter in characters and longer
under every token-counting scheme tested. That is **compatible with** a
per-space length incentive and **hard to square with** a straightforward
token-counting one. It does not identify the training objective: the tokenizer
and the emitted sequence are both unobserved, learned behaviour need not
minimize any one reward term on every sample, and a token penalty could coexist
with learned shorthand, imitation of terse human notes, or other rewards.

### Which boundaries get glued (Q2)

278 of 489 word boundaries in the glued excerpts have no space. (The glued
excerpts hold 495 gaps in total; six are paragraph breaks in the report rather
than places the model chose between a space and none, and are excluded
throughout.)

Mid-sentence boundaries only (sentence boundaries are never glued, and are also
where the tokenizer happens to reward gluing):

| | glued | kept |
| --- | --- | --- |
| mean canonical tokens gluing adds | **+0.183** | **+0.358** |

Shuffling the glued/kept labels *within each excerpt* (so an excerpt that both
glues a lot and is cheap overall cannot produce the effect): one-sided
**p = 0.0024** over 20,000 permutations (`o200k_harmony` 0.0024, `cl100k_base`
0.0048).

Split by whether gluing costs anything at all:

| set | drop rate where gluing is token-free | where it costs tokens | Fisher p |
| --- | --- | --- | --- |
| all 489 boundaries | 60.7% (n=394) | 41.1% (n=95) | 0.00076 |
| mid-sentence only | 66.6% (n=359) | 42.4% (n=92) | 0.000037 |

There are only 9 boundaries where gluing would outright *save* a canonical
token; the model glued 1, and only 2 of the 9 are mid-sentence, so that split
is too small to test and the free-versus-costly one above replaces it. (An
earlier draft ran a Fisher test on it and reported p = 0.013; it turned
entirely on the single `severity | 4?` boundary, itself one of the contestable
digit-label calls. Withdrawn.)

What else predicts a kept space:

| predictor | drop rate | n | drop rate otherwise |
| --- | --- | --- | --- |
| left word ends a sentence (`.?!;:`) | 0% | 38 | 62% |
| right word contains a digit | 100% | 11 | 56% |
| left word ≤ 3 chars | 67% | 99 | 54% |
| left word ≥ 8 chars | 45% | 135 | 61% |
| right word is a code identifier | 54% | 13 | 57% |
| right word starts uppercase | 51% | 59 | 58% |

Only `left_ends_sentence` is both large and spread across excerpts (0/38 over
all 6 glued CoT excerpts). The other two striking rows are near-duplicates of a
single item each: `right_has_digit` is 9 severity labels (`L1`/`L2`/`P0`/`p0`)
plus `severity 4?`, 7 of them in one excerpt; the digit-label class is also the
one the segmentation is least sure about. This is a 6-predictor scan with no
multiplicity correction, so treat anything but the sentence-boundary row as a
single observation.

Run structure: mean glued-run length 2.24 words **carries no information** — it
equals (boundaries + excerpts) / (kept boundaries + excerpts) = 2.24 whatever
the arrangement, so it cannot detect clustering. On the testable statistic,
run-length variance is 2.56 against 2.80 for independent per-boundary dropping
at the same rate simulated per excerpt (95% 1.90–4.03, p = 0.64).

*Interpretation, separate:* among mid-sentence boundaries the model glues the
token-cheaper ones. This is **not** evidence of token optimization. The obvious
alternative is lexical: common words have a bare vocabulary entry, so gluing
before them costs nothing, and the model glues before common words. Boundaries
within an excerpt are also not independent, and the cost is evaluated in the
observed context. An earlier draft of this section claimed "no token-awareness
in boundary choice"; two independent reviews showed that was the null reading of
an untested mean comparison, and it is withdrawn. The run-length result is
consistent with independent per-boundary decisions, which is what a linear
per-space incentive would give — though a linear incentive does not by itself
predict independence, since syntax and readability could correlate the choices.
The withdrawn claim that gluing "respects code tokens" was an artifact: the
identifier detector required a lowercase first character and so missed every
PascalCase class name in the report. Corrected, the effect disappears (54% vs
57%), and 7 of the 8 boundaries before PascalCase identifiers are in fact glued.

### How hard the glued text is to emit canonically (Q3)

**240 of 278 glued words (86%) are canonically chunked exactly as if emitted
alone**, so for most of the text word-at-a-time emission *is* canonical
emission and no non-canonical token is required. The other 38 are where a
canonical token reaches across a word boundary (23 of the 278 boundaries are
straddled):

| glued text | canonical chunks | the words' own chunks |
| --- | --- | --- |
| `donotsend` | `don` `ots` `end` | `do` `not` `send` |
| `Dummymonitor` | `D` `umm` `ym` `onitor` | `Dummy` `monitor` |
| `pathsame` | `paths` `ame` | `path` `same` |
| `avoiddehydration` | `avo` `idde` `hydr` `ation` | `avoid` `de` `hydr` `ation` |

Context sweep, 150 word types from the excerpts × 248 left contexts from the
same vocabulary, mean distinct chunk patterns per item. The GPT-2 integer rows
run the *same* sweep on random integers, which is the like-for-like comparison;
"clipped" counts only the bytes of the right item, the convention the integer
baseline uses.

| sweep | unclipped | clipped | pairs differing from standalone (clipped) |
| --- | --- | --- | --- |
| glued English (o200k) | 4.65 (max 11) | 2.09 (max 4) | 12.3% |
| glued English, space before the left word | 5.92 (max 16) | 2.11 (max 4) | 11.8% |
| GPT-2 3-digit integers | 3.62 (max 5) | 2.10 (max 3) | 42.6% |
| GPT-2 4-digit integers | 3.74 (max 7) | 1.60 (max 3) | 13.7% |

For reference, Beren Millidge's [population view][beren] of the same tokenizer,
which is *not* comparable to the sweep: 9.7% of consecutive 4-digit integers
chunk differently from their successor across 5 distinct patterns, and 16.0% for
3-digit across 3 patterns. His exact examples (2249 → `2`+`249`, 2250 →
`22`+`50`, 2251 → `225`+`1`) are asserted in the tests.

[beren]: https://www.beren.io/2023-02-04-Integer-tokenization-is-insane/

Figure: `cot_spacing/figures/chunking.png` draws the glued-English and 4-digit-integer
sweeps side by side, clipped convention, 25 sampled right items each.

*Interpretation, separate:* on the matched sweep, a word glued after another
word is about as context-dependent in its chunking as a GPT-2 integer is, and
the three metrics do not agree on an ordering: clipped, words are 2.09 against
2.10 (3-digit) and 1.60 (4-digit); unclipped they are higher than both (4.65 vs
3.62 and 3.74); on pairs differing from standalone they are lower than 3-digit
(12.3% vs 42.6%) and level with 4-digit (13.7%). "Same order" is supported; a
ranking in either direction is not. An earlier draft said "arguably worse",
which compared 4.75 chunkings-per-word against the 5 patterns in the whole
integer population — a category error, since per integer the count is 1 by
construction. Withdrawn.
Note also that a pattern count does not establish comparable learning
difficulty; it describes how irregular the mapping is, not what a model can
learn.

### Visible vs invisible channel (Q4)

Over the 15 excerpts (`o200k_base`). Both figures are upper bounds on capacity
at a fixed reference text, counted per canonical reference token:

- space/no-space at 860 word boundaries: ≤ 860 bits, **0.77 bits per reference
  token**, plainly visible to anyone reading the transcript.
- re-segmenting the same characters into other vocabulary tokens: 983/1113
  tokens have at least one alternative, 4951 bits, **4.45 bits per reference
  token**, invisible in a transcript stored as text.

*Interpretation, separate:* neither number is a throughput a model could rely
on. Spending the invisible capacity emits more tokens than the reference has
(uniform selection among the counted segmentations would emit roughly 3.5× as
many), so per *emitted* token it is far smaller than 4.45; and the visible
figure assumes every boundary is free when sentence boundaries are never glued.
What the comparison supports is that the total combinatorial room at fixed text
is much larger in the channel that leaves no trace. That is a reason to doubt
the steganography reading of *this* artifact, not a refutation of it:
availability is not learnability, and what an intended receiver observes is
untested.

This repo's own prior work is the stronger version of the argument.
Unpublished follow-up work to [the retok release][retok] records 7+
GRPO runs across Qwen3-1.7B/0.6B and OLMo-2-1B in which **tokenization is never
used as a channel even though it would pay slightly more**, and in which a
partial-credit reward makes RL find the cheapest *visible* channel within 50–80
steps. The binding constraint there was reachability, not capacity:
teacher-forced non-canonical split probability was ~7.5e-6 per copy for
Qwen3-1.7B, and an entropy bonus large enough to hold 0.15–0.35 nats/token still
produced zero non-canonical copies.


### Status of the run 1 predictions

| prediction | outcome |
| --- | --- |
| Q1 characters down, tokens up | **confirmed** (−7.6% / +7.9%; 6/7 excerpts positive; robust to the digit-label re-segmentation) |
| Q1 character vs space-separated-word penalty distinguishable | **no** — the two are observationally equivalent, so only the character count is reported |
| Q2 no token-awareness in boundary choice | **falsified** — glued boundaries are the token-cheaper ones (mid-sentence 0.18 vs 0.36, p=0.0024) |
| Q3 canonical emission needs lookahead often | **weakened** — 14% of glued words, 86% are canonical as-is |
| Q3 per-word context dependence ≥ GPT-2 integers | **same order, not greater** (clipped 2.09 vs 2.10) |
| Q4 invisible channel much larger | **confirmed** as total capacity at fixed text (4.45 vs 0.77 bits per reference token), not as a usable rate |
| Q5 concentrated in CoT | **confirmed in this sample** (268/600 CoT boundaries vs 10/151 and 0/109), but OpenAI chose the excerpts |

## Run 2 — which tokenization an open BPE model would emit (2026-10-21)

**Command**

```bash
# the three small models, CPU
uv run python -m experiments.cot_spacing.emission_logprob
# = --models Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen3-1.7B HuggingFaceTB/SmolLM2-1.7B

# the gpt-oss pair, GPU (MXFP4 is native on Hopper and Blackwell)
uv run python -m experiments.cot_spacing.emission_logprob --gpt-oss \
    --device cuda --out results/emission_logprob_gpt_oss.json
```

The three small models run on CPU in float32, ~15 min (Qwen2.5 and Qwen3 share
a tokenizer; SmolLM2 has its own). The gpt-oss pair was added afterwards on a
rented RTX PRO 6000 Blackwell 96 GB, where MXFP4 is native — 13.8 GB and
65.4 GB of weights resident — for about 17 minutes of pod time. Each model is
scored in **its own** vocabulary — 151,665, 151,669 and 49,152 tokens, none of them the o200k
used everywhere else here — because the question is which sequence that model
would emit. So the divergence positions are not the o200k straddles of run 1,
and the counts differ between models: where a tokenizer already splits the
glued string at its word boundaries there is nothing to diverge about. Raw output in
`cot_spacing/results/emission_logprob.json`.

Run 1 says what each emission model would have *cost*, which cannot tell us
which one GPT-6 used. Whether word-at-a-time emission is plausible for a BPE LM
is testable on open models, and it decides whether the stored transcript is the
token sequence that ran. Both tokenizations decode to identical characters, so
they are scored two ways: total sequence log-probability, and the one-step
choice at each point where they diverge (same prefix, two candidate next
tokens).

### One-step: the word-respecting token is the likelier of the two

Each model is scored in its own vocabulary. gpt-oss is the one to read: its
tokenizer gives token-for-token identical ids to the `o200k_base` used in run
1, so its divergences are the same positions run 1 measures. The others are a
cross-lineage check whose divergences sit elsewhere and are not directly
comparable.

**Only decisions at a glued boundary count.** The two emission models also
disagree at the report's paragraph breaks, where canonical merges `.\n` into
one token and word-at-a-time does not. That is the report's formatting, not a
choice the model made, and it accounted for **12 of 126** decisions before the
filter — so an earlier version of this table read 17/24 for gpt-oss-120b where
the corrected figure is 15/21.

**Which positions.** The two tokenizations partition the same characters
differently. At every offset where both put a cut the two paths are in the same
state; between two shared cuts they either chose the same tokens or they did
not, and where they did not the first token of that stretch is the decision,
taken after an identical prefix. Nothing is selected by hand. For
`notsendmessages`: canonical cuts after 1, 4, 7, 15 (`n`/`ots`/`end`/
`messages`), word-at-a-time after 3, 7, 15 (`not`/`send`/`messages`); they
share the cut at 7, so that excerpt contributes one divergence (`n` against
`not`), not two.

| model | divergences | word-at-a-time more likely | median rank (1-indexed) | whole excerpt (nats) |
| --- | --- | --- | --- | --- |
| gpt-oss-120b | 21 | 15/21 | 2,365 vs **596** | -10.5 |
| gpt-oss-20b | 21 | 14/21 | 848 vs **172** | +21.2 |
| Qwen2.5-1.5B-Instruct | 26 | 17/26 | 209 vs **30** | -38.3 |
| Qwen3-1.7B | 26 | 18/26 | 302 vs **110** | -8.4 |
| SmolLM2-1.7B | 20 | 14/20 | 417 vs **193** | +29.3 |

Ranks are 1-indexed: rank 1 is the model's first choice. (An earlier version of
this table reported 0-indexed ranks as ordinals, so every number was one too
small; `emission_logprob.py` now produces 1-indexed ranks directly.)

On gpt-oss-120b the starkest case is after `assinstruction? This`, where the
canonical `isa` is the model's 3,602nd choice and `is` is its 42nd.

Median ranks are 2,365 canonical against 596 word-respecting.

**Both candidates are deep in the tail.** For gpt-oss-120b the canonical token
is ranked below 100th at 18 of 21 of these positions, and the
word-respecting one at 14 of 21. The model finds the glued text unlikely however it is cut up;
the measurement says which of two improbable options is the less improbable,
not that either is one it would readily produce. The small models put the
word-respecting token higher (median 36–193), but in vocabularies where the
divergence positions are different.

**How few decision points there are.** This is the whole evidence base for
run 2: 21 decisions for gpt-oss-120b, covering 23 of the 278 glued boundaries
in the corpus, because a diverged stretch spanning two glued boundaries is
scored once. The alternative construction — at every glued boundary, with the
word-at-a-time prefix, compare the bare next word against what a fresh
tokenization from that point would emit — is better posed (one decision per
boundary, no paragraph-break artifacts by construction) but yields *fewer*
comparisons, not more: the two candidates differ at only **13** of the 278
boundaries, and at only **4** does the from-here token run past the end of the
next word (`don`, `nom`, `isa`, `paths`). Under either construction the effect
has very few observable decision points in this corpus.

**Control.** On the spaced version of the same excerpts the two tokenizations
are nearly identical — all but 3 positions across all 7 excerpts, for every
model. Word-at-a-time emission of normal English *is* canonical emission, which
is why the question only arises for glued text.

### Whole sequence: disagrees for three of the five

The last column of the table above is the whole-excerpt total. Summed over an
excerpt rather than taken at the point of choice, the canonical path comes out
*better* for gpt-oss-120b and both Qwen models, and worse for gpt-oss-20b and
SmolLM2. That total is a subtraction, not an attribution: it covers the later
tokens inside a diverged stretch plus every token identical in both paths but
conditioned on a differently segmented prefix.

*Interpretation, separate:* the one-step measurement is the one that
corresponds to sampling — a decoder makes the local choice, and a token ranked
in the thousands is not sampled because its sequence recovers later. On that
measurement the re-tokenized transcript is probably not the sequence that ran.
The whole-sequence numbers do not support that, and this data cannot separate
the two readings.

**Caveat.** gpt-oss-120b is a mixture-of-experts model of about 120B total
parameters, not GPT-6, and none of these models was trained to glue words.
They establish that the straddling tokens sit in the tail for BPE models of
this kind; they cannot establish what a model that learned to glue would do,
since RL on a text-scored reward does not distinguish the two tokenizations
and the pretrained prior breaks the tie.

**The live threat to this hypothesis.** This repo's prior work
([tokenization-hidden-computation-experiment][retok]) measured wild
non-canonical rates on 7 open models at 0–5% *per token*, driven by temperature
and language, 0.08% greedy — and 0/346 induced at frontier scale by prompting.
The word-at-a-time mechanism needs non-canonical emission at the 14% of glued
words where it diverges from canonical, which is orders of magnitude above
those rates. The reconciliation available is that those rates were measured
while models generated *normally spaced* text, where the two tokenizations
coincide (exactly as the control above shows) and there is nothing to be
non-canonical about; conditional on producing glued text the local
probabilities look very different. That is a reconciliation, not a measurement,
and it is the weakest link in the chain.

[retok]: https://github.com/brendanlong/tokenization-hidden-computation-experiment

**Caveat.** These are 1.5–1.7B open models, not GPT-6, and none was RL-trained
to glue words. They establish that the straddling tokens are deep in the tail
for BPE LMs in general; they cannot establish what a model that learned to glue
would do, since RL on a text-scored reward does not distinguish the two
tokenizations and the pretrained prior is what breaks the tie.

### Status of the Q6 prediction

| prediction | outcome |
| --- | --- |
| word-at-a-time is the more probable emission | **confirmed at the point of choice** in 5/5 models — 15/21 glued-boundary divergences for gpt-oss-120b; **not confirmed** as whole-sequence log-probability (3/5 favour canonical) |
| (unregistered) either candidate is a plausible emission | **no** — on gpt-oss-120b both sit below rank 100 at most divergences |

## Known gaps (both runs)

- n = 7 glued excerpts from 2 incidents, and **OpenAI chose which excerpts to
  publish**. Nothing here generalizes to a rate at which GPT-6 does this, and
  the concentration in CoT may be entirely a selection effect.
- The reference segmentation is OpenAI's own and is not blind. The digit-label
  sensitivity above bounds the headline's exposure to that (±0.2pp), but no
  independent segmentation was produced. `statutor guards` in
  `reference_impl-19` looks like a misspelling of "statutory guards" that no
  segmentation can repair.
- No GPT-6 token IDs and no public GPT-6 tokenizer, so the round-trip metric
  this repo normally insists on for tokenization claims (`encode(decode(ids))
  != ids` on emitted IDs) is unavailable in principle. Everything is an
  emission model.
- The whole-sequence and one-step measurements in run 2 point opposite ways
  under Qwen, and the residual was not decomposed.
- The boundary predictors are a 6-way scan on 489 boundaries with
  within-excerpt dependence and no multiplicity correction.
- `EXPERIMENT_PLAN.md`, the code and the first results landed in a single
  commit, so the plan's priority is asserted, not verifiable from history.
