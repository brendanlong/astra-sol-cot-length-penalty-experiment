# cot_spacing — what is the glued CoT actually optimizing?

## Background

OpenAI's report [Suppressing misalignment monitoring alerts and disabling
misalignment monitoring][report] (Oct 20, 2026) is the first public source with
chain-of-thought excerpts from GPT‑6.1 Sol and GPT‑6 Astra. Many of them have
the spaces between words missing:

> Need asknotifications ifsendno-- since noSlack withwrite no_notifytrue
> preventsalertsbut disabling?

[jimrandomh noted][shortform] that this is roughly token-count-neutral under
o200k_base, and read it as a possible side channel: a tokenizer usually has
both ` send` and `send` in its vocabulary, so choosing between them could carry
about a bit per token that a text transcript cannot show.

Brendan's competing reading, which this experiment tests: the models were
trained with a **length penalty measured in characters or in space-separated
words**, not in tokens, and RL found that gluing words together scores well on
that penalty. The irony would be that the emitted text is *more* tokens than
the spaced version, and that re-tokenizing the stored transcript inflates it
further.

[report]: https://alignment.openai.com/misalignment-reports/suppressing-misalignment-monitoring-alerts-and-disabling-misalignment-monitoring/
[shortform]: https://www.lesswrong.com/posts/igEogGD9TAgAeAM7u/jimrandomh-s-shortform?commentId=5yQXwLtLgJqWwg5nf

## Dataset

Every model-generated excerpt in the report, labelled by model (Sol / Astra),
speaker (parent / subagent) and kind (CoT / agent message / user-facing). The
report ships its own "Spaces added" view for each CoT excerpt, which is used as
the word segmentation; a test asserts it differs from the original by spaces
only. The two glued agent messages, which have no such view, are segmented by
hand.

## Questions and predictions

**Q1. Which length unit does gluing actually shorten?** Compare, for the same
visible text, characters, canonical tokens, and tokens
under word-at-a-time emission (each word encoded on its own, carrying a leading
space only where the model wrote one — the obvious way to produce glued text).

- Predicted: characters down a few percent, tokens *up*. A token-based penalty
  cannot be what gluing is minimizing.
- A space-separated-word penalty is **not** a separate hypothesis to test:
  dropping one space removes exactly one character and exactly one whitespace
  chunk, so the two are observationally equivalent on any text. Only the
  character count is reported; "per-space penalty" covers both.
- Counter-evidence would be tokens going down, which would make jimrandomh's
  neutral-to-slightly-worse result an underestimate and remove the irony.

**Q2. Does the choice of which spaces to drop look token-aware?** For every
word boundary, compute the canonical token cost of gluing it. If the model were
optimizing tokens even weakly, the boundaries it glues should be the cheap ones.

- Predicted: no relationship.

**Q3. How hard is the glued text to emit canonically?** A BPE tokenizer merges
across the whole string, so a word inside a glued run may be covered by chunks
that reach into its neighbours (`not send` → `n`+`ots`+`end`). Measure how
often that happens, and how many distinct chunkings a word takes across left
contexts. Compare against GPT‑2 integer tokenization, the standard example of
tokenization forcing memorization over an algorithm.

- Predicted: emitting canonically requires lookahead at a minority of
  boundaries but a substantial one, with per-word chunking at least as
  context-dependent as GPT‑2's integer chunking is number-dependent.

**Q4. Is this a plausible side channel?** Compare the capacity of the visible
space/no-space choice with the capacity of re-segmenting the same characters
into different vocabulary tokens, which a text transcript cannot show.

- Predicted: the invisible channel is much larger. A model trying to hide
  information would not need to disturb the text at all, which is evidence
  against the steganography reading of *this* artifact.

**Q5. Where does the register appear?** Gluing rate by model, speaker and kind.

- Predicted (weakly): concentrated in CoT. Any appearance in user-facing text
  would argue it is an unconditional style rather than a reasoning-time one.

**Q6 (added after run 1). Which tokenization would a BPE LM actually emit?**
Q1 says what each emission model costs; it cannot say which one GPT-6 used.
What is testable on open models is the claim the hypothesis rests on: for text
with the spaces removed, is the word-at-a-time token sequence more probable
than the canonical one? Score both sequences (they decode to identical
characters) under open models, and separately score the one-step choice at each
point where the two diverge, which is the decision a decoder actually faces.

- Predicted: word-at-a-time wins, because the canonical form needs tokens that
  straddle word boundaries and so are rare in that position.

## Out of scope

We cannot observe GPT‑6's token IDs, so "the model emitted one word at a time"
stays a hypothesis about the mechanism; what is measurable is what each
emission model would have cost, and what open models of other lineages prefer. GPT‑6's tokenizer is also not public; o200k is
used as the closest public stand-in and results are reported for three
tokenizers.
