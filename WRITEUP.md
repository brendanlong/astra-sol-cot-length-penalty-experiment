# The missing spaces in GPT‑6's chain-of-thought cost it tokens

OpenAI's [misalignment report][report] of 20 October 2026 is the first public
source containing chain-of-thought from GPT‑6.1 Sol and GPT‑6 Astra. Much of it
is missing the spaces between words:

> Need asknotifications ifsendno-- since noSlack withwrite no_notifytrue
> preventsalertsbut disabling? parent shouldn'tset no_notifytrue since policy
> triggeruserno messages requiresnotproactive tool.

[jimrandomh noticed this][shortform] and read it as a possible side channel: a
BPE tokenizer usually holds both ` send` and `send`, so choosing between them
could carry about a bit per token that a text transcript cannot show. Checking
with o200k_base, he found the gluing roughly token-count-neutral.

Measured over the seven glued excerpts in the report, the gluing cuts
characters by 7.6% and *raises* the o200k token count by 7.9% — the same
direction he found, and more than slight. Two findings follow from that, then
four results that cut against the strong version of the story.

## 1. Gluing costs tokens and saves characters

Over the seven glued excerpts in the report, against OpenAI's own
space-restored reading of the same text:

| measure | as written | spaces restored | change |
| --- | --- | --- | --- |
| characters | 3,361 | 3,639 | **−7.6%** |
| tokens, canonical o200k | 667 | 618 | **+7.9%** |
| tokens, word-at-a-time | 652 | 621 | **+5.0%** |

![Diverging bar chart of three length measures for the 7 glued excerpts, each
against the same words with the spaces restored. Characters fall 7.6% (3,361 vs
3,639), while word-at-a-time tokens rise 5.0% (652 vs 621) and canonical o200k
tokens rise 7.9% (667 vs 618). The unit a per-character length penalty counts
goes down; the unit a token counter counts goes up.](cot_spacing/figures/length.png)

Six of the seven excerpts are individually positive in canonical tokens, by
between 1.7% and 12.3% (mean over all seven 5.8%), so this is not one outlier
carrying a pooled ratio; the seventh is a five-boundary fragment that comes out
exactly even. It holds under `cl100k_base` too (+9.1%), though `o200k_harmony`
returns counts byte-identical to `o200k_base`, so that is two distinct
tokenizers and not three. It survives the most contestable re-segmentation a
reader could insist on: merging all ten digit-label boundaries, where the
report renders `severity4?` as two words while leaving `L2` and `P0` as one,
moves the headline from −7.6% / +7.9% to −7.4% / +8.1%.

This is **compatible with a length penalty counting characters, and hard to
square with one counting tokens**. It does not identify the training
objective. GPT‑6's tokenizer is unpublished and its emitted token IDs are not
observable, so o200k is a stand-in; learned behaviour need not minimize any one
reward term on every sample; and a token penalty could coexist with learned
shorthand, imitation of terse human notes, or other rewards pushing the same
way.

## 2. The stored transcript is probably not the token sequence that ran

If the models emit glued text one word at a time — each word carrying the
tokenization it has on its own — then re-tokenizing the stored transcript
yields a different, **2.3% longer** sequence. Whether that is what happens is
testable on open models, because both tokenizations decode to identical
characters, so their probabilities are comparable.

Each model is asked in *its own* vocabulary, since the question is what that
model would emit — three different BPE tokenizers (151,665, 151,669 and 49,152
tokens), none of them o200k. Where the two tokenizations diverge, the
word-respecting token is the more likely one at most divergences, in all three:

| model | divergences | word-at-a-time more likely | median rank (1-indexed): canonical vs word-at-a-time |
| --- | --- | --- | --- |
| Qwen2.5-1.5B-Instruct | 29 | 20/29 | 246 vs **36** |
| Qwen3-1.7B | 29 | 21/29 | 284 vs **45** |
| SmolLM2-1.7B | 20 | 14/20 | 417 vs **193** |

Individual cases are starker than the medians. After
`...MonitorPublicationScope"`, Qwen2.5 ranks the canonical next token ` asn`
**34,498th** and the word-respecting ` as` **7th**. After `ownmonitor userask`,
the canonical `edy` is 2,693rd and `ed` is the model's **first choice**.

So producing the glued text canonically would mean picking a token the model
ranks below 100th at about two thirds of these divergences — 29 positions out
of 667 canonical tokens, not every token. The other third go the other way, and
systematically: where the left word ends in a prefix that itself begins a
common word, the merge is natural (`usesdefault` → `D` at rank 557 against
`Dummy` at 12,557). On the spaced version of the same text the question does
not arise: the two tokenizations are nearly identical (SmolLM2:
identical; Qwen: 3 divergences across all seven excerpts).

The divergence counts differ between models because their vocabularies do:
where a tokenizer already splits the glued string at the word boundaries, there
is nothing to diverge about. Qwen's does that for `notsendmessages`; SmolLM2's
splits it the way o200k does.

These are 1.5–1.7B open models, not GPT‑6, and none was trained to glue words.
They show the straddling tokens are deep in the tail for BPE language models
generally. They cannot show what a model that *learned* to glue would do, since
RL on a text-scored reward does not distinguish the two tokenizations — the
pretrained prior is what breaks the tie.

## 3. Four results that cut the other way

**Most of this text needs no unusual tokens at all.** 240 of 278 glued words
(86%) are canonically chunked exactly as if emitted alone, because o200k has a
bare-word entry for most common English words. The lookahead problem is real at
the other 38 words, which 23 straddling tokens account for — 8% of the glued
boundaries — giving `donotsend` → `don`+`ots`+`end` and `Dummymonitor` →
`D`+`umm`+`ym`+`onitor`.

**Glued English is about as ad hoc as GPT‑2 integers, the usual example of
tokenization forcing memorization — not worse.** Running the same context sweep
on both — how many distinct chunkings an item takes across 248 left contexts,
counting only the item's own bytes — gives 2.09 for a word glued after a word,
against 2.10 for a 3-digit integer glued after an integer and 1.60 for a
4-digit one. The sweep's three metrics do not agree on an ordering: counting
each straddling token's full length instead, words come out higher (4.65
against 3.62 and 3.74), while on the fraction of pairs that chunk differently
from standalone they come out well below 3-digit integers (12.3% against 42.6%)
and level with 4-digit (13.7%). "Same order" is what this supports; a ranking
is not.

![Two grids of chunk patterns side by side, each 25 right items by 248
left-context items, shaded light where an item chunks exactly as it does alone
and darker for its 2nd, 3rd and rarer chunkings. Left, GPT-2 on an integer
glued after an integer: mean 1.60 chunkings per item, 13.7% of pairs chunked
differently from standalone. Right, o200k on a word glued after a word: mean
2.09 chunkings per item, 12.3% of pairs. Both panels are mostly light with a
scattered dark speckle in comparable density, and in each the rows range from
items that never move to items that move under most left
contexts.](cot_spacing/figures/chunking.png)

**The longest Sol chain-of-thought in the report has no glued boundaries at
all** — 143 of them, all spaced, a few turns after a chain-of-thought in the
same conversation that glued 80 of 109. Whatever drives the gluing varies turn
to turn within a single trajectory.

**The model glues the token-cheaper boundaries.** Among mid-sentence boundaries
(sentence-final ones are never glued: 0 of 38), gluing costs +0.18 canonical
tokens where the model did it against +0.36 where it did not — shuffling the
labels within each excerpt gives p = 0.0024. The drop rate is 66.6% where
gluing is token-free against 42.4% where it costs a token (Fisher p = 3.7e‑5).
This was pre-registered as a null and is not one. The likely explanation is
lexical rather than token-aware: common words have a bare vocabulary entry, so
gluing before them costs nothing, and the model glues before common words.

## 4. On the side-channel reading

The space/no-space choice is visible. Anyone reading the transcript can see it,
which is a strange property for a covert channel, and the alternative that
leaves no trace is much larger: over all 15 excerpts with a reference spacing,
re-segmenting the same characters into different vocabulary tokens gives 4.45
bits per reference token against at most 0.77 for the space choice, with 983 of
1,113 tokens having at least one alternative segmentation. Both figures are upper bounds on capacity at fixed
text, not usable rates — spending the invisible capacity emits more tokens than
the reference has, and the visible figure assumes every boundary is free when
sentence boundaries never are.

None of this rules out steganography. It says this artifact is a poor
candidate for it.

## 5. The weakest link

The word-at-a-time mechanism needs non-canonical emission at the 14% of glued
words where it diverges from canonical, which is far above the rates measured
in open models generating ordinary text ([earlier work][retok] puts them at
0–5% per token). The reconciliation available is that those rates were measured
on *normally spaced* text, where the two tokenizations coincide and there is
nothing to be non-canonical about, exactly as the spaced control in §2 shows.
Conditional on producing glued text the local probabilities look very
different. That is a reconciliation, not a measurement.

There is a second tension inside §2. Summed over a whole excerpt rather than at
the point of choice, the canonical path is *better* under both Qwen models
(−38.3 and −8.4 nats) and worse under SmolLM2 (+29.3). The one-step measurement
is the one that corresponds to sampling — a decoder makes the local choice, and
a 34,498th-ranked token is not sampled because its sequence recovers later —
but the whole-sequence numbers do not support the conclusion and this data
cannot separate the two readings.

## What would settle it

- **Token IDs.** This is all an emission *model*. The round-trip test
  (`encode(decode(ids)) != ids` on emitted IDs) is the only direct measurement
  and it needs logprobs or IDs that no API exposes.
- **A length-penalty ablation.** Train two models with a per-character and a
  per-token CoT penalty and see which one glues. That is the experiment this
  analysis is a shadow of.
- **An independent blind segmentation.** The word boundaries here are OpenAI's
  own "Spaces added" view, which is an exact space-insertion of the original
  (asserted in the tests), but it is not blind, and in places it is a reading
  rather than a fact.

## Caveats

Seven glued excerpts from two incidents, and **OpenAI chose which excerpts to
publish** — plausibly in part because the gluing was striking. The
concentration in chain-of-thought (268 of 600 CoT boundaries glued, against 10
of 151 in agent-to-agent messages and 0 of 109 in user-facing text) could be a
selection effect end to end. Nothing here establishes a rate at which GPT‑6
does this.

Two review rounds on the first draft of this analysis withdrew four claims,
including the "no token-awareness" null above, which the corrected test
reverses. [RESULTS.md](RESULTS.md) records each one and what replaced it.

[report]: https://alignment.openai.com/misalignment-reports/suppressing-misalignment-monitoring-alerts-and-disabling-misalignment-monitoring/
[shortform]: https://www.lesswrong.com/posts/igEogGD9TAgAeAM7u/jimrandomh-s-shortform?commentId=5yQXwLtLgJqWwg5nf
[retok]: https://github.com/brendanlong/tokenization-hidden-computation-experiment
