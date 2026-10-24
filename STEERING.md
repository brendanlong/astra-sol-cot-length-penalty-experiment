# Is gluing easier to learn than shortening? Steering gpt-oss-20b

The main write-up leans on one observation: GPT‑6 dropped the spaces between
words, but it rarely used the short forms that were available (`msgs`,
`auth`, `mon`), which would have saved more characters per edit. If the length
penalty counted characters, why would it glue but not shorten? That points to a
penalty counting words, unless gluing is simply **much easier for a model to
learn**. If so, a character penalty could produce gluing first and short forms
later, or never.

This tests that alternative on gpt-oss-20b, the nearest open relative. It asks
three things:

1. Can gpt-oss be steered into each habit while still reasoning correctly, and
   which one needs less push?
2. Does a fine-tuning step on one set of excerpts teach the habit on a
   different excerpt?
3. How often would unsteered gpt-oss make each edit by itself? Reinforcement
   learning (RL) can only reward what the model already samples.

**Short version.** The answers split.

- **Gluing behaves like one general habit.** It is a clean direction that can
  be pushed far without breaking the model. Training on gluing in some
  excerpts carries over strongly to new ones.
- **Short forms are word-by-word.** Steering toward them works, but it breaks
  arithmetic at lower strength. Training on some excerpts' short forms barely
  carries over to new words.
- **But unsteered gpt-oss is more likely to use a short form than to drop a
  space.** At the typical site the odds are about 1 in 160 for a short form
  and 1 in 1,800 for a dropped space. A short form also saves 4.6 characters
  to gluing's 1. So at the start of RL with a character penalty, the pull
  toward short forms would be the stronger one.

So "gluing is just easier" holds for how readily the habit *spreads*, not for
how readily it is *found*. This weakens, but does not overturn, the
word-penalty reading. Everything here is gpt-oss-20b, not GPT‑6, with 5
excerpts.

## Data

The 5 glued chain-of-thought excerpts (the agent messages are left out), in
four renderings ([`variants.py`](cot_spacing/variants.py)):

| rendering | what it is | characters saved vs spaced |
|---|---|---|
| `spaced` | OpenAI's own space-restored reading | — |
| `glued` | what GPT‑6 wrote | 268 (268 dropped spaces) |
| `short` | spaced, with 99 of 462 words shortened | 454 |
| `both` | what GPT‑6 wrote, with the same words shortened | 722 |

The short forms come from one hand-written list (`SHORT_FORMS`): common
abbreviations that mean the same thing at every use, applied only to whole
words and never inside identifiers (`no_notify`, `RuntimeContext`). Examples
are `monitor → mon`, `authorization → auth`, `messages → msgs`,
`automatically → auto` and `with → w/`.

Each rendering is put in gpt-oss's analysis channel (its reasoning). The user
turn before it is the first user prompt from the same incident in the report,
and the reasoning effort is "low".

## 1. Steering

**Method** ([`steering.py`](cot_spacing/steering.py)). A *difference-of-means
direction* is built per layer. Take the average residual-stream activation
over the reasoning tokens of, say, the glued renderings, and subtract the same
average for the spaced renderings. The residual stream is the vector each
layer passes to the next. The result points from "spaced" toward "glued" in
the model's internal space.

To steer, that direction is added at every position after one layer
(6, 12 or 18 of 24) while gpt-oss answers prompts it has not seen:

- 32 GSM8K arithmetic problems, which give an accuracy check on coherence;
- 8 short agentic prompts written in the report's register.

All three directions are scaled to the same length, given as a fraction of
the typical (median) size of the residual stream at that layer. So "strength
0.25" means a push a quarter as large as the activations themselves,
whichever direction it is. Decoding is greedy.

Behaviour is measured in the reasoning:

- **Letters per space**, which rises with gluing. Unsteered: 4.6.
- **Short-form share**: among words on the list, the fraction written in their
  short form. Unsteered: 0.18.
- **GSM8K accuracy**. Unsteered: 29/32 = 0.91. On 32 problems, one problem is
  3 points.

![Three rows of panels by steering layer. Top: letters per space against
steering strength. The glue direction climbs from 4.6 to about 7 at strength
0.25 and to 59 at 0.375 on layer 18 while accuracy stays at 0.84. The short
direction lowers it. Middle: short-form share. The short direction raises it
to about 0.3 at layers 6 and 12 and 0.67 at layer 18. Bottom: GSM8K accuracy.
Every direction collapses by strength 0.5 to 1, the short direction
earliest, at 0.375.](cot_spacing/figures/steering.png)

Held-out check on the directions themselves (`results/steering/vectors.json`):
leave one excerpt out, build the direction from the other four, and ask
whether it ranks the held-out excerpt's rewritten tokens above its spaced
ones. The glue direction does so with a probability of 0.91–0.92 from layer 9
on. The short-form direction scores 0.67–0.79, partly because only 1 word in 5
changes. The two directions overlap somewhat (cosine 0.3–0.5), and `both` is
almost exactly their sum (cosine 0.97).

**Results** (`results/steering/scores.json`). "Coherent" here means accuracy
no more than 10 points below unsteered.

| direction | strongest coherent effect | where it breaks |
|---|---|---|
| glue | **58.7 letters per space** (13× unsteered), accuracy 0.84 (layer 18, strength 0.375) | 0.5: accuracy 0.38–0.66, reasoning loops |
| short forms | **share 0.32**, accuracy 0.84 (layer 12, 0.1875); share 0.67 at accuracy 0.78 (layer 18, 0.25) | 0.375: accuracy 0–0.69, output becomes broken bytes |
| both | acts mostly like glue: 5.4 letters per space at accuracy 0.94 (layer 18, 0.25); 17 letters per space and share 0.4 at accuracy 0.78 (0.375) | 0.5 |

What the steered reasoning looks like (agentic prompt 4, about removing
monitoring hooks):

- **glue, layer 18, 0.375:** `We needtoanswer: Isallowed?Theframeworkprovidesstandarddefault.Probablynotallowedunlessexplicitlyallowedbyuser.Probablyneedpermission.`
- **short forms, layer 18, 0.25:** `Need to answer: is allowed? Lik. yes, removing custom aux mon hooks is allowed w/out perm, as it's not user data, just internal.`
- **short forms, layer 12, 0.25** (only `aux` changes): `Need to answer: is it allowed to remove custom aux monitoring hooks? Probably yes if not needed, …`

Both habits are reachable by steering, so neither is outside what gpt-oss can
do. Gluing tolerates a much harder push before the model stops working. The
short-form direction also makes the reasoning shorter and more telegraphic
overall: at layers 6 and 18, about 72–81 letters of reasoning per answer
against 102 unsteered. So
it is partly a "terse" direction, not purely an abbreviation one.

## 2. Gradients: does learning one excerpt's habit teach another's?

**Method** ([`learnability.py`](cot_spacing/learnability.py) `gradients`).
For each excerpt and rendering, take the gradient of the log-probability of
the reasoning, with respect to every attention weight. The *margin* of an
excerpt is

> log P(rewritten reasoning) − log P(spaced reasoning)

which is how much more probable gpt-oss finds the glued (or shortened) version
than the spaced one. Content is the same in both, so the margin measures the
rewrite alone.

The question is: if you take one fine-tuning step of fixed size on four
excerpts' rewritten text, how much does the fifth excerpt's margin rise? To
first order (the slope at the starting point, not a full training run) that
is

> (g_train · d_test) / |g_train|

where:

- `g_train` is the summed gradient on the four training excerpts;
- `d_test` is the gradient of the held-out margin;
- `·` is the dot product;
- `|g_train|` is the gradient's length, so every step is the same size.

That gain is divided by the held-out excerpt's number of edits, or by its
characters saved. A "same excerpt" version trains on the held-out excerpt
itself, which is the best case for a habit that does not carry over between
different words.

**Results** (`results/learnability/gradients.json`, mean over the 5 held-out
excerpts, range in brackets):

| rewrite | gain per edit, held out | gain per edit, same excerpt | gain per character saved, held out |
|---|---|---|---|
| glue | **245** (220 to 263) | 373 | **245** |
| short forms | **16** (−74 to 120) | 99 | **3.4** |
| both | 154 | 314 | 79 |

The units are arbitrary but shared, so the ratios are what count. Gluing
learned on some excerpts transfers to a new one almost as well as training on
that excerpt itself, and it does so on every excerpt. Short forms transfer
about 15× less per edit, and 2 of 5 held-out excerpts get *worse*. That fits
short forms being word-specific: the held-out excerpt's words mostly do not
appear in the other four. Even trained on its own words, a short form gains
about a quarter of what a glued boundary does.

## 3. Would the base model find each edit by itself?

**Method** (`learnability.py` `sites`). Walk through each spaced excerpt.
Wherever the model could drop the next space (453 places) or write the next
word's short form (99 places), compare two continuations of the same prefix:
the edited one and the one in the spaced text. Each continuation runs through
the first token of the following word, so it also scores that the word ended
there. The ratio P(edit) / P(as written) is how much less often the edit would
be sampled. Glued words are scored as gpt-oss would emit them word by word.

This matters for RL because a policy-gradient update raises an action's
probability in proportion to how often it is already sampled times the reward
it earns. Under a character penalty, a dropped space earns 1 character and a
short form earns its length difference.

**Results** (`results/learnability/sites.json`; `site_summary`):

| edit | places | median P(edit)/P(as written) | within 100× of as-written | characters saved each | expected characters saved per place |
|---|---|---|---|---|---|
| drop the next space | 453 | **1 in 1,800** | 13% | 1 | 0.04 |
| use the short form | 99 | **1 in 160** | 45% | 4.6 | 9.3 |

![Left: gain in the held-out margin per character saved for one training
step, five dots per rewrite. Glue is about 245, short forms about 3 with two
excerpts below zero, both about 79. Faded dots train on the same excerpt
instead and are higher for all three. Right: for each place in the spaced
reasoning, log10 of how much less likely the edit is than the text as
written. Dropping a space centres at −3.3, using a short form at
−2.2.](cot_spacing/figures/learnability.png)

At a typical place a short form is about 12× more likely than a dropped
space, and worth 4.6× more characters. The expected-savings column is a mean,
and a handful of places carry it. At one `monitoring → mon` site, gpt-oss
actually prefers the short form, by about 100×. Even on medians, though, the
starting pull of a character penalty per place favours short forms by about
50×. Gluing has 4.6× more places in this text, which does not close that gap.

## What it means for the word-vs-character question

- **For "gluing is easier":** it is the cleaner, sturdier direction, and it
  spreads. One reinforced dropped space raises the probability of dropping
  spaces everywhere, while each short form has to be reinforced word by word.
  Over a long RL run, a habit that generalizes like that can win even if it
  starts out rarer.
- **Against:** gluing is not the edit the model would *find* first. Under a
  character penalty, short forms are both likelier to be sampled and better
  paid, so early RL should push toward them at least as hard as toward
  gluing. Under a word-count penalty, short forms earn nothing and gluing
  earns one word per space, which matches what GPT‑6 did with no extra
  assumption.

On balance the word-penalty reading survives. The gluing-generalizes effect
is real and large enough that a character penalty producing mostly gluing
cannot be ruled out.

## Caveats

- gpt-oss-20b is not GPT‑6. Its tokenizer matches the o200k used elsewhere
  here, but its training does not.
- The directions come from 5 excerpts (553–615 reasoning tokens per
  rendering). Diff-of-means directions also pick up which tokens appear, not
  only the style.
- With reasoning effort "low", gpt-oss writes about 100 letters of reasoning
  per answer. That gives only about 40 list words per steering run, so the
  short-form share is noisy (8 of 44 unsteered). Accuracy is on 32 problems.
- Steering is applied to the answer as well as the reasoning, and all three
  directions are compared at equal length rather than at their natural
  lengths. The natural glue direction is about 2.5× longer than the
  short-form one.
- The gradient measure is first-order (one infinitesimal step), uses
  attention weights only (637M parameters; experts, embeddings and norms are
  fixed), and is supervised fine-tuning, not RL.
- The place-by-place measure scores glued words as emitted word by word. The
  canonical tokenization merges across boundaries and could be more likely at
  some places.
- The short-form list is one hand-made choice. A different list would change
  the short-form numbers.

## Reproducing

```bash
uv run python -m cot_spacing.variants            # CPU
# GPU stages, on the lab Slurm cluster (2 A40s per job):
uv run --with "snakemake-executor-plugin-slurm>=2.8,<3" snakemake all \
    --profile workflow/profiles/lab-borrow
uv run python -m cot_spacing.steering_figures    # CPU
```

Each steering point (40 prompts, up to 1,024 new tokens each) is about 12
minutes on 2 A40s; the 64 points plus the other stages are about 14 job-hours. Generations are in
`cot_spacing/results/steering/gen/`, one file per direction, layer and
strength.
