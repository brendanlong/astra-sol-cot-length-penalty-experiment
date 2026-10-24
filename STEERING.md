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

1. Can gpt-oss be steered into each habit while still reasoning correctly?
2. Does training the habit on some excerpts teach it on another excerpt?
3. How often would unsteered gpt-oss make each edit by itself? Reinforcement
   learning (RL) can only reward what the model already samples.

**Short version.**

- **Both habits are within gpt-oss's reach by steering.** A gluing direction
  produces reasoning at least as glued as GPT‑6's while the model still solves
  arithmetic: a median of 29 letters per space, against 10–19 in the
  excerpts. A short-form direction raises the share of short forms in its
  answers from 11% to 65% at the same accuracy. Measured against each
  direction's own natural size, the short-form one can be pushed further
  before gpt-oss breaks.
- **Gluing carries over from one excerpt to another better than short forms.**
  Raising a rewrite's probability in four excerpts raises it in the fifth
  about 3× more per edit for gluing. Trained on its own excerpt, a short form
  is learned as fast as a glued boundary, so the difference is in how far the
  habit spreads.
- **Unsteered gpt-oss is more likely to write a short form than to drop a
  space.** At the typical place, the odds are about 1 in 160 against 1 in
  1,800. A short form also saves 4.6 characters to gluing's 1. So at the start
  of RL with a character penalty, the pull toward short forms should be the
  stronger one. A rough two-option model puts it at about 24× per place, but
  that is not a model of a real RL update.

So "gluing is just easier" holds only for how readily the habit *spreads*, and
only by about 3×. It does not hold for how readily the habit is *found*. This
weakens the word-penalty reading somewhat but does not overturn it. Everything
here is gpt-oss-20b, not GPT‑6, built from 5 excerpts.

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
`automatically → auto` and `with → w/`. Of the 99 shortened words, 56 are
words that are also shortened in at least one other excerpt.

Each rendering is put in gpt-oss's analysis channel (its reasoning). The user
turn before it is the first user prompt from the same incident in the report,
and the reasoning effort is "low".

## 1. Steering

**Method** ([`steering.py`](cot_spacing/steering.py)). A *difference-of-means
direction* is built per layer. For each excerpt, take the average
residual-stream activation over the reasoning tokens of, say, its glued
rendering, and subtract the same average for its spaced rendering. Then
average those differences over the 5 excerpts, so every excerpt counts
equally. The residual stream is the vector each layer passes to the next. The result points from "spaced" toward "glued" in
the model's internal space.

To steer, that direction is added at every position (prompt included) after
one layer (6, 12 or 18 of 24) while gpt-oss answers prompts it has not seen:

- 32 GSM8K arithmetic problems, which give an accuracy check on coherence;
- 8 short agentic prompts written in the report's register.

All three directions are scaled to the same length, given as a fraction of
the typical (median) size of the residual stream at that layer. So "strength
0.25" means a push a quarter as large as the activations themselves,
whichever direction it is. Decoding is greedy.

Measured per run of 40 answers:

- **Letters per space** in the reasoning, the median over answers. It rises
  with gluing. Unsteered: 4.7.
- **Short-form share**: among words on the list, the fraction written short,
  pooled over answers, and left blank when fewer than 20 list words appear.
  It is measured in the reasoning and, separately, in the answer.
  - At reasoning effort "low", gpt-oss writes only about 75 letters of
    reasoning, so the reasoning has about 40 list words per run. The answers
    have about 300, so their share is the well-measured one. Steering acts on
    both.
  - Code (inline or fenced) is skipped: `git diff` or `npm view <pkg>` is
    syntax, not a choice of word.
  - Unsteered: 8 of 44 in the reasoning, 30 of 279 in the answers.
  - Short forms are not visible inside glued runs of letters, so the share
    says little about glued output.
- **GSM8K accuracy.** Unsteered: 32/32. One problem is 3 points. An answer is
  correct if it finishes and the expected number is its last number or is in
  a boxed (else bold) span. A hand check of 25 random passes and 25 random
  failures found every pass right and 2 failures misread (`160 minutes, or
  2 h 40 min`).
- **Finished**: the answer closes within 1,024 new tokens. Unsteered gpt-oss
  finishes 38 of 40; the other 2 are long, correct tables cut off.
- **Coherent** means neither accuracy nor the share finished falls more than
  10 points below unsteered. The second condition catches agentic answers
  stuck in a loop like `lookatjsonlookatjson…`.

![Four rows of panels by steering layer, against steering strength. Row 1:
median letters per space in the reasoning. The glue direction rises from 4.7
to 6 at strength 0.25 and to 29 at 0.375 on layer 18 while still coherent.
Rows 2 and 3: short-form share in the reasoning and in the answer. The short
direction raises the answer share from 0.11 to about 0.6 while still
coherent. Row 4: GSM8K accuracy, 1.0 unsteered. Every direction collapses by
strength 0.5 to 1. Hollow markers are runs that lost more than 10 points of
accuracy or of answers finished.](cot_spacing/figures/steering.png)

**Results** (`results/steering/scores.json`):

| direction | strongest coherent effect | first incoherent strength |
|---|---|---|
| glue | reasoning **29 letters per space** (6× unsteered), answer 26; accuracy 0.94, 35 of 40 finished (layer 18, strength 0.375) | 0.375 at layer 6, 0.5 at layer 18. Layer 12 is uneven: 0.1875 loses 12 points, 0.25 is fine, 0.375 breaks |
| short forms | answer share **0.65** (132 of 204; unsteered 30 of 279), accuracy 0.94 (layer 18, 0.1875). Layer 12, 0.1875: 0.59. Layer 6, 0.25: 0.55. Reasoning share at most 0.33 (14 of 43) | 0.375 at layer 6, 0.25 at layers 12 and 18 |
| both | answer share 0.36 and reasoning 5.0 letters per space at accuracy 1.0 (layer 18, 0.25) | 0.375 |

At equal absolute strength, gluing survives a harder push than short forms
do. But the two directions are different sizes to begin with. The natural
difference of means for gluing is 2.4–2.6× longer than the one for short
forms at these layers. Measured in multiples of each direction's own natural
length, the largest coherent push is:

- gluing: 1.8–3.0×;
- short forms: 3.7–4.3×.

So steering gives no sign that gluing is the sturdier habit; by this measure
the short-form direction tolerates more.

The short-form direction also shortens reasoning a little: at its strongest
coherent settings, a median of 59–68 letters against 75 unsteered. So it is
partly a "terse" direction, not purely an abbreviation one. The `both`
direction acts as a weaker mix of the two.

What the steered reasoning looks like (agentic prompt 4, about removing
monitoring hooks):

- **glue, layer 18, 0.375:** `We needtoanswer: Isallowed?Theframeworkprovidesstandarddefault.Probablynotallowedunlessexplicitlyallowedbyuser.Probablyneedpermission.`
- **short forms, layer 18, 0.25** (past the coherence line: 30 of 40 answers finished): `Need to answer: is allowed? Lik. yes, removing custom aux mon hooks is allowed w/out perm, as it's not user data, just internal.`

In answers at layer 18, strength 0.1875, the most-used short forms are
`rm`, `impl`, `msg`, `ref`, `pkg`, `auth`, `env`, `arch`, `min` and `dir`.

**Held-out check on the directions.** Leave one excerpt out, build the
direction from the other four, and ask whether it ranks the held-out
excerpt's rewritten tokens above its spaced ones (`results/steering/vectors.json`).

- Glue: probability 0.87–0.92 from layer 9 on.
- Short forms: 0.65–0.79. Only 1 word in 5 changes, so most tokens are the
  same in both renderings, which caps this number.
- The renderings have different tokens, so this partly measures which tokens
  appear, not only the style.
- The two directions overlap somewhat (cosine 0.3–0.6), and `both` is almost
  exactly their sum (cosine 0.97 or more).
- The sweep was run with an earlier version of the directions that pooled
  tokens across excerpts, weighting each by its token count. At the steered
  layers the two versions agree to a cosine of 0.98 or more, so the sweep was
  not rerun.

## 2. Gradients: does training one excerpt's habit teach another's?

**Method** ([`learnability.py`](cot_spacing/learnability.py) `gradients` and
`step_gains`). For each excerpt and rendering, take the gradient of the
log-probability of its reasoning with respect to every attention weight. The
*margin* of an excerpt is

> log P(rewritten reasoning) − log P(spaced reasoning)

which is how much more probable gpt-oss finds the glued (or shortened) version
than the spaced one. The content is the same in both, so the margin measures
the rewrite alone.

Take one training step of fixed size on four excerpts and measure how much
the fifth excerpt's margin rises. To first order (the slope at the starting
point, not a full training run), a step along direction `t` changes it by

> (t · m_test) / |t|

where:

- `m_test` is the gradient of the held-out margin;
- `·` is the dot product;
- `|t|` is the step's length, so every step is the same size.

Two kinds of step:

- **Margin step:** `t` is the summed gradient of the four training margins.
  This rewards the rewrite over the original, as an RL reward for a shorter
  rendering would. It is the main measure.
  - Dividing by `|m_test|` as well gives a cosine, between −1 and 1. The
    cosine does not depend on how large either side's gradients are, so it is
    the fairest comparison between rewrites.
- **Fine-tuning step:** `t` is the summed gradient of plain log-probability
  of the four rewritten texts. This is weaker as a comparison:
  - the glued training text has 2.7× more edits;
  - most of the short-form text's gradient is about the content, not the
    rewrite.

Gains are divided by the held-out excerpt's number of edits. A "same excerpt"
version trains on the held-out excerpt itself.

**Results** (`results/learnability/gradients.json`; mean over the 5 held-out
excerpts, range in brackets):

| rewrite | margin step: cosine, held out | margin step: gain per edit, held out | margin step: gain per edit, same excerpt | fine-tuning step: gain per edit, held out |
|---|---|---|---|---|
| glue | **0.59** (0.42 to 0.69) | 276 (252 to 331) | 477 | 245 |
| short forms | **0.21** (0.15 to 0.25) | 98 (72 to 163) | 524 | 16 (−74 to 120) |
| both | 0.41 (0.28 to 0.56) | 156 | 399 | 154 |

The units of the gains are arbitrary but shared, so ratios are what count.

- **Gluing carries over about 3× better.** Raising the glued rendering in four
  excerpts raises it in the fifth by 0.59 of the most it could (the cosine),
  against 0.21 for short forms. Per edit, the gain is 2.8× larger.
- **Short forms do carry over**, positively on all 5 excerpts.
- **On a single excerpt, a short form is learned as fast as a glued
  boundary** (524 against 477 per edit). The gap is entirely in transfer.
  That gap is not just because the excerpts use different words: 56 of the 99
  shortened words also appear shortened elsewhere. Gluing is one rule that
  applies at every boundary, while each short form is its own word-to-word
  swap.
- The fine-tuning step gives a larger, 15× gap, and 2 of 5 excerpts go
  negative for short forms. That version is confounded as described above, so
  it is reported but not relied on.

## 3. Would the base model find each edit by itself?

**Method** (`learnability.py` `sites` and `site_summary`).

- Walk through each spaced excerpt. Wherever the model could drop the next
  space (453 places) or write the next word's short form (99 places), compare
  two continuations of the same prefix: the edited one, and the one in the
  spaced text.
- Each continuation runs through the first token of the following word, so
  it also scores that the word ended there.
- The ratio P(edit) / P(as written) is how much less often the edit would be
  sampled.
- Glued words are scored as gpt-oss would emit them word by word.

This matters for RL because a policy-gradient update raises an action's
probability in proportion to how often it is already sampled times the reward
it earns. Under a character penalty, a dropped space earns 1 character and a
short form earns its length difference.

Two summaries per place treat it as a choice between just the two options,
with q = P(edit) / (P(edit) + P(as written)):

- **Conditional expected saving**, q × characters saved. This is the saving
  per place *given* that one of the two is written, so it cannot exceed the
  characters on offer.
- **Gradient weight**, q × (1 − q) × characters saved. For a softmax choice
  between the two, this is how hard a reward of that many characters pushes
  on the edit's logit.

Both ignore the probability of everything else the model might write there,
and neither is a model of an actual RL update, with its sampling, correctness
reward and KL penalty. They show which way the starting pull points, and
roughly how lopsided it is.

**Results** (`results/learnability/sites.json`):

| edit | places | median P(edit)/P(as written) | within 100× of as-written | characters saved each | conditional expected saving per place | gradient weight per place |
|---|---|---|---|---|---|---|
| drop the next space | 453 | **1 in 1,800** | 13% | 1 | 0.014 | 0.008 |
| use the short form | 99 | **1 in 160** | 45% | 4.6 | 0.59 | 0.20 |

![Left: for each held-out excerpt, the cosine between the gradient that
raises the rewrite in the other four excerpts and the gradient that raises it
in that one. Glue averages 0.59, short forms 0.21 with all five positive,
both 0.41. Right: for each place in the spaced reasoning, log10 of how much
less likely the edit is than the text as written. Dropping a space centres at
−3.3, using a short form at −2.2.](cot_spacing/figures/learnability.png)

- At a typical place a short form is about 12× more likely than a dropped
  space, and it is worth 4.6× more characters.
- Per place, the conditional expected saving favours short forms by about
  42×, and the gradient weight by about 24×.
- Gluing has 4.6× more places in this text, which does not close the gap.
- No single place drives this. The most extreme, a `monitoring → mon` site
  where gpt-oss prefers the short form about 100 to 1, supplies 10% of the
  short-form total.
- The places GPT‑6 actually glued are no likelier to be glued by gpt-oss than
  the ones it left spaced (median 10^−3.32 against 10^−3.21).

## What it means for the word-vs-character question

- **For "gluing is easier":** gluing spreads. Rewarding a dropped space in one
  place raises the probability of dropping spaces everywhere, about 3× more
  than rewarding a short form raises other short forms. Over a long RL run, a
  habit that generalizes better can win even if it starts out rarer.
- **Against:** the base model would *find* short forms first. Under a
  character penalty they are likelier to be sampled and better paid, so the
  starting pull per place favours them by an order of magnitude (24–42× by
  the rough measures above). These numbers and the 3× transfer ratio measure
  different things, so they do not combine into a prediction of which habit
  an RL run ends up with. That would need an actual RL run.
- **The word-count alternative needs no extra assumption.** Under a
  word-count penalty, short forms earn nothing and gluing earns one word per
  dropped space, which matches what GPT‑6 did.
- Steering does not separate the two hypotheses. Both habits are reachable,
  and the short-form direction is, if anything, the sturdier one relative to
  its size.

On balance, the word-penalty reading survives, somewhat weakened. The
measurements here give no strong reason to expect a character penalty to
produce gluing without short forms, but they cannot rule it out.

## Caveats

- gpt-oss-20b is not GPT‑6. Its tokenizer matches the o200k used elsewhere
  here, but its training does not.
- The directions come from 5 excerpts (553–615 reasoning tokens per
  rendering). Difference-of-means directions also pick up which tokens
  appear, not only the style.
- Accuracy is on 32 problems, so about ±5 points, and the answer parser
  misreads some correct answers (2 of 25 audited failures). The reasoning
  short-form share rests on about 40 words per run, which is why the answer
  share leads.
- The gradient measure is first-order (one very small step) and uses
  attention weights only (637M parameters; experts, embeddings and norms are
  fixed). The margin step stands in for an RL update; it is not one.
- The place-by-place measure scores glued words as emitted word by word. The
  canonical tokenization merges across boundaries and could be more likely at
  some places.
- The short-form list is one hand-made choice. A different list would change
  the short-form numbers.

## Reproducing

The generations and steering vectors are on Hugging Face
([brendanlong/cot-spacing-steering](https://huggingface.co/datasets/brendanlong/cot-spacing-steering)),
not in git. To recheck every steering number and figure on CPU:

```bash
uv run python -m cot_spacing.steering_data      # downloads ~6 MB
uv run python -m cot_spacing.steering score cot_spacing/results/steering/gen/*.jsonl \
    --out cot_spacing/results/steering/scores.json
uv run python -m cot_spacing.steering_figures
```

To regenerate them instead, the GPU stages run on the lab Slurm cluster
(2 A40s per job):

```bash
uv run python -m cot_spacing.variants
uv run --with "snakemake-executor-plugin-slurm>=2.8,<3" snakemake all \
    --profile workflow/profiles/lab-borrow
```

Each steering point (40 prompts, up to 1,024 new tokens each) is about 12
minutes; the 64 points plus the other stages are about 14 job-hours. The
uploaded generations were made with an earlier version of the directions
(see section 1); a rerun uses the current ones.
