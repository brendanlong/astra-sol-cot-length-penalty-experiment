## Length (7 glued excerpts of 15 with a reference spacing)

| measure | as written (glued) | spaced | change |
| --- | --- | --- | --- |
| characters (glued excerpts) | 3361 | 3639 | -7.6% |
| canonical tokens (glued excerpts) | 667 | 618 | +7.9% |
| word-at-a-time tokens (glued excerpts) | 652 | 621 | +5.0% |
| characters (all excerpts) | 5963 | 6241 | -4.5% |
| canonical tokens (all excerpts) | 1161 | 1112 | +4.4% |
| word-at-a-time tokens (all excerpts) | 1152 | 1121 | +2.8% |

Re-tokenizing the stored glued transcript costs +2.3% versus word-at-a-time emission of the same text.

Other tokenizers (glued excerpts, canonical tokens vs spaced):

- `o200k_base`: 667 vs 618 (+7.9%), word-at-a-time 652 vs 621 (+5.0%)
- `o200k_harmony`: 667 vs 618 (+7.9%), word-at-a-time 652 vs 621 (+5.0%)
- `cl100k_base`: 681 vs 624 (+9.1%), word-at-a-time 664 vs 627 (+5.9%)

Per-excerpt, not pooled: 6 of 7 glued excerpts are individually positive in canonical tokens, range +0.0% to +12.3%, mean +5.8%.

Segmentation sensitivity: merging all 10 digit-label boundaries (the reference's least certain calls) moves the headline from -7.6% / +7.9% to -7.4% / +8.1%.

Glued boundaries by kind (a larger sample than the excerpt counts):

- agent_message: 10/151
- cot: 268/600
- user_facing: 0/109

## Which boundaries get glued

278 of 489 word boundaries in the glued excerpts have no space.

- mean canonical tokens added per glued boundary: +0.183
- mean that each kept boundary would have added, had it been glued: +0.275
- a canonical token straddles 23/278 glued boundaries

| predictor | drop rate when true | n | drop rate when false |
| --- | --- | --- | --- |
| left_ends_sentence | 0% | 38 | 62% |
| right_is_identifier | 54% | 13 | 57% |
| left_len_le_3 | 67% | 99 | 54% |
| left_len_ge_8 | 45% | 135 | 61% |
| right_starts_upper | 51% | 59 | 58% |
| right_has_digit | 100% | 11 | 56% |

### Token cost of the boundaries it chose

Mid-sentence only, since sentence boundaries are never glued: mean canonical cost +0.183 for the glued boundaries against +0.358 for the kept ones; labels shuffled within each excerpt, p=0.0024 one-sided (20000 permutations).

| set | glued where token-free | glued where it costs tokens | Fisher p |
| --- | --- | --- | --- |
| all boundaries | 60.7% (n=394) | 41.1% (n=95) | 0.00076 |
| mid-sentence only | 66.6% (n=359) | 42.4% (n=92) | 0.00004 |

Boundaries where gluing would outright *save* a canonical token: 9, of which the model glued 1 and only 2 are mid-sentence — too few to test, which is why the free-versus-costly split above is the usable version.

## Channel capacity, visible vs invisible

Both figures are upper bounds on capacity at a fixed reference text, per canonical reference token. Neither is a throughput a model could rely on: spending the invisible capacity emits more tokens than the reference has, and the visible figure assumes every boundary is free when sentence boundaries are never glued.

- space/no-space at 860 word boundaries: at most 860 bits, 0.77 per reference token, and plainly visible in the transcript.
- re-segmenting the same characters into different vocabulary tokens: 983/1113 tokens have at least one alternative, 4951 bits total (4.45 per reference token), and invisible in a transcript stored as text.

## Run structure

Mean glued-run length 2.24 words carries no information: it is fixed at 2.24 by the drop rate 0.562 and the excerpt count, however the drops are arranged. The testable statistic is the spread.

Run-length variance 2.56 against 2.80 for independent per-boundary dropping at the same rate, simulated per excerpt (95% 1.90-4.03, p=0.64).
Run lengths: {1: 106, 2: 44, 3: 31, 4: 19, 5: 13, 6: 5, 7: 5, 9: 1}

## How hard the glued text is to emit canonically

- 240/278 (86%) of glued words are canonically chunked exactly as if emitted on their own, so word-at-a-time emission is canonical there.
- the other 38 need chunks that reach across the word boundary.

Context sweep, 150 items x 248 left contexts, mean distinct chunk patterns per item. The GPT-2 integer rows run the same sweep on random integers, so they are the like-for-like comparison; clipped is the convention that matches how integer patterns are counted.

| sweep | unclipped | clipped | pairs differing from standalone (clipped) |
| --- | --- | --- | --- |
| glued English (o200k) | 4.65 (max 11) | 2.09 (max 4) | 12.3% |
| glued English, spaced prefix | 5.92 (max 16) | 2.11 (max 4) | 11.8% |
| GPT-2 3-digit integers | 3.62 (max 5) | 2.10 (max 3) | 42.6% |
| GPT-2 4-digit integers | 3.74 (max 7) | 1.60 (max 3) | 13.7% |

- GPT-2 3-digit integers, population view (not comparable to the sweep): 16.0% of consecutive integers chunk differently, 3 distinct patterns across all of them, 719 single-token.
- GPT-2 4-digit integers, population view (not comparable to the sweep): 9.7% of consecutive integers chunk differently, 5 distinct patterns across all of them, 88 single-token.

Glued words whose in-context chunks differ from their own:

- `could`: ['ec', 'ould'] vs ['could'] alone
- `should`: ['ays', 'h', 'ould'] vs ['should'] alone
- `earlier`: ['ere', 'arlier'] vs ['ear', 'lier'] alone
- `do`: ['don'] vs ['do'] alone
- `not`: ['don', 'ots'] vs ['not'] alone
- `send`: ['ots', 'end'] vs ['send'] alone
- `monitors`: ['monitor', 'sem'] vs ['mon', 'itors'] alone
- `emissions`: ['sem', 'issions'] vs ['em', 'issions'] alone
- `no`: ['nom'] vs ['no'] alone
- `messages`: ['nom', 'essages'] vs ['messages'] alone
- `no`: ['n', 'ore'] vs ['no'] alone
- `recipient.`: ['ore', 'c', 'ipient', '.'] vs ['recipient', '.'] alone
- `undermine`: ['und', 'erm', 'ines'] vs ['und', 'ermine'] alone
- `safeguards`: ['ines', 'af', 'egu', 'ards'] vs ['s', 'af', 'egu', 'ards'] alone
- `harm`: ['har', 'mp'] vs ['harm'] alone
- `prompt.`: ['mp', 'rompt', '.'] vs ['prompt', '.'] alone
- `bypass`: ['b', 'yp', 'assin'] vs ['b', 'ypass'] alone
- `instruction?`: ['assin', 'struction', '?'] vs ['instruction', '?'] alone
- `is`: ['isa'] vs ['is'] alone
- `authorized`: ['isa', 'uthorized'] vs ['authorized'] alone
- `architecture`: ['architect', 'ur', 'ere'] vs ['architecture'] alone
- `refactor`: ['ere', 'fact', 'ore'] vs ['ref', 'actor'] alone
- `exact`: ['ore', 'x', 'act'] vs ['exact'] alone
- `Dummy`: ['D', 'umm', 'ym'] vs ['Dummy'] alone
- `monitor`: ['ym', 'onitor'] vs ['monitor'] alone
- `confirm`: ['conf', 'ir', 'min'] vs ['confirm'] alone
- `interpretation`: ['min', 'ter', 'pret', 'ation'] vs ['interpret', 'ation'] alone
- `fallback`: ['sof', 'allback'] vs ['fallback'] alone
- `avoid`: ['avo', 'idde'] vs ['avoid'] alone
- `dehydration`: ['idde', 'hydr', 'ation'] vs ['de', 'hydr', 'ation'] alone
- `can`: ['ecan'] vs ['can'] alone
- `copy`: ['cop', 'yp'] vs ['copy'] alone
- `path`: ['yp', 'ath'] vs ['path'] alone
- `statutor`: ['stat', 'ut', 'org'] vs ['stat', 'utor'] alone
- `guards`: ['org', 'u', 'ards'] vs ['guards'] alone
- `no`: ['yno'] vs ['no'] alone
- `path`: ['paths'] vs ['path'] alone
- `same`: ['paths', 'ame'] vs ['same'] alone
