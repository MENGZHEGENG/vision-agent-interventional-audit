# Visual verification reliability and self-training utility

This package reproduces a controlled comparison of visual correctness gates and
fixed-budget self-training. It provides program-generated shape and color
counting tasks, four pinned checkpoints, six admission rules, a no-update
reference, complete-epoch updates, and paired-scene analysis.

The code and scientific prediction records are separate from the manuscript.
Benchmark construction and recorded-output replay use Python's standard library.
Live inference and training require the optional model environment and suitable
CUDA hardware. Recorded-output replay recomputes measurements from saved model
responses; it does not perform a new model run.

## Lightweight benchmark

Use Python 3.11 or newer. No external packages are required for these commands.
Run them from this directory in a new workspace:

```bash
python3 scripts/reproduce.py benchmark --workspace output
python3 scripts/reproduce.py prepare --workspace output
```

The benchmark creates and validates 96 source scenes per split and three paired
variants per scene. The release keeps the split identifiers `evolution` and
`selection`: the paper calls these the update-selection and checker-measurement
splits, respectively. Held-out evaluation uses a fresh split. The configuration
records the exact generation seeds. Model inputs contain only sample
identifiers, image paths and questions. Program answers are stored separately
for evaluation.

The fixed identifier files are scientific inputs. Selection ranks
`SHA256(selection_seed + ":" + sample_id)`, so new opaque identifiers would
change its ordering. The provided mapping is checked against generated source
identifiers and image bytes. No private identifier-generation key is needed.

## Recorded-response replay

After benchmark construction and input preparation, recompute the captured
development and held-out measurements without loading model weights:

```bash
python3 scripts/reproduce.py replay --workspace output
```

Run this once in a fresh workspace after the two lightweight commands above.
It validates response parsing, fixed candidate identities, all image conditions,
selected and consumed examples, and complete held-out coverage. It recomputes
admission measurements, selected-target precision, arm accuracies, and paired
conditional intervals. Results appear in `output/replay_analysis/`.
The captured records contain scientific predictions and training exposure;
they are distinct from a new inference or training run.

## Current five-seed selection extension

`selection_seed_extension_v1/` is the current recorded-output add-on used by
the manuscript. It contains 110 task identities, 36 frozen selection sets,
two 288-row held-out label files, 31,680 label-free predictions, compact
per-task accuracies, and a standard-library validator. Run

```bash
python3 selection_seed_extension_v1/scripts/replay_selection_extension.py \
  --root selection_seed_extension_v1
```

The validator checks the recorded outputs and does not run a model. Labels are
joined only after the label-free selection matrix and selected IDs are frozen.
The recorded protocol does not independently audit a physical boundary that
would prevent labels from entering model selection, so the manuscript keeps
the corresponding statement descriptive and scoped.
The executed analysis used Python 3.11. Recorded replay explicitly preserves
its sequential floating-point summation on newer interpreters, so bootstrap
endpoints match that analysis without depending on later changes to `sum`.

## Original-seed target-source diagnostic

The paper also compares proposer-answer and program-answer training targets on
the same 48 selected examples for Random and Self-check at selection seed
`20261004`. Each paired condition uses three fitted seeds and the same 288
held-out examples. The saved predictions contain 24 runs and 6,912
label-free predictions. Replay the comparison with the Python standard
library:

```bash
python3 scripts/replay_target_source_diagnostic.py
```

The script checks complete prediction and label coverage, pairs the two target
sources by task and sample, and reports mean held-out accuracy and fitted-seed
ranges. Predictions outside the answer range 0--4 count as incorrect.

## Optional live model reproduction

Create a fresh `live_output` workspace for live execution. Recorded replay
creates candidate, checker, and selected-example files exclusively; a live run
cannot reuse the workspace written by replay. Generate and prepare live inputs
before proposing answers:

```bash
python3 scripts/reproduce.py benchmark --workspace live_output
python3 scripts/reproduce.py prepare --workspace live_output
```

Install the pinned direct dependencies in a separate Python 3.11 environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements_model.txt
```

Choose a model-cache directory with space for all four checkpoint snapshots.
The following command downloads the exact revisions in the configuration:

```bash
.venv/bin/python scripts/reproduce.py models --model-cache model_cache
.venv/bin/python scripts/reproduce.py propose --workspace live_output --model-cache model_cache
```

Run checking separately for each family, checkpoint and image condition. For
example:

```bash
.venv/bin/python scripts/reproduce.py check --workspace live_output --model-cache model_cache --family shape_count --model internvl3_5_2b_instruct --condition original
```

Repeat that command for both configured families, all four configured model keys,
and `original`, `no_image`, and `shuffled`. Then select the training examples:

```bash
python3 scripts/reproduce.py select --workspace live_output
```

Each rule retains 48 examples from distinct source scenes. The primary external
checker is InternVL3.5-2B-Instruct. Random selection uses no verdict;
self-checking uses Qwen3-VL-2B-Instruct; consensus requires both of those
checkpoints. The two other rules use Qwen3-VL-8B-Instruct and InternVL3.5-8B.
In the recorded configuration, `same_model` is the Qwen3-VL-2B self-check,
`independent` is the InternVL3.5-2B external check, and `independent_8b` is
the InternVL3.5-8B external check. `independent` is an internal arm name; it
does not assert statistical independence.

Run training separately for every configured family, seed and arm. For example:

```bash
.venv/bin/python scripts/reproduce.py train --workspace live_output --model-cache model_cache --family shape_count --arm independent --seed 20261005
```

Each command starts from fresh Qwen3-VL-2B-Instruct weights. Updating arms use
48 singleton optimizer steps, one complete selected-example epoch, attention
LoRA rank 8, alpha 16, dropout 0.05, AdamW defaults with learning rate 0.0001,
and gradient clipping at 1. The no-update arm skips adaptation. Evaluation uses
the merged adapter and freezes response bytes before opening program answers.
The three configured seeds change initialization and example order; each arm's
selected set is fixed across those seeds.

Development responses use deterministic decoding with 16 new tokens, an entire
integer response from 0 to 4, and exact `YES` or `NO` verification. Held-out
evaluation preserves the original 64-token decoding and first standalone
integer parser. InternVL uses one image resized to 448 by 448 with the specified
normalization, not dynamic image tiling.

The original 42-result configuration below is retained as a baseline replay
package. It is not the current five-seed extension used for the manuscript.
After all 42 family/seed/arm baseline results exist, recompute the comparisons:

```bash
python3 scripts/reproduce.py aggregate --workspace live_output
```

This command checks complete coverage, selected and consumed identifiers,
the seeded order, frozen and scored responses, image/answer digests, and fixed
checkpoint identities. It resamples 96 scene pairs with 20,000 percentile draws,
averaging three variants and three fitted training seeds within each pair.
Intervals condition on the realized selected sets and fitted seeds. They do not
estimate variation across new training sets or future optimization seeds.
The comparisons are descriptive; an interval containing zero is not evidence
of equivalence.

## Output units and groups

JSON accuracies, rates, precision values, paired differences, and confidence
interval endpoints are fractions. Multiply by 100 to express them as %. Counts
remain counts. For example, accuracy `0.75` is 75%, and an accuracy difference
`0.05` is an absolute difference of 5% in accuracy.

`pair_count` and bootstrap `clusters` count source-scene groups: 96 per family.
Each group has three input variants. Subgroup counts are input variants within
one training seed. Each edit-kind subgroup contains 96 variants, one per
source-scene group; an answer subgroup can contain multiple variants from one
group. The 864 predictions per arm are three evaluations of the same 288 input
variants across training seeds, not 864 independent groups.

## Source and rights

`SOURCE_REVISION` identifies the scientific worker revision. The portable
interfaces change path management and indexing while preserving the scientific
generation, model calls, training, parsing and pair-bootstrap functions.
The package's file index binds the actual portable source and configuration.

All scene images are generated by the provided renderer. No third-party image
dataset or model weights are distributed. Checkpoint access and dependency terms
remain those of their respective providers. No software or generated-data reuse
license is specified here; author-approved terms are still needed before general
redistribution. The package is not described as openly licensed.

Live execution of the portable model pipeline is a separate verification step
from benchmark generation and recorded-output replay. Floating-point libraries,
hardware and model implementations can affect new outputs; retain the pinned
versions and compare per-example responses when reproducing the study.
