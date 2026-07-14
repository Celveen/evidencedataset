# Old Action Space Failure Analysis

Settings used:
- Baseline old action space: `text_search`, `image_search`, `answer`
- OCR off
- `cross_modal=false`
- Larger comparison run: requested N=30, actual available queries: InfoSeek 30, ScienceQA 25, MRAG-Bench 25, CRAG-MM 25

Source files:
- `data/pipeline_30q_r5_baseline_ocr_off/summary.md`
- `data/pipeline_30q_r5_baseline_ocr_off/*/trajectories.jsonl`
- `data/pipeline_30q_r5_cross_modal_on_ocr_off/*/trajectories.jsonl`

## Query-level result

Supported cross-modal datasets only:

| setting | avg EM | avg soft | avg F1 | trajectories | steps | rationale QC |
| --- | --- | --- | --- | --- | --- | --- |
| old action space | 0.531 | 0.564 | 0.528 | 153 | 334 | 334/334 |
| new action space | 0.664 | 0.676 | 0.552 | 389 | 1282 | 975/1282 |

Old-failed / new-solved counts:

| dataset | old failed but new solved |
| --- | --- |
| infoseek | 7 queries |
| scienceqa | 2 queries |
| mrag_bench | 4 queries |

CRAG-MM is excluded from this judgment because `cross_modal=true` is ignored there.

## Case 1: InfoSeek `infoseek_val_00039755`

Question:
`Which diocese does this building belong to?`

Gold:
`Archdiocese of Mercedes-Luján` / `Diocese of Mercedes-Luján`

Old action-space trajectories:

1. `text_search("Which diocese does this building belong to?")`
   - top evidence: `Mission Santa Barbara`, text about Diocese of Both Californias
   - second evidence: `Holy Spirit Cathedral, Minsk`
2. `answer`
   - wrong answers included `Diocese of Monterey-Los Angeles`, `Diocese of Monterey`, `Holy Spirit Cathedral, Minsk`

Why old action space struggles:
- The query contains a deictic visual reference: `this building`.
- Old action space has no valid way to map the query image to the building identity when the corpus has no image retriever for InfoSeek.
- It falls back to generic text search for `diocese + building`, which retrieves popular/distractor religious buildings.

New action-space successful trajectory:

1. `image_search(<state image>)`
2. `text_to_image("diocese this building belong")`
3. `text_search("Cathedral of Our Lady of Luján")`
   - top evidence: `Basilica of Our Lady of Luján`
   - text says it is part of the Archdiocese of Mercedes-Luján
4. `answer("Archdiocese of Mercedes-Luján")`

Diagnosis:
- This is a structural old-action-space failure.
- The missing operation is visual/entity bridging: image or text-to-image signal helps identify the building, then text search can answer.

## Case 2: InfoSeek `infoseek_val_00019262`

Question:
`What is the country of origin of this aircraft?`

Gold:
`Brazil`

Old action-space trajectories:

1. `text_search("SkyEurope OM-SAY")`
   - top evidence title: `Embraer EMB 120 Brasilia`
   - snippet is about incidents/references and does not directly expose country of origin
2. `answer("Slovakia")`

Another old trajectory:

1. `text_search("SkyEurope OM-SAY")`
2. `text_search("Sky Europe aircraft origin")`
   - retrieved unrelated `Umeda Sky Building`
3. `answer("Slovakia")`

New action-space successful trajectory:

1. `image_search(<state image>)`
2. `text_to_image("country origin this aircraft")`
3. `image_to_text(<state image>)`
   - evidence title: `Embraer EMB 120 Brasilia`
4. `answer("Brazil")`

Diagnosis:
- Mixed case.
- Old action space did retrieve the right aircraft title once, so this is not purely impossible.
- But old text search had weak snippets and no robust visual-to-entity grounding. Cross-modal actions made the Embraer identity more salient, allowing Qwen to answer Brazil.
- Fix options: new action space, better entity linking, or better text search snippets.

## Case 3: MRAG-Bench `mrag-22`

Question:
`Can you identify this animal? Choices: A. black-footed_ferret; B. European_Polecat; C. Steppe_Ferret; D. Domestic_Ferret`

Gold:
`A. black-footed_ferret`

Old action-space trajectories:

1. `image_search(<state image>)`
   - top evidence: `MRAG visual corpus image 010417`, text: `Unlabelled image from the MRAG-Bench retrieval corpus.`
   - second evidence: also unlabelled image
2. `answer("European_Polecat")` or `answer("B. European_Polecat")`

Why old action space struggles:
- Retrieved images are visually similar but unlabelled.
- The old action space can compare image-to-image, but it cannot turn the query image or candidate images into semantic class evidence.
- `text_search` is mostly useless in this MRAG visual corpus because many documents are image placeholders with text like `Unlabelled image`.

New action-space successful trajectory:

1. `image_to_text(<state image>)`
2. `text_to_image(question + choices)`
3. `image_search(<state image>)`
4. `answer("A. black-footed_ferret")`

Diagnosis:
- This is close to a structural old-action-space failure for visual fine-grained classification.
- Old image_search can retrieve similar images, but because the corpus evidence is unlabelled, it cannot provide class-name grounding.
- New cross-modal actions increase the chance of aligning option text with visual exemplars.

## Case 4: MRAG-Bench `mrag-3`

Question:
`Can you tell me the typical engine type for this car model and the cylinder liter size? Choices: A. 2.0L turbocharged inline-4; B. 3.0L V6; C. 2.5L inline-5; D. 1.8L turbocharged inline-4`

Gold:
`A. 2.0L turbocharged inline-4`

Old action-space trajectories:

1. `image_search(<state image>)`
   - top evidence: unlabelled/weakly labelled visual corpus images
2. sometimes `text_search("Audi RS6 2.0L turbocharged inline-4")`
   - retrieved zero-score generic MRAG corpus image placeholders
3. wrong answers: `D. 1.8L turbocharged inline-4`, `B. 3.0L V6`

New action-space successful trajectory:

1. `text_to_image(question + choices)`
2. `image_search(<state image>)`
3. `image_to_text(<state image>)`
4. `answer("A. 2.0L turbocharged inline-4")`

Diagnosis:
- Old action space does not have a strong operation for mapping car image -> car model -> engine spec.
- `image_search` gives visual neighbors, but text evidence is weak.
- New actions help by injecting option text into visual retrieval, although the evidence can still be noisy.

## Case 5: ScienceQA `scienceqa-smoke-14`

Question:
`What is the expected ratio of offspring with black eyes to offspring with red eyes? Choices: A. 1:3; B. 2:2; C. 4:0; D. 3:1; E. 0:4`

Gold:
`C. 4:0`

Old action-space trajectory:

1. `text_search(question)`
   - top evidence: `Science lesson 15`
   - second evidence: `Question context 15`, which includes the relevant genetics/Punnett-square context
2. `answer("D. 3:1")`

New action-space successful trajectory:

1. `image_to_text(<state image>)`
   - retrieves `Question context 15`
2. `image_search(<state image>)`
3. `answer("C. 4:0")`

Diagnosis:
- This is not a structural old-action-space failure.
- The old action space already retrieved the correct context, but Qwen selected the wrong ratio.
- The bottleneck is reasoning/answer selection, not retrieval action coverage.
- Better prompt, more rollouts, or answer verifier may help without adding new actions.

## Summary of old-action-space failure types

| failure type | example | old action space issue | truly action-space-limited? |
| --- | --- | --- | --- |
| Visual deictic entity not identified | InfoSeek building diocese | text query says `this building`; no image-to-entity bridge | yes |
| Image entity weakly identified | InfoSeek aircraft origin | old text search got weak/partial entity evidence | partially |
| Fine-grained visual classification with unlabelled corpus | MRAG animal/car examples | image_search returns unlabelled similar images | mostly yes |
| Correct evidence retrieved but wrong reasoning | ScienceQA genetics | text_search found correct context, answer wrong | no |
| CRAG-MM failures | CRAG-MM | cross-modal disabled/ignored; official indexes only | not evaluated here |

## Conclusion

Old action space is not fundamentally broken, but it has clear structural limits:

1. It cannot reliably resolve `this image/object/building/aircraft` into an entity when only text search is available.
2. It struggles when image_search returns visually similar but unlabelled images.
3. It cannot use answer-option text to guide visual retrieval.
4. It cannot fix reasoning errors when correct evidence is already retrieved.

Therefore, the right conclusion is not `always use the new action space`; it is:

- keep old action space as default for cheap and clean runs;
- selectively enable cross-modal actions for samples with visual deictic references, fine-grained visual classification, or text-option-to-image matching needs;
- do not enable cross-modal actions for pure text reasoning cases where old text_search already retrieves the relevant context.
