# InfoSeek fetch kit

Run this on a machine **with internet access** (your laptop or workstation). It
downloads only what the selected InfoSeek questions need and packs it into one
directory (~5–6 GB) that you copy to the construction server, which cannot reach
Google Cloud Storage, HuggingFace or Wikimedia.

## Requirements

* a checkout of this repository (the kit imports the split logic from `src/`)
* Python ≥ 3.10 with `pip install requests pillow huggingface_hub`
* free disk: ~8 GB for Wiki6M + the largest OVEN shard you download + ~6 GB for
  the kit (shards are deleted one by one after extraction)

## Run

```bash
cd EvidenceTree
python DatasetConstruct/fetch_kit/fetch_infoseek.py all --kit ~/infoseek_kit
```

or step by step (every step resumes where it stopped):

```bash
K=~/infoseek_kit
python DatasetConstruct/fetch_kit/fetch_infoseek.py annotations --kit $K
python DatasetConstruct/fetch_kit/fetch_infoseek.py splits --kit $K --dry-run   # report only
python DatasetConstruct/fetch_kit/fetch_infoseek.py splits --kit $K
python DatasetConstruct/fetch_kit/fetch_infoseek.py plan --kit $K               # which images, how big
python DatasetConstruct/fetch_kit/fetch_infoseek.py oven --kit $K               # query images
python DatasetConstruct/fetch_kit/fetch_infoseek.py wiki6m --kit $K             # 60K-entity corpus
python DatasetConstruct/fetch_kit/fetch_infoseek.py wiki-images --kit $K        # corpus thumbnails
python DatasetConstruct/fetch_kit/fetch_infoseek.py finalize --kit $K
```

| Step | Downloads | Notes |
| --- | --- | --- |
| `annotations` | InfoSeek val + entity map (~100 MB) | |
| `splits` | nothing | 2,000 eval + 20,000 seeds + 2,000 backups; same code as the server |
| `plan` | OVEN's id → source mapping (small) | lists exactly which images are needed (`oven/wanted_images.tsv`) and the shard sizes on HuggingFace (`oven/plan.json`) |
| `oven` | OVEN shards, one at a time (`--oven-shards`, default shard04/05/08) | extracts only the needed images (resized to 768 px), deletes each shard; missing ids are listed in `oven/missing_image_ids.txt` — add shards if many are missing |
| `wiki6m` | Wiki6M_ver_1_0.jsonl.gz (7.4 GB, resumable) | two streaming passes; keeps gold + hard negatives + random fill = 60K pages |
| `wiki-images` | one 512-px thumbnail per entity from Wikimedia | polite rate (2 workers), backs off on 429; rerun to retry failures |
| `finalize` | nothing | seeds without an image are replaced from the backups; writes `manifest.json` |

## Which images do I need?

You do not pick images; the questions do. Every InfoSeek question (a line of
`infoseek_val.jsonl`) names exactly one OVEN picture in its `image_id` field
(e.g. `oven_04990048`). `splits` chooses the questions — 2,000 for evaluation,
20,000 seeds and 2,000 backups — and the images you need are the distinct
`image_id`s of those questions (several questions often share one picture).

`plan` writes that list to `oven/wanted_images.tsv` (image id, its original
source path from OVEN's mapping, one question that uses it, the shard it was
found in once known, and whether it is already downloaded) and prints:

* how many distinct images are needed, by id prefix (`oven_04…`) and by source
  dataset (iNaturalist, Google Landmarks, FGVC-Aircraft, …);
* every shard in the OVEN repository with its size, marking the planned ones.

OVEN only distributes images inside large tar shards and publishes no
image → shard index, so single images cannot be fetched. `oven` therefore
downloads one planned shard at a time, extracts only the wanted images,
deletes the shard, and stops as soon as nothing is pending. It saves each
scanned shard's member list to `oven/shard_index/`, so later runs skip shards
known not to contain pending images, and `plan` can then say where each
image lives. The previous build found the InfoSeek validation images in
`shard04`, `shard05` and `shard08`, plus the FGVC-Aircraft archive — hence the
defaults. Images still missing after that are listed in
`oven/missing_image_ids.txt`; either add shards (`--oven-shards`) or let
`finalize` replace those seeds with backups.

Options worth knowing: `--image-prefixes oven_04,oven_05` restricts the selection to
images with those id prefixes (less download, same code path) — but if shards follow
source datasets this skews the questions toward some entity types, so prefer
downloading the planned shards; `--wiki6m <path>` reuses an
already-downloaded Wiki6M gzip; `--corpus-entities`, `--hard-per-gold`.

## Copy to the server

`raw/` (Wiki6M) and `oven/` (shard cache) are local caches — leave them out:

```bash
tar -cf infoseek_kit.tar --exclude=infoseek_kit/raw --exclude=infoseek_kit/oven -C ~ infoseek_kit
scp infoseek_kit.tar <server>:~/
```

On the server, from the repository root:

```bash
tar -xf ~/infoseek_kit.tar -C ~
python DatasetConstruct/data_prep/import_infoseek_kit.py --kit ~/infoseek_kit
```

The importer verifies the manifest checksums before writing anything.

## GQA scene graphs

GQA's `sceneGraphs.zip` is also unreachable from the server:

```bash
python DatasetConstruct/fetch_kit/fetch_infoseek.py gqa-scenegraphs --out ~/sceneGraphs.zip
scp ~/sceneGraphs.zip <server>:<repo>/data/raw/GQA/
```
