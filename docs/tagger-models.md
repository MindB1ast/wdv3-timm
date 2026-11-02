# Tagger models

The model catalogue covers all 21 entries in the
[dbv4-full ranklist](https://huggingface.co/spaces/animetimm/dbv4-full-ranklist/blob/main/ranklist.csv),
plus the five newer `dbv4-full` releases listed by
[AnimeTimm](https://huggingface.co/animetimm), and all five
[SmilingWolf](https://huggingface.co/SmilingWolf) WD v3 timm taggers.
The catalogue snapshot was checked on 2026-11-02. Metadata compatibility
means the required timm resources are published; it does not mean every
checkpoint has been downloaded and run on this machine.

The aliases `vit`, `swinv2`, `convnext`, `big`, and `bigvit` retain their original
SmilingWolf WD v3 repositories. The identifier resolver also accepts full IDs,
including `animetimm/convnextv2_huge.dbv4-full` and
`animetimm/caformer_b36.dbv4-full`. Accepting an identifier alone does not establish
that its weights, labels and preprocessing are compatible.

The checked-in manifest `Scripts/tagger_models.json` records every repository
from both authors, its revision, file inventory and eligibility. The public
source snapshots in `audit/fixtures/tagger-catalog` make catalogue coverage
testable without network access. The catalogue is never fetched when opening
a model selection control.

The dropdown is sorted by the ranklist's Macro F1 at threshold 0.40, highest
first. Models absent from that ranklist follow by repository publication date
(newest first); dates are not inferred from last-modified timestamps. Legacy
aliases remain selectable, and the default stays `vit`. After updating the
extension, restart the ComfyUI backend and reload the page to refresh node
definitions in an already open workflow.

Older SmilingWolf WD v2 repositories without timm configurations/checkpoints
require a separate ONNX or TensorFlow backend and are excluded from this timm
change. SmilingWolf embedding models do not produce the required tag dictionary.
AnimeTimm `dbv4a`, `dbv4-full-witha`, `e621v1`, and `pexelsv0` releases are different
datasets/label spaces and are not selected automatically as `dbv4-full` models.
Each excluded repository remains recorded in the manifest with a reason.

Gated repositories require both acceptance of their conditions in the browser
and local Hugging Face authentication using the same account. Use
`.venv\Scripts\python.exe -c "from huggingface_hub import login; login()"`; never put a token in a notebook,
workflow or exported model metadata. An HTTP 401/403 is an access error,
not permission to substitute another model's preprocessing.

The serializable input contract retains `(C,H,W)` as separate positive axes,
CHW/NCHW layouts, float32 dtype, pre-normalization pixel range, RGB/BGR order,
the metadata source and the complete ordered operation list (including mean/std).
Its fingerprint changes whenever any effective contract field changes. This
representation does not itself implement an operation or establish parity with
a publisher's preprocessing; the runtime interpreter must validate it separately.

## Runtime preprocessing

The notebook `model` option accepts a full repository ID, for example:

```python
from Scripts import BatchTagging, ScriptOptions
results = BatchTagging(ScriptOptions(
    ImageFolder="./pictures", model="animetimm/caformer_b36.dbv4-full",
    save_txt=False, output_file=None,
))
```

Use `animetimm/convnextv2_huge.dbv4-full` for Huge or `vit` for legacy WD.
Only the chosen checkpoint is downloaded. Full images, YOLO/SAM regions and saved
region retagging share the same preparation code. Image geometry remains in its
original coordinate system; a model's input resize does not resize stored crops.

AnimeTimm uses the exact ordered `test` list in `preprocess.json`: supported
operations are `pad_to_size`, `resize`, `center_crop`, `maybe_to_tensor`, and
`normalize`. `pad_to_size` uses the publisher's `(width,height)` convention and
rescales to fit before centering on its background; resize/crop use torchvision
semantics. RGB channels are preserved through normalization. WD retains white
square padding, timm evaluation transforms and its original BGR channel order.
Each path checks float32 CHW input and the declared height/width. Unknown
operations or inconsistent metadata fail before inference.

Unit tests compare RGB and RGBA rectangular pixel inputs against an independent
implementation of the published Pillow/torchvision pipeline with zero tensor
difference. Batch-vs-single score tests allow `1e-6`; real model parity allows
`rtol=1e-5, atol=1e-6` for normal CPU floating-point differences. Scores use one
sigmoid; CSV row/category order is preserved, including tag names such as `nan`.
Additional `best_threshold` columns do not override the user's thresholds.

## Offline bundles and cache

New downloads go into `<model_folder>/<owner>/<repository>` (in ComfyUI, the
shared `models/wdv3/taggers` root). Known old WD directories
`<model_folder>/<repository>` remain readable without moving existing weights.
Repo IDs sharing a basename never share a new cache. `local_model_path` can
select a complete folder directly. A `tagger-source.json` file records the repo
ID and revision when known; selecting a different repository for that folder
raises an error. Hub downloads pin the companion files to the configuration's
resolved revision when available.

A complete bundle contains `config.json`, `selected_tags.csv`, one checkpoint
(`model.safetensors`, or `pytorch_model.bin` loaded with `weights_only=True`), and
`preprocess.json` for AnimeTimm. Disabling `download_missing` in ComfyUI lists
missing required files without network access. Local complete bundles work
offline in both interfaces. Only an HTTP 404 for an absent safetensors export
permits trying the bin checkpoint; access errors never trigger that fallback.

The profile fingerprint and dictionary identity participate in evidence caching.
Changing preprocessing, labels or weights invalidates raw scores; changing
only a filter threshold continues to reuse them. Loader `model_info` and result
provenance include the actual input contract. Re-run
`python audit/fetch_tagger_profiles.py` to refresh the pinned metadata fixtures
after repository access is granted; it records access failures separately.

Seven repositories currently return HTTP 403 pending author approval for this
account: `mobilenetv3_large_100.dbv4-full.r224`, `mobilevitv2_200.dbv4-full`,
`convformer_s36.dbv4-full`, `convformer_b36.dbv4-full`, `repvit_m2_3.dbv4-full`,
`eva_giant_patch14_560.dbv4-full`, and `vit_giantopt_patch16_siglip_384.dbv4-full`.
They remain selectable and use their own metadata when accessible; this access
restriction is not treated as evidence that their preprocessing was verified.
See `audit/fixtures/tagger-profiles/access.json` and the verification report.
