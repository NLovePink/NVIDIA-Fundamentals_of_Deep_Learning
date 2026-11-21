# Data

- `data.py` — helper module used by the tutorials
- `doggy_door_images/` — images for the doggy-door classifier (module 02)
- `imagenet_class_index.json` — ImageNet class index

## fruits/ — assessment dataset (not in git)

`fruits/` is 1.9 GB / 13,599 PNGs, so it is intentionally not committed.
Recreate it from Kaggle:

- Source: https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification
- Unzip the dataset, then restructure it as:


data/fruits/train/{freshapples,freshbanana,freshoranges,rottenapples,rottenbanana,rottenoranges}/
data/fruits/valid/{freshapples,freshbanana,freshoranges,rottenapples,rottenbanana,rottenoranges}/

The zip's `dataset/test` folder becomes `fruits/valid`.
Verify with: `find data/fruits -name '*.png' | wc -l` → 13599
