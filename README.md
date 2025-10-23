# FairGen: Debiasing Diffusion Models with Attribute Adapters

FairGen reduces bias in text-to-image diffusion models by attaching and training lightweight attribute-specific adapters only at cross-attention layers. During inference, a distribution indicator selects the adapter per sample so generated sets follow a prescribed target distribution (e.g., uniform over gender or race).

## Setup
```bash
conda create -n fairgen python=3.10
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Training
```bash
# Gender (train two adapters)
python train_fairgen.py --config_file configs/gender/male/config.yaml
python train_fairgen.py --config_file configs/gender/female/config.yaml

# Race (train four adapters)
python train_fairgen.py --config_file configs/race/WMELH/config.yaml
python train_fairgen.py --config_file configs/race/Asian/config.yaml
python train_fairgen.py --config_file configs/race/Black/config.yaml
python train_fairgen.py --config_file configs/race/Indian/config.yaml
```

## Inference with Distribution Indicator
```bash
# Gender (2 categories)
python infer_fairgen.py \
  --config configs/generation_gender.yaml \
  --adapter_paths output/gender/male output/gender/female \
  --base_model CompVis/stable-diffusion-v1-4 \
  --precision fp16 \
  --categories 2 --uniform

# Race (4 categories)
python infer_fairgen.py \
  --config configs/generation_race.yaml \
  --adapter_paths output/race/WMELH output/race/Asian output/race/Black output/race/Indian \
  --base_model CompVis/stable-diffusion-v1-4 \
  --precision fp16 \
  --categories 4 --uniform
```

 

