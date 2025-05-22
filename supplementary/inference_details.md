### Time Series Generation

Inference with Prototype
```bash
python inference.py \
  --base config.yaml \ 
  --resume true \
  --ckpt_name ./checkpoints/ \ 
  --use_pam \
```

Inference with both Text and Prototype
```bash
python inference.py \
  --base config.yaml \ 
  --resume true \
  --ckpt_name ./checkpoints/ \
  --use_pam \
  --use_text  \
  --text_emb_dir \
```

Inference with neither Text nor Prototype
```bash
python inference.py \
  --base config.yaml \ 
  --resume true \
  --ckpt_name ./checkpoints/ \
```

Inference with influence guidance
```bash
python inference.py \
  --base config.yaml \ 
  --resume true \
  --ckpt_name ./checkpoints/ \
  --use_guidance \
  --uncond \
  --downstream_pth_path ./classifier/checkpoints/best_model.pt \
  --guidance_path ./classifier/data/guidance_tuple.pkl
```