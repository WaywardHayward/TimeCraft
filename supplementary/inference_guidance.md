# TimeCraft - Generation Modes

This document describes how to use various generation modes supported by the `inference.py` script within the TimeCraft framework.

---

## 5.1 Controllable Generation with Domain Prompts

In this mode, TimeCraft leverages learned **semantic prototypes** (also referred to as "domain prompts") to control the generation of synthetic time-series data. These prototypes encode structural or categorical properties of specific domains (e.g., medical, financial, climate), enabling the model to generate data that conforms to domain-specific characteristics without relying on explicit textual input.

This setup is ideal for:
- Cross-domain generation using a few representative samples
- Ablation studies (prototype-only guidance)
- Low-resource settings where textual descriptions are unavailable

**Example Command:**

```bash
python inference.py \
  --base config.yaml \
  --resume true \
  --ckpt_name ./checkpoints/ \
  --use_pam
```

> `--use_pam` enables prototype-based control  
> Back to [5.1](#51-controllable-generation-with-domain-prompts)

---

## 5.2 Controllable Generation with Domain Prompts and Text

This mode combines **textual conditioning** with semantic **prototypes** to offer more expressive and fine-grained control over the generated time series. By including natural language prompts, users can specify high-level trends (e.g., "a rising curve with seasonal dips") or domain-specific features (e.g., "heartbeat pattern after exercise").

This setting is especially powerful when:
- Users want to guide generation using natural language descriptions
- Additional domain knowledge needs to be injected beyond prototypes
- Interpretability and human-aligned generation are critical

**Example Command:**

```bash
python inference.py \
  --base config.yaml \
  --resume true \
  --ckpt_name ./checkpoints/ \
  --use_pam \
  --use_text \
  --text_emb_dir ./your_text_embedding_dir/
```

> `--use_pam` + `--use_text` = joint prototype-text conditioning  
> `--text_emb_dir` points to pre-computed text embeddings  
> Back to [5.2](#52-controllable-generation-with-domain-prompts-and-text)

---

## 5.3 Target-Aware Generation for Specific Downstream Tasks

This advanced mode enables **target-aware generation**, where the model produces time-series data that is **optimized to improve performance on a specific downstream task** (e.g., classification, detection). It integrates **gradient-based guidance** from a pre-trained classifier into the generation process, steering synthetic data toward task-relevant attributes.

This setup is useful when:
- You want synthetic data to enhance downstream task models
- You need to generate hard or rare samples for classifier robustness
- Controllability is required based on task-specific feedback

**Example Command:**

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

> `--use_guidance` enables classifier-informed generation  
> `--guidance_path` must point to a `.pkl` containing the guidance tuples  
> This assumes the classifier is already trained and stored at `downstream_pth_path`  
> Back to [5.3](#53-target-aware-generation-for-specific-downstream-tasks)

