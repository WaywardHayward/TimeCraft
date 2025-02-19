

# TimeDP: Learning to Generate Multi-Domain Time Series with Domain Prompts

## Envirionment Setup
We recommand using conda as python enviroment manager:
```bash
conda env create -f environment.yml
```

After setting up python enviroment, please specify the path of data as `DATA_ROOT` environment variable. For example:
```bash
export DATA_ROOT=/path/to/your/data
```


## Examples
Training TimeDP:
```bash
python main_train.py --base configs/multi_domain_tsgen.yaml --gpus 0, --logdir ./logs/ -sl 168 -up -nl 16 --batch_size 128 -lr 0.0001 -s 0
```

Training without PAM:
```bash
python main_train.py --base configs/multi_domain_tsgen.yaml --gpus 0, --logdir ./logs/ -sl 168 --batch_size 128 -lr 0.0001 -s 0
```

Training without domain prompts (unconditional generation model):
```bash
python main_train.py --base configs/multi_domain_tsgen.yaml --gpus 0, --logdir ./logs/ -sl 168 --batch_size 128 -lr 0.0001 -s 0 --uncond
```

Visualization of domain prompts:
```bash
python visualize.py --base configs/multi_domain_tsgen.yaml --gpus 0, --logdir ./logs/ -sl 168 --batch_size 128 -lr 0.0001 -s 0 --uncond
```