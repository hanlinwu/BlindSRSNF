## Conditional Stochastic Normalizing Flows for Blind Super-Resolution of Remote Sensing Images

by Hanlin Wu, Ning Ni, and Libao Zhang, details are in [paper](https://arxiv.org/abs/2210.07751).

## Usage

### Clone the repository:
```
git clone https://github.com/hanlinwu/BlindSRSNF.git
```

## Requirements:
- pytorch==1.13.0
- pytorch-lightning==1.5.5
- numpy
- opencv-python
- easydict
- tqdm

### Test with our pretrained models
1. Download the checkpoints from this [url](). 
2. Unzip the downloaded file, and put the files on path: `logs/`


### Train:

1. Download the training datsets from this [url](). 
2. Unzip the downloaded dataset, and put the files on path: `load/`
3. Do training:

   For ansio degradation:

   ```
   python train.py --config configs/blindsrsnf_aniso.yaml
   ```
   For iso degradation:

   ```
   python train.py --config configs/blindsrsnf_iso.yaml
   ```
   For ansio degradation with the WorldStrat dataset:

   ```
   python train.py --config configs/blindsrsnf_iso.yaml
   ```

### Test:

```
python test_diff.py --checkpoint logs/your_checkpoint_path
```
or
```
sh scripts/test_diff_aniso.sh logs/your_checkpoint_path
```