# Hyphen

Implementation of [Public Wisdom Matters! Discourse-Aware Hyperbolic Fourier Co-Attention for Social-Text Classification](https://arxiv.org/abs/2209.13017), accepted at NeurIPS 2022, as an Oral (Spotlight) paper. 

## Run

```
CUDA_VISIBLE_DEVICES=2 python3 run.py --lr 0.001 --dataset politifact --manifold PoincareBall --batch-size 32 --epochs 5 --max-sents 20 --max-coms 10 --max-com-len 10 --max-sent-len 10 --log-path logging/run

```