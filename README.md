# Hyphen

Implementation of [Public Wisdom Matters! Discourse-Aware Hyperbolic Fourier Co-Attention for Social-Text Classification](https://arxiv.org/abs/2209.13017), accepted at NeurIPS 2022, as an Oral (Spotlight) paper. 

## Dataset processing

Generate the Abstract Meaning Representations for all user comments in a dataset:
```
CUDA_VISIBLE_DEVICES=2 python3 amr/amr_gen.py --dataset politifact --max-comments 50
```

Modify attributes and instances variable names across all AMRs.
```
python3 amr/amr_var.py --dataset politifact
```

Coreference resolution.
```
python3 amr/amr_coref/amr_coref.py --dataset politifact
```

Adding the dummy node, and egdes and final step in merging AMRs to form macro-AMR.
```
python3 amr/amr_dummy.py --dataset politifact
```

Convert the generated macro-amr to subgraphs in DGL format
```
python3 amr/amr_dgl.py --dataset politifact --test-split 0.1
```
## Run

```
CUDA_VISIBLE_DEVICES=2 python3 run.py --lr 0.001 --dataset politifact --manifold PoincareBall --batch-size 32 --epochs 5 --max-sents 20 --max-coms 10 --max-com-len 10 --max-sent-len 10 --log-path logging/run

```

