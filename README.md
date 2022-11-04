# Hyphen

Implementation of [Public Wisdom Matters! Discourse-Aware Hyperbolic Fourier Co-Attention for Social-Text Classification](https://arxiv.org/abs/2209.13017), accepted at NeurIPS 2022, as an Oral (Spotlight) paper. 

<p align="center">
  <img width="600px" src="img/model.png" >
</p>

## Custom dataset processing

Generate the Abstract Meaning Representations for all user comments in a dataset:
```python
CUDA_VISIBLE_DEVICES=2 python3 amr/amr_gen.py --dataset politifact --max-comments 50
```

Modify attributes and instances variable names across all AMRs.
```python
python3 amr/amr_var.py --dataset politifact
```

Coreference resolution.
```python
python3 amr/amr_coref/amr_coref.py --dataset politifact
```

Adding the dummy node, and egdes and final step in merging AMRs to form macro-AMR.
```python
python3 amr/amr_dummy.py --dataset politifact
```

Convert the generated macro-amr to subgraphs in DGL format
```python
python3 amr/amr_dgl.py --dataset politifact --test-split 0.1
```
## Run

```python
CUDA_VISIBLE_DEVICES=2 python3 run.py --manifold PoincareBall --lr 0.001 --dataset politifact  --batch-size 32 --epochs 5 --max-sents 20 --max-coms 10 --max-com-len 10 --max-sent-len 10 --log-path logging/run
```

To run `Hyphen-euclidean`, use the following script:

```python
CUDA_VISIBLE_DEVICES=2 python3 run.py --manifold Euclidean --lr 0.001 --dataset politifact  --batch-size 32 --epochs 5 --max-sents 20 --max-coms 10 --max-com-len 10 --max-sent-len 10 --log-path logging/run
```

To run `Hyphen-euclidean w/o Fourier`, use the following script:

```python
CUDA_VISIBLE_DEVICES=2 python3 run.py --no-fourier --manifold Euclidean --lr 0.001 --dataset politifact  --batch-size 32 --epochs 5 --max-sents 20 --max-coms 10 --max-com-len 10 --max-sent-len 10 --log-path logging/run
```

To run `Hyphen-euclidean w/o Comments`, use the following script:

```python
CUDA_VISIBLE_DEVICES=2 python3 run.py --no-comments --manifold Euclidean --lr 0.001 --dataset politifact  --batch-size 32 --epochs 5 --max-sents 20 --max-coms 10 --max-com-len 10 --max-sent-len 10 --log-path logging/run
```
To run `Hyphen-euclidean w/o Content`, use the following script:

```python
CUDA_VISIBLE_DEVICES=2 python3 run.py --no-content --manifold Euclidean --lr 0.001 --dataset politifact  --batch-size 32 --epochs 5 --max-sents 20 --max-coms 10 --max-com-len 10 --max-sent-len 10 --log-path logging/run
```

## ✏️ Citation

If you think that this work is helpful, please feel free to leave a star ⭐️ and cite our paper:

```
@article{grover2022public,
  title={Public Wisdom Matters! Discourse-Aware Hyperbolic Fourier Co-Attention for Social-Text Classification},
  author={Grover, Karish and Angara, SM and Akhtar, Md and Chakraborty, Tanmoy and others},
  journal={arXiv preprint arXiv:2209.13017},
  year={2022}
}
```