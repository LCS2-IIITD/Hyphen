# amr_coref

**A python library / model for creating co-references between AMR graph nodes.**

## About
amr_coref is a python library and trained model designed to do co-referencing
between [Abstract Meaning Representation](https://amr.isi.edu/) graphs.

The project follows the general approach of the [neuralcoref project](https://github.com/huggingface/neuralcoref)
and it's excellent
[blog on the co-referencing](https://medium.com/huggingface/how-to-train-a-neural-coreference-model-neuralcoref-2-7bb30c1abdfe).
However, the model is trained to do direct co-reference resolution between graph nodes and does not depend on
the sentences the graphs were created from.

The trained model achieves the following scores
```
MUC   :  R=0.647  P=0.779  F₁=0.706
B³    :  R=0.633  P=0.638  F₁=0.630
CEAF_m:  R=0.515  P=0.744  F₁=0.609
CEAF_e:  R=0.200  P=0.734  F₁=0.306
BLANC :  R=0.524  P=0.799  F₁=0.542
CoNLL-2012 average score: 0.548
```

## Installation and usage
There is currently no pip installation. To use the library, simply clone the code and use it in place.

The pre-trained model can be downloaded from the assets section in [releases](https://github.com/bjascob/amr_coref/releases).

To use the model create a `data` directory and un-tar the model in it.

The script `40_Run_Inference.py`, is an example of how to use the model.


## Training
If you'd like to train the model from scratch, you'll need a copy of the
[AMR corpus](https://catalog.ldc.upenn.edu/LDC2020T02).
To complete training, run the scripts in order.
- 10_Build_Model_TData.py
- 12_Build_Embeddings.py
- 14_Build_Mention_Tokens.py
- 30_Train_Model.py.

You'll need `amr_annotation_3.0` and `GloVe/glove.6B.50d.txt` in your `data` directory

The first few scripts will create the training data in `data/tdata` and the model training
script will create `data/model`. Training takes less than 4 hours.


## Advancement
I'm considering this a beta release and I'd like to get feedback on usefulness and
any needed features. Please send me feedback on any changes / improvements.  Eventually
I will move this code into [amrlib](https://github.com/bjascob/amrlib) but wanted to
post it here first, until it sees a little utilization.

If you're interested in working towards improving scores, feel free to experiment.  I'd
be happy to answer questions and/or lend assistance.  If scores can be improved, I'll
plan to add those changes back into the project.
