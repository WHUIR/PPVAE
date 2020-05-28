# PPVAE
The official Keras implementation of ACL 2020 paper "Pre-train and Plug-in: Flexible Conditional Text Generation with Variational Auto-Encoders".

## Description
We put the conditional data (extracted from the original Yelp and News Title dataset) in the `conditional data` directory. You may want to download the original dataset to train PretrainVAE.

First, train Pretrain VAE with `pretrainVAE.py` for each dataset. Then you may want to train task-specific classifiers with `Classifier_for_evaluation.ipynb`. You can then train and evaluate PluginVAE with `PPVAE-single.ipynb` (for single-condition generation) and `PPVAE.ipynb` (for the standard setting).

Code credit: [Yu Duan](mailto:derrick.dy@alibaba-inc.com?cc=xucanwen@whu.edu.cn)
