# PPVAE
The official Keras implementation of ACL 2020 paper "[Pre-train and Plug-in: Flexible Conditional Text Generation with Variational Auto-Encoders](https://arxiv.org/abs/1911.03882)".

## Citation
If you use the dataset or code in your research, please kindly cite our work:
```bibtex
@inproceedings{duan-etal-2020-pre,
    title = "Pre-train and Plug-in: Flexible Conditional Text Generation with Variational Auto-Encoders",
    author = "Duan, Yu  and
      Xu, Canwen  and
      Pei, Jiaxin  and
      Han, Jialong  and
      Li, Chenliang",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.23",
    pages = "253--262",
}
```

## Description
We put the conditional data (extracted from the original Yelp and News Title dataset) in the `conditional data` directory. You may want to download the original dataset to train PretrainVAE.

First, train Pretrain VAE with `pretrainVAE.py` for each dataset. Then you may want to train task-specific classifiers with `Classifier_for_evaluation.ipynb`. You can then train and evaluate PluginVAE with `PPVAE-single.ipynb` (for single-condition generation) and `PPVAE.ipynb` (for the standard setting).

Code credit: [Yu Duan](mailto:derrick.dy@alibaba-inc.com?cc=xucanwen@whu.edu.cn)
