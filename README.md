## Robust Deep Representation Learning for Road Crack Detection [PDF](# "Downdoald the paper from here")
# RoCDE

## Abstract
_Computer vision (CV) based inspection has recently attracted considerable attention and is progressively replacing traditional visual inspection which is subject to poor accuracy, high subjectivity and inefficiency. This paper, benefiting from hybrid structures of multichannel parallel convolutional neural networks (pCNNs), introduces a unique deep learning framework for road crack detection. Ideally, CNN-based frameworks require a relatively huge computing resources for accurate image analysis. However, the portability objective of this work necessitates the utilization of low power processing units. To that purpose, we propose robust deep representation learning for \textit{\textbf{\underline{Ro}}ad \textbf{\underline{C}}rack \textbf{\underline{De}}tection} \textsc{(RoCDe)} which uses a multichannel pCNNs. Bayesian optimization algorithm (BOA) was used to optimize the multichannel pCNNs training with the fewest possible neural network (NN) layers to achieve the maximum accuracy, improved efficiency and minimum processing time. The CV training was done using two distinct optimizers namely Adam and RELU on a sufficiently available datasets through image preprocessing and data augmentation. Experimental results show that, the proposed algorithm can achieve high accuracy around 95\% in crack detection, which is good enough to replace human inspections normally conducted on-site. This is largely due to well-calibrated predictive uncertainty estimates (WPUE). Effectiveness of the proposed model is demonstrated and validated empirically via extensive experiments and rigorous evaluation on large scale real world datasets. Furthermore, the performance of hybrid CNNs is compared with state-of-the art NN models and the results provides remarkable difference in success level, proving the strength of multichannel pCNNs._

## rocde.py 
This file contains RoCDE implemetation algorithm 
#### 

## Datasets
### Check details in dataset directory
## Dependencies
1. Tensoflow 2 and above
2. Keras 2.3.1
3. Python 3.6

## Dependencies
1. Tensoflow 2
2. Keras 2.3.1
3. Python 3.6
4. Matplotlib
5. tqdm

If you find this code useful in your research, please, consider citing our paper:

# Reference
```
  @inproceedings{wamburaicvip2021,
	AUTHOR = "Shadrack Fred Mahenge and Stephen Wambura and Ala A. Alsanabani", 
	TITLE = "Robust Deep Representation Learning for Road Crack Detection",
	publisher = {Association for Computing Machinery},
	booktitle  = {ICVIP 2021: Proceedings of The 5th International Conference on Video and Image Processing},
	address   = {Guangzhou, China},
	month  = {22-25 December},
	pages  = {},
	YEAR = {2021},
}
```
# License
RoCDE is distributed under Apache 2.0 license.

Contact: Stephen Wambura - stephen.wambura@dit.ac.tz
