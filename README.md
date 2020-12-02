# Personalized DNN Recommendation Systems based on User-dependent Gating Mechanism

[![Python Versions](https://img.shields.io/pypi/pyversions/deepctr.svg)](https://pypi.org/project/deepctr) [![TensorFlow Versions](https://img.shields.io/badge/TensorFlow-1.4+/2.0+-blue.svg)](https://pypi.org/project/deepctr)

Most existing works ignore a fact that a fixed number of model parameters is difficult to model the diversified user interests well. User-dependent Gating (UDG) mechanismcan can transform the base model parameters dynamically based on the current user, the item to predict for, as well as the click history, and therefore can model the diversity of the user interests effectively.

The code of this work is based on the open-source CTR models framework named [DeepCTR](https://github.com/shenweichen/DeepCTR).

## Usage

```python
cd examples/
python udg.py DeepFM_UDG amazon w normal 128 0

DeepFM_UDG -> model name
amazon -> dataset name
w -> w or wo represents if feed the user profile(e.g. gender, age) into model
normal -> normal or focal represents focal loss or binary crossentropy loss
128 -> the UDG embedding size 
0 -> which GPU to use
```

## Note

Due to github's limitation on the size of uploaded files, we make a small sample of amazon dataset for code test only. If you want to train and inference on your own dataset, please pre-process the data according to the code requirements.
