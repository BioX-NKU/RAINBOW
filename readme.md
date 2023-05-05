
# Accurate cell type annotation for single-cell chromatin accessibility data via contrastive learning and reference guidance

## Installation

Install RAINBOW from PYPI

```
pip install scrainbow
```

You can also install RAINBOW from GitHub via

```
git clone git://github.com/BioX-NKU/RAINBOW.git
cd RAINBOW
python setup.py install
```

The dependencies will be automatically installed along with RAINBOW.

## Quick Start

### Input:

**h5ad file** Files from the training set scCAS data and files from the scCAS data that need to be annotated.

### Output:

**pred_labels**: Array object which contains cell type annotation results.

### Using tutorial:

```python
import scrainbow as rainbow 
pred_labels = rainbow.run(train_path,test_path)
```

If there is reference data can be incorporated, you can get annotation results via

```python
pred_labels = rainbow.run(train_path,test_path,refer_path,refer=True)
```

If you want to identify the novel type:

```python
pred_labels = rainbow.run(train_path,test_path,pred_novel=True)
```

