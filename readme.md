[![PyPI](https://img.shields.io/pypi/v/scrainbow.svg)](https://pypi.org/project/scrainbow)

[![Downloads](https://pepy.tech/badge/scrainbow)](https://pepy.tech/project/scrainbow)

# Accurate cell type annotation for single-cell chromatin accessibility data via contrastive    learning and reference guidance

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

**train_set**  AnnData object of shape `n_obs` × `n_vars`with cell type labels. Rows correspond to cells and columns to genes.

**test_set**  AnnData object of shape `n_obs` × `n_vars` without cell type labels. Rows correspond to cells and columns to genes.

### Output:

**pred_labels**: Array object which contains cell type annotation results.

### Using tutorial:

```python
import scrainbow as rainbow
pred_labels = rainbow.run(train_set,test_set)
```

If there is reference data (AnnData object) can be incorporated, you can get annotation results via

```python
pred_labels = rainbow.run(train_set,test_set,refer_set)
```

If you want to identify the novel type:

```python
pred_labels = rainbow.run(train_set,test_set,pred_novel=True)
```

In this situation, the identified novel-type cells will be annotated as "novel".

The source datasets are available at [here](https://www.dropbox.com/s/40xeiqcbglzk6oj/data.rar?dl=0). 

