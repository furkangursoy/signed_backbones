# Signed Backbone Extraction

This Python package provides tools to extract the signed backbones of intrinsically dense weighted networks.

Its counterpart in R can be found at https://github.com/furkangursoy/signed.backbones.


## Dependecies

1. [numpy](https://numpy.org/)
2. [pandas](https://pandas.pydata.org/)

Tested for numpy==1.20.1 and pandas==1.2.2 but should work with most versions.


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install signed_backbones.

```bash
pip install signed_backbones
```

## Example Usage

```python

import signed_backbones as sb
import pandas as pd

karate_net = pd.read_csv('karate.txt', header=None, sep='\t')

karate_sbb = sb.extract(karate_net, directed = False, significance_threshold = 2.576, vigor_threshold = (-0.1, 0.1))

```

See _examples/KarateViz.ipynb_ for visualizations of the original Karate network and its extracted signed backbone.

## Citation

If you find this software useful in your work, please cite:

Furkan Gursoy and Bertan Badur. ["Extracting the signed backbone of intrinsically dense weighted networks"](https://arxiv.org/abs/2012.05216).


## Reproducibility
The folder *reproducibility* contains Jupyter notebooks and R Markdown files that are used to create the figures and tables in the manuscript.

## Contributing

Please feel free to open an issue for bug reports, change requests, or other contributions.


## License

[MIT](https://choosealicense.com/licenses/mit/)

Packaged with: [Flit](https://buildmedia.readthedocs.org/media/pdf/flit/latest/flit.pdf)
