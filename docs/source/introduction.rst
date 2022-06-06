Getting Started
===============

Installation
------------

The ``DUNEdn`` package is open source and available at https://github.com/marcorossi5/DUNEdn.git

The package can be installed manually:

.. code-block:: bash

  git clone https://github.com/marcorossi5/DUNEdn.git DUNEdn
  cd DUNEdn
  pip install .

or if you are planning to extend or develop code replace last command with:

.. code-block:: bash

  pip install -e .

Motivation
----------

ProtoDUNE Single Phase (SP, `Technical Design Report <https://arxiv.org/abs/1706.07081>`_)
detector collects data containing inherent noise. This package implements deep
learning models to denoise ProtoDUNE-SP events.

Neural Networks
---------------

The package, as described in the `paper <https://doi.org/10.1007/s41781-021-00077-9>`_, 
implements two kind of networks for the denoising  task:


.. toctree::

   gcnn_intro
   uscg_intro

All the models are implemented with the help of the PyTorch library.

The networks support also exporting to ONNX format for device agnostic inference
via ONNX Runtime. More information at the relevant page:

.. toctree::

   onnx_intro

How to cite DUNEdn?
===================

When using this software in your research, please cite the following publication:

.. image:: https://zenodo.org/badge/484051775.svg
   :target: https://zenodo.org/badge/latestdoi/484051775

Bibtex:

.. code-block:: latex

  @software{Rossi_DUNEdn_2022,
    author = {Rossi, Marco},
    license = {GPL-3.0},
    month = {6},
    title = {{DUNEdn}},
    url = {https://github.com/marcorossi5/DUNEdn},
    version = {2.0.0},
    year = {2022}
  }

How to contribute?
==================

For more information on how to contribute, email the author at
`marco.rossi@cern.ch <marco.rossi@cern.ch>`_.

FAQ
===

Refer to examples in the
`GitHub <https://github.com/marcorossi5/DUNEdn/tree/main/examples>`_ folder for more
information.