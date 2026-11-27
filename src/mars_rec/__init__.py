"""mars-rec — supervised training framework for recommender systems.

A lightweight version of MARS-Gym focused on the supervised side of
recsys research: data preparation, deep-learning model training, and
ranking-metric evaluation. Built on Luigi (pipeline), PyTorch /
PyTorch Lightning (training) and pandas (data).

Top-level entry points:

- :mod:`mars_rec.meta_config` — the :class:`ProjectConfig`
  contract every project declares.
- :mod:`mars_rec.simulation.training` — the
  :class:`SupervisedModelTraining` Luigi task that drives the whole
  pipeline (also exposed as ``mars-rec run supervised``).
- :mod:`mars_rec.inference` — standalone top-k prediction
  (``mars-rec predict``) and re-evaluation
  (``mars-rec evaluate``) entry points.
- :mod:`mars_rec.data.task` — :class:`BasePrepareDataFrames`,
  the abstract Luigi task projects subclass to materialise CSVs.
- :mod:`mars_rec.data.dataset` — PyTorch ``Dataset`` classes
  with built-in negative sampling variants.
- :mod:`mars_rec.model.abstract` — :class:`RecommenderModule`,
  the abstract base class for every model.

See ``examples/`` for end-to-end projects (MovieLens, Last.fm,
Online Retail, Amazon Beauty, MovieLens-100K slate ranking).
"""
__version__ = '0.1.0'

import mars_rec.torch.metrics
