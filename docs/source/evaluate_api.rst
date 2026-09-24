.. _evaluate_api:


Evaluating your Model
======================

:bdg-warning:`Intermediate`

``datamint.evaluate.Evaluator`` predicts and scores one or more models against a fixed
dataset. It works the same way whether
a model is registered (loaded locally), deployed (predicted remotely), or a bare
in-memory model instance you never registered at all.


Quick Start
-----------

.. code-block:: python

   from datamint import ImageDataset
   from datamint.evaluate import Evaluator

   dataset = ImageDataset(project="BUSI_Segmentation")
   evaluator = Evaluator(dataset=dataset)

   results = evaluator.evaluate(models=["unetpp_r34", "deeplabv3plus"])

   for model_name, result in results.items():
       print(model_name, result.scores.dataset)

The dataset is set once, at construction, and reused by every ``.evaluate()`` call on
that ``Evaluator`.

By default, ``evaluate()`` only scores predictions. Pass ``save_results=True`` 
to also save each model's predictions back as annotations on their resources, 
whether the model was run locally or through a deployed serving pod.


Model Resolution
-----------------

Each entry in ``models=[...]`` can be:

- a registered model name (``str``), resolved to its latest version,
- a ``Model`` or ``ModelVersion`` object, for a specific version,
- a bare ``BaseDatamintModel`` instance you never registered at all, or
- a ``(model, config_dict)`` pair -- ``config_dict`` is passed as ``predict()``
  params and merged into the logged hyperparameters.

.. code-block:: python

   results = evaluator.evaluate(models=[
       "unetpp_r34",
       ("deeplabv3plus", {"confidence_threshold": 0.6}),
   ])

If a model is both registered and deployed, ``evaluate()`` loads it locally by
default (``prefer_deployed=False``). Pass
``prefer_deployed=True`` to predict through the deployed serving pod instead.

Two versions of the same registered model can be compared in one call by passing
``ModelVersion`` objects instead of names -- the results dict and MLflow run names are
disambiguated by version (``model_name_v19``, ``model_name_v10``).

Hyperparameters
-----------------

For a model trained through a one-line trainer, hyperparameters are captured
automatically and logged with zero extra setup. 

For a model registered manually outside a trainer, or a bare in-memory instance, pass
them explicitly through the ``(model, config_dict)`` form shown above instead.

MLflow Logging
----------------

With ``log_to_mlflow=True`` (the default), each model gets its own MLflow run,
hyperparameters and Dice/IoU logged as run params/metrics. 
Every run from the same ``.evaluate()`` call shares an ``evaluation_id`` tag
(a UTC timestamp).

Runs land in the ``'Evaluation'`` experiment by default, every call across every
project accumulates there, so evaluations stay easy to find over time. Override the
name with ``experiment_name=`` if you want a dedicated one. 

.. code-block:: python

   results = evaluator.evaluate(
       models=["unetpp_r34"],
       log_to_mlflow=True,
       experiment_name="liver-segmentation-eval",
   )


Examples
----------------

The `evaluation tutorial notebook <https://github.com/SonanceAI/datamint-python-api/blob/main/notebooks/07_evaluating/01_evaluate_models.ipynb>`_
walks through one example per task type:

- **Segmentation** -- comparing two models with Dice/IoU.
- **Classification** -- a single-label model with accuracy, precision, recall and F1.
- **Detection** -- comparing two models with mAP50 and mAP50:95.
