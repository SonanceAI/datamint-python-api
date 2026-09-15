Tutorials
=========

Hands-on examples for every step of the Datamint workflow, from your first upload to a deployed model. The same
notebooks live in the `notebooks/ directory <https://github.com/SonanceAI/datamint-python-api/tree/main/notebooks>`_
of the GitHub repository if you want to run them yourself.

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: Getting Started
      :link: tutorials/getting_started
      :link-type: doc

      :bdg-success:`Beginner`

      Upload resources to a project and explore data already there.

   .. grid-item-card:: Annotations
      :link: tutorials/annotations
      :link-type: doc

      :bdg-success:`Beginner`

      Import classification annotations and geometry (boxes, lines) to resources.

   .. grid-item-card:: Datasets
      :link: tutorials/datasets
      :link-type: doc

      :bdg-warning:`Intermediate`

      Build PyTorch-ready datasets, split them safely, and import already-labeled data.

   .. grid-item-card:: Experiment Tracking
      :link: tutorials/experiment_tracking
      :link-type: doc

      :bdg-warning:`Intermediate`

      Log runs and register models to MLflow through Datamint.

   .. grid-item-card:: Deployment
      :link: tutorials/deployment
      :link-type: doc

      :bdg-warning:`Intermediate`

      Deploy registered or external models, validate them, and run predictions.

   .. grid-item-card:: End-to-End Examples
      :link: tutorials/end_to_end/index
      :link-type: doc

      :bdg-danger:`Advanced`

      Complete pipelines from data upload to deployment on public datasets.

.. toctree::
   :hidden:

   tutorials/getting_started
   tutorials/annotations
   tutorials/datasets
   tutorials/experiment_tracking
   tutorials/deployment
   tutorials/end_to_end/index
