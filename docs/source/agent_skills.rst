Agent Skills
============

Datamint provides skills that teach AI coding agents (Claude Code, Cursor, OpenCode, ...) how to use the SDK correctly.
Once installed, the agent loads the right skill automatically based on what you ask it to do.

Available skills
----------------

+----------------------------+--------------------------------------------------------------------------+
| Skill                      | Covers                                                                   |
+============================+==========================================================================+
| ``datamint-upload``        | Uploading resources, segmentations, AI pre-annotations and worklists     |
+----------------------------+--------------------------------------------------------------------------+
| ``datamint-data-access``   | Exploring, filtering, loading and downloading data and annotations       |
+----------------------------+--------------------------------------------------------------------------+
| ``datamint-training``      | Datasets, splits, built-in trainers and benchmarking                     |
+----------------------------+--------------------------------------------------------------------------+
| ``datamint-custom-models`` | Bringing your own model or training loop into Datamint                   |
+----------------------------+--------------------------------------------------------------------------+
| ``datamint-experiments``   | MLflow tracking, runs, model registry and evaluation                     |
+----------------------------+--------------------------------------------------------------------------+
| ``datamint-deploy``        | Deployment, remote and local inference                                   |
+----------------------------+--------------------------------------------------------------------------+
| ``datamint-end-to-end``    | The full pipeline, from upload to deployed model                         |
+----------------------------+--------------------------------------------------------------------------+

Installation
------------

**Claude Code**

.. code-block:: bash

   claude plugin marketplace add SonanceAI/datamint-python-api
   claude plugin install datamint@datamint

To update to the latest version:

.. code-block:: bash

   claude plugin marketplace update datamint

**Other agents** (any agent that reads ``.claude/skills/``)

.. code-block:: bash

   # all skills
   npx skills add SonanceAI/datamint-python-api

   # a single skill
   npx skills add SonanceAI/datamint-python-api --skill datamint-upload
