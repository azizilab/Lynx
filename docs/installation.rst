Installation
============

LYNX targets **Python 3.9.5**. The deep-learning stack (PyTorch +
PyTorch Geometric) and the spatial-omics libraries are the trickiest parts to
reproduce, so a conda environment is recommended; a pip-only track is provided
as a fallback.

.. note::

   The PyTorch Geometric companion wheels (``torch-sparse``,
   ``torch-geometric``) are **not on plain PyPI** for CUDA builds. They must be
   installed from the PyG find-links index matching your ``torch`` and CUDA
   versions (examples below use ``torch 2.3.1`` + CUDA 12.1).


Conda (recommended)
-------------------

.. code-block:: bash

   conda env create -f environment.yml
   conda activate lynx

   # PyG companion wheels (match torch 2.3.1 + your CUDA build):
   pip install torch-sparse==0.6.18 torch-geometric==2.6.1 \
       -f https://data.pyg.org/whl/torch-2.3.1+cu121.html

   # Install the lynx package itself (and remaining pip deps):
   pip install -e .


Pip (fallback)
--------------

.. code-block:: bash

   python --version          # should report 3.9.5

   pip install torch==2.3.1 torchvision==0.18.1
   pip install torch-sparse==0.6.18 torch-geometric==2.6.1 \
       -f https://data.pyg.org/whl/torch-2.3.1+cu121.html

   pip install -r requirements.txt
   pip install -e .


Verify the install
------------------

.. code-block:: bash

   python -c "import torch, torch_geometric, squidpy, scanpy, lynx; print('ok')"

If you are running directly from the project's bundled interpreter, substitute
``env/bin/python`` for ``python`` in the commands above.


Installing into an existing interpreter
=======================================

When installing into a specific interpreter such as ``env/bin/python``, note
that LYNX builds with the **Poetry** backend, so the build environment needs
``poetry-core``. Always invoke the **target interpreter's own pip** — ``pip`` on
your ``PATH`` may belong to a different virtualenv:

.. code-block:: bash

   env/bin/python -m pip install poetry-core
   env/bin/python -m pip install -e . --no-deps   # --no-deps keeps the pinned stack untouched

Verify it resolves from any working directory (not just the repo root):

.. code-block:: bash

   cd /tmp && env/bin/python -c "import lynx; print(lynx.__version__)"


Using the tutorials in Jupyter
==============================

The tutorial notebooks ``import lynx`` directly, so the running **kernel must be
the interpreter you installed LYNX into** (e.g. ``env``). Register that
interpreter as a named kernel and select it in the notebook:

.. code-block:: bash

   env/bin/python -m ipykernel install --user --name lynx-env \
       --display-name "Python (LYNX env)"

Then open a tutorial and choose the **Python (LYNX env)** kernel.
