Installation Guide
==================

Installing BrainSMASH requires:

- Python 3+
- `numpy <http://www.numpy.org/>`_
- `pandas <https://pandas.pydata.org/>`_
- `nibabel <http://nipy.org/nibabel>`_
- `matplotlib <https://matplotlib.org/>`_
- `scikit-learn <http://scikit-learn.org/stable/index.html>`_

BrainSMASH was developed and tested using Python 3.7. Backwards-compatibility
with other Python3 versions is expected but not guaranteed. All Python package version dependencies are handled automatically if installing
BrainSMASH via ``pip`` (described below).

.. note::
   Using ``nibabel`` versions >2.1.0 will break the code, though
   this incompatibility will be resolved in a future release.

Dependencies
------------

In addition to the Python packages listed above, *if and only if* you wish to use
the additional utilities provided for Connectome Workbench users, you must have
Connectome Workbench installed with the ``wb_command`` executable locatable in your
system PATH environment variable. For Mac users with the unzipped ``workbench``
directory (downloaded `here <https://www.humanconnectome.org/software/get-connectome-workbench>`_ )
located in your ``Applications`` folder, this can be achieved by simply
adding ``export PATH="/Applications/workbench/bin_macosx64:$PATH"`` to your ``.bash_profile``.

.. note:: For non-login interactive shells (which does not include Terminal), you may also need
  to configure ``PATH`` in your ``.bashrc`` file, as described above. Alternatively, for a single session only,
  you may run ``echo 'export PATH=$PATH:/Applications/workbench/bin_macosx64' >> ~/.bashrc``.

Installation
------------

BrainSmash is most easily installed via the Python package manager ``pip``:

.. code-block:: bash

        pip install brainsmash

You may also clone and install the source files manually via the `GitHub repository <https://github.com/jbburt/brainsmash>`_:

.. code-block:: bash

        git clone https://github.com/jbburt/brainsmash.git
        cd brainsmash
        python setup.py install

If you're concerned about clashing dependencies, BrainSMASH can be installed
in a new ``conda`` environment:

.. code-block:: bash

        conda create -n brainsmash python=3.7
        conda activate brainsmash
        pip install brainsmash
