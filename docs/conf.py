# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys
import brainsmash

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))
sys.path.insert(0, os.path.abspath("../../"))
sys.path.insert(0, os.path.abspath("../brainsmash/"))
sys.path.insert(0, os.path.abspath("../brainsmash/mapgen/"))
sys.path.insert(0, os.path.abspath("../brainsmash/workbench/"))
sys.path.insert(0, os.path.abspath("../brainsmash/utils/"))

# autodoc_mock_imports = ['matplotlib', 'numpy', 'nibabel', 'pandas', 'scipy']

# -- Project information -----------------------------------------------------

project = 'BrainSMASH'
copyright = '2020, Joshua B. Burt, John D. Murray.'
author = 'Joshua B. Burt'

# The full version, including alpha/beta/rc tags
version = brainsmash.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx.ext.autosummary',
              ]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_param = True
napoleon_use_ivar = True
napoleon_use_rtype = True
add_function_parentheses = False
autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

source_suffix = '.rst'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

html_sidebars = {
    '**': [
        'relations.html',  # needs 'show_related': True theme option to display
        'searchbox.html',
    ]
}

numfig = True
numfig_secnum_depth = 1

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

html_logo = 'images/logo.png'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# # If true, `todo` and `todoList` produce output, else they produce nothing.
# todo_include_todos = False

# Output file base name for HTML help builder.
htmlhelp_basename = 'BrainSMASHdoc'
