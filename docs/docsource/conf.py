import os
import sys
import nova_utils
sys.path.insert(0, os.path.abspath('../../nova_utils/'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NOVA-Utils'
copyright = '2023, Dominik Schiller'
author = 'Dominik Schiller'
release = nova_utils.__version__


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx_rtd_theme',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'myst_parser',
    'sphinxarg.ext'
]
source_suffix = ['.rst', '.md']
autodoc_typehints = "description"
#napoleon_use_param = False
napoleon_google_docstring = True  # Enable parsing of Google-style pydocs.
napoleon_use_ivar = True  # to correctly handle Attributes header in class pydocs

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'press'
#html_static_path = ['_static']

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'globaltoc_collapse': True,
    'globaltoc_maxdepth': -1,
}
html_static_path = ['_static']
html_sidebars = {"**": ["globaltoc.html", "localtoc.html", "searchbox.html"]}


#autodoc_default_options = {
#     'undoc-members': False,
#     #'special-members': True
# }

# def skip_uncommented_functions(app, what, name, obj, skip, options):
#     if what == 'function' and obj.__doc__ is None:
#         print('Skip skip skip skip:', what)
#         return True
#     return skip
#
# # Configure autodoc-skip-member to use the custom function
# def setup(app):
#     app.connect('autodoc-skip-member', skip_uncommented_functions)

