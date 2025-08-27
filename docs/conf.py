"""
Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import os
import sys
from pathlib import Path

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

# Add the project root directory to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

# -- Project information -----------------------------------------------------

project = 'MEA-Flow'
copyright = '2024, MEA-Flow Team'
author = 'MEA-Flow Team'

# The full version, including alpha/beta/rc tags
release = '0.1.0'
version = '0.1.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'nbsphinx',
    'myst_parser',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    '**.ipynb_checkpoints',
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Extension configuration -------------------------------------------------

# autodoc configuration
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
    'show-inheritance': True,
}

# autosummary configuration
autosummary_generate = True
autosummary_mock_imports = ['pyspike', 'umap']

# napoleon configuration for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True

# intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}

# nbsphinx configuration
nbsphinx_execute = 'never'  # Don't execute notebooks during build
nbsphinx_allow_errors = True
nbsphinx_kernel_name = 'python3'

# MyST parser configuration for Markdown files
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_admonition",
    "html_image",
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom CSS files
html_css_files = [
    'custom.css',
]

# The name for this set of Sphinx documents.
html_title = f'{project} v{release}'

# A shorter title for the navigation bar.
html_short_title = project

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# html_logo = '_static/logo.png'

# The name of an image file (within the static path) to use as favicon of the
# docs.
# html_favicon = '_static/favicon.ico'

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
html_sidebars = {
    '**': [
        'relations.html',  # needs 'show_related': True theme option to display
        'searchbox.html',
    ]
}

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    'papersize': 'letterpaper',
    
    # The font size ('10pt', '11pt' or '12pt').
    'pointsize': '10pt',
    
    # Additional stuff for the LaTeX preamble.
    'preamble': '',
    
    # Latex figure (float) alignment
    'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    ('index', 'MEA-Flow.tex', 'MEA-Flow Documentation',
     'MEA-Flow Team', 'manual'),
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ('index', 'mea-flow', 'MEA-Flow Documentation',
     [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    ('index', 'MEA-Flow', 'MEA-Flow Documentation',
     author, 'MEA-Flow', 'Neural Population Dynamics Analysis for Multi-Electrode Arrays.',
     'Miscellaneous'),
]

# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']

# -- Mock imports for autodoc ------------------------------------------------

autodoc_mock_imports = [
    'pyspike',
    'umap',
    'umap-learn',
]

# -- Setup function ----------------------------------------------------------

def setup(app):
    """Sphinx setup function."""
    app.add_css_file('custom.css')