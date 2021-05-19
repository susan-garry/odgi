# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../lib/'))


# -- Project information -----------------------------------------------------

project = u'odgi'
copyright = '2021, Erik Garrison. Revision v0.5.1-f75fafa'
author = u'Andrea Guarracino, Simon Heumos, ... , Pjotr Prins, Erik Garrison'

# The short X.Y version
version = 'v0.5.1'
# The full version, including alpha/beta/rc tags
release = 'f75fafa'


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'm2r2']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = ['.rst', '.md']
# source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [u'_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = None


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_theme_path = ["_themes",]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'odgidoc'


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'odgi.tex', u'odgi documentation',
     u'Andrea Guarracino, Simon Heumos, ... , Pjotr Prins, Erik Garrison', 'manual'),
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
# we list the authors seperately
EG = 'Erik Garrison'
AG = 'Andrea Guarracino'
SH = 'Simon Heumos'
man_pages = [
    ('man/odgi', 'odgi', u'dynamic succinct variation graph tool',
     [author], 1),
    ('man/odgi_bin', 'odgi_bin', u'binning of pangenome sequence and path information in the graph',
     [EG, SH], 1),
    ('man/odgi_break', 'odgi_break', u'break cycles in the graph and drop its paths',
     [EG], 1),
    ('man/odgi_build', 'odgi_build', u'construct a dynamic succinct variation graph',
     [EG], 1),
    ('man/odgi_chop', 'odgi_chop', u'divide nodes into smaller pieces',
     [EG, AG], 1),
    ('man/odgi_cover', 'odgi_cover', u'find a path cover of the variation graph',
     [AG], 1),
    ('man/odgi_degree', 'odgi_degree', u'describe the graph in terms of node degree',
     [EG], 1),
    ('man/odgi_depth', 'odgi_depth', u'find the depth of graph as defined by query criteria',
     [AG], 1),
    ('man/odgi_draw', 'odgi_draw', u'variation graph visualizations in 2D',
     [EG], 1),
    ('man/odgi_explode', 'odgi_explode', u'breaks a graph into connected components in their own',
     [AG], 1),
    ('man/odgi_extract', 'odgi_extract', u'extract parts of the graph as defined by query criteria',
     [AG], 1),
    ('man/odgi_flatten', 'odgi_flatten', u'generate linearization of the graph',
     [EG], 1),
    ('man/odgi_groom', 'odgi_groom', u'resolve spurious inverting links',
     [EG, AG], 1),
    ('man/odgi_kmers', 'odgi_kmers', u'show and characterize the kmer space of the graph',
     [EG], 1),
    ('man/odgi_layout', 'odgi_layout', u'use SGD to make 2D layouts of the graph',
     [EG, AG, SH], 1),
    ('man/odgi_matrix', 'odgi_matrix', u'write the graph topology in sparse matrix formats',
     [EG], 1),
    ('man/odgi_normalize', 'odgi_normalize', u'compact unitigs and simplify redundant furcations',
     [EG], 1),
    ('man/odgi_overlap', 'odgi_overlap', u'find the paths touched by the input paths',
     [AG], 1),
    ('man/odgi_panpos', 'odgi_panpos', u'get the pangenome position of a given path and nucleotide',
     [SH], 1),
    ('man/odgi_pathindex', 'odgi_pathindex', u'create a path index for a given path',
     [SH], 1),
    ('man/odgi_paths', 'odgi_paths', u'embedded path interrogation',
     [EG], 1),
    ('man/odgi_position', 'odgi_position', u'position parts of the graph as defined by query criteria',
     [EG], 1),
    ('man/odgi_prune', 'odgi_prune', u'remove complex parts of the graph',
     [EG], 1),
    ('man/odgi_server', 'odgi_server', u'start a HTTP server with a given index file to query a',
     [SH], 1),
    ('man/odgi_sort', 'odgi_sort', u'sort a variation graph',
     [SH, AG, EG], 1),
    ('man/odgi_squeeze', 'odgi_squeeze', u'squeezes multiple graphs into the same file',
     [AG], 1),
    ('man/odgi_stats', 'odgi_stats', u'metrics describing variation graphs',
     [EG, AG], 1),
    ('man/odgi_test', 'odgi_test', u'run odgi unit tests',
     [EG, SH, AG], 1),
    ('man/odgi_unchop', 'odgi_unchop', u'merge unitigs into single nodes',
     [EG, AG], 1),
    ('man/odgi_unitig', 'odgi_unitig', u'output unitigs of the graph',
     [EG], 1),
    ('man/odgi_validate', 'odgi_validate', u'validate the graph (currently, it checks if the paths',
     [AG], 1),
    ('man/odgi_version', 'odgi_version', u'display the version of odgi',
     [SH], 1),
    ('man/odgi_view', 'odgi_view', u'projection of graphs into other formats',
     [EG], 1),
    ('man/odgi_viz', 'odgi_viz', u'variation graph visualizations',
     [EG, AG], 1),
]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'odgi', u'odgi Documentation',
     author, 'odgi', 'One line description of project.',
     'Miscellaneous'),
]


# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']
