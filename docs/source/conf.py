import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "MBTR"
copyright = "2020–2026, Lorenzo Nespoli"
author = "Lorenzo Nespoli"

from mbtr import __version__  # noqa: E402

release = __version__
version = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
autosummary_generate = True
autodoc_typehints = "description"
napoleon_numpy_docstring = True
templates_path = ["_templates"]
exclude_patterns = []
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = f"MBTR {release}"

latex_engine = "xelatex"
latex_show_urls = "footnote"
master_doc = "index"
