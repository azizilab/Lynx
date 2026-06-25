"""Utility subpackage for LYNX (IO, plotting, trajectory, zonation, etc.).

Several modules here reference ``LOGGER`` via ``from __init__ import LOGGER``.
We expose the same root logger as the project root ``__init__`` so those
bare imports keep resolving whether the directory is treated as a package
or added directly to ``sys.path``.
"""

import logging

LOGGER = logging.getLogger()
