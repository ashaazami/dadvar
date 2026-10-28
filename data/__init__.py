"""Dataset layer: public datasets converted to the typed-decisions schema.

Sources declare what they may be used for (benchmarking, training); the builder loads them at a
pinned revision, samples them, and guarantees train and eval rows never overlap.
"""

from .build import DEFAULT_N, DEFAULT_SEED, Split, build  # noqa: F401
from .sources import PROMPT_VERSION, SOURCES  # noqa: F401
from .targets import target_distribution  # noqa: F401
