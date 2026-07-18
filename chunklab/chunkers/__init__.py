"""Importing this package runs every chunker module's @register side effect.

Import order sets display order in the UI: the seven strategies appear as
fixed, recursive, document, semantic, llm, agentic, hierarchical.
"""

from . import fixed        # noqa: F401
from . import recursive    # noqa: F401
from . import document     # noqa: F401
from . import semantic     # noqa: F401
from . import llm          # noqa: F401
from . import agentic      # noqa: F401
from . import hierarchical # noqa: F401
