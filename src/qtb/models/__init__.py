"""Model adapters.

The harness talks to a model through :class:`qtb.models.base.Model`. Messages use the
Anthropic Messages content-block shape as the common format (text, tool_use and
tool_result blocks as plain dicts), because it is the only provider wired up so far.
"""
