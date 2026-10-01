"""llmanifold: one front door for many LLM endpoints.

Routes requests by model alias to pools of endpoints, translates between the
OpenAI and Anthropic APIs, falls back by rule, and shows it all on a dashboard.
"""

__version__ = "0.3.1"
