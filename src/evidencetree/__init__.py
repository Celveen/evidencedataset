"""EvidenceTree: multimodal RAG with MCTS over a retrieval action space.

The system searches a retrieval action space with MCTS, scores each retrieval
action with a grounded PRM, and adapts the exploration strength lambda at
inference time with a Thompson-Sampling bandit.

See ``EvidenceTree_实现报告_for_ClaudeCode.md`` for the staged implementation plan.
"""

__version__ = "0.1.0"
