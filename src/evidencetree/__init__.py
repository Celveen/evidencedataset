"""EvidenceTree: multimodal RAG with MCTS over a retrieval action space.

The system searches a retrieval action space with MCTS (UCB1 over the PRM's Q)
and scores each retrieval action with a grounded, action-typed PRM.

See ``EvidenceTree_项目实现报告.md`` for the design document and staged plan.
"""

__version__ = "0.1.0"
