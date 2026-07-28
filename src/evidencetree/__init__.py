"""EvidenceTree: multimodal RAG with MCTS over a retrieval action space.

The system searches a retrieval action space with MCTS (UCB1 over the PRM's Q)
and scores each retrieval action with a grounded, action-typed PRM.

See the README for the pipeline overview and entry points.
"""

__version__ = "0.1.0"
