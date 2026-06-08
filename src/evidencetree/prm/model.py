"""PRM scorer.

In Stage 0.1 we use a *frozen* VisualPRM-8B to score whole retrieval
trajectories, to measure how well its scores correlate with outcome
correctness (the go/no-go signal for the whole project).

Two modes:
* ``mock=True``  -> synthetic score with a tunable correlation to the outcome,
  so the pipeline + correlation analysis can be validated locally. The outcome
  is passed as an EVAL-ONLY ``outcome_hint`` (never seen by the real scorer).
* ``mock=False`` -> load VisualPRM-8B from HuggingFace and read a score from the
  model. Score-only (no rationale text generated) per the implementation report.

NOTE: the exact score-token extraction below is a reasonable default that MUST
be validated against the real VisualPRM-8B I/O format on the GPU server. It is
isolated in ``_score_real`` so only that method needs adjusting.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class TrajectoryStep:
    """One retrieval/answer step in a trajectory."""

    action_type: str            # "text_search" | "image_search" | "answer"
    action_input: str           # query text / region / answer text
    observation: str = ""       # retrieved evidence (text) for this step


@dataclass
class Trajectory:
    """A full retrieval trajectory for a single query."""

    question: str
    steps: list[TrajectoryStep] = field(default_factory=list)
    final_answer: str = ""
    image_path: str | None = None

    def as_text(self) -> str:
        """Flatten the trajectory into a single text block (for scoring)."""
        lines = [f"Question: {self.question}"]
        for i, step in enumerate(self.steps, 1):
            lines.append(f"Step {i} [{step.action_type}]: {step.action_input}")
            if step.observation:
                lines.append(f"  Evidence: {step.observation}")
        lines.append(f"Final answer: {self.final_answer}")
        return "\n".join(lines)


class VisualPRMScorer:
    """Frozen VisualPRM-8B trajectory scorer (real) or synthetic scorer (mock)."""

    def __init__(
        self,
        model_name: str = "OpenGVLab/VisualPRM-8B",
        mock: bool = False,
        device: str | None = None,
        mock_correlation: float = 0.2,
        mock_seed: int = 0,
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.mock = mock
        self.device = device
        self.mock_correlation = float(mock_correlation)
        self._rng = random.Random(mock_seed)
        self._model = None
        self._tokenizer = None
        if not mock:
            self._load_real()

    # ------------------------------------------------------------------ #
    def score(self, trajectory: Trajectory, **kwargs: Any) -> float:
        """Return a scalar score in [0, 1] for the trajectory."""
        if self.mock:
            return self._score_mock(trajectory, **kwargs)
        return self._score_real(trajectory)

    def score_batch(
        self, trajectories: Sequence[Trajectory], hints: Sequence[float] | None = None
    ) -> list[float]:
        hints = hints if hints is not None else [None] * len(trajectories)
        return [
            self.score(traj, outcome_hint=h) for traj, h in zip(trajectories, hints)
        ]

    # ------------------------------------------------------------------ #
    # Mock scoring
    # ------------------------------------------------------------------ #
    def _score_mock(self, trajectory: Trajectory, **kwargs: Any) -> float:
        """Blend an (eval-only) outcome hint with noise at a controlled ratio.

        Expected Spearman(score, outcome) ~= mock_correlation. Used to emulate the
        report's hypothesis that a *frozen* VisualPRM correlates weakly with
        outcome (so Stage 0.1 prints a "needs fine-tuning" verdict).
        """
        noise = self._rng.random()
        hint = kwargs.get("outcome_hint")
        if hint is None:
            return noise
        rho = self.mock_correlation
        return max(0.0, min(1.0, rho * float(hint) + (1.0 - rho) * noise))

    # ------------------------------------------------------------------ #
    # Real scoring (VisualPRM-8B)
    # ------------------------------------------------------------------ #
    def _load_real(self) -> None:
        try:
            import torch  # noqa: F401
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:  # pragma: no cover - server only
            raise ImportError(
                "Real VisualPRM scoring needs torch + transformers. "
                "Install with: pip install -r requirements/models.txt"
            ) from e

        self._torch = __import__("torch")
        device = self.device or ("cuda" if self._torch.cuda.is_available() else "cpu")
        self.device = device
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self._model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            trust_remote_code=True,
            device_map="auto" if device == "cuda" else None,
        ).eval()

    def _score_real(self, trajectory: Trajectory) -> float:  # pragma: no cover
        """Score-only forward pass.

        Default heuristic: prompt the model to judge whether the trajectory
        correctly answers the question, then read the probability mass on the
        positive token ("Yes") vs. the negative token ("No") from the next-token
        logits. VALIDATE this against the real VisualPRM-8B template on the server.
        """
        torch = self._torch
        prompt = (
            f"{trajectory.as_text()}\n\n"
            "Is the final answer correct given the evidence? Answer Yes or No.\n"
            "Answer:"
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self._model(**inputs).logits[0, -1]
        yes_id = self._tokenizer(" Yes", add_special_tokens=False).input_ids[-1]
        no_id = self._tokenizer(" No", add_special_tokens=False).input_ids[-1]
        pair = torch.softmax(torch.stack([logits[no_id], logits[yes_id]]), dim=0)
        return float(pair[1].item())
