"""Shared data models for test-guard."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Verdict(Enum):
    """Test adequacy verdict for a file or overall report.

    Priority (worst-wins): FAIL > WARNING > PASS > SKIP.
    SKIP means the layer was unable to produce a verdict (e.g., no coverage data).
    """
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"


@dataclass(frozen=True)
class FileVerdict:
    """Verdict for a single file."""

    file: str
    verdict: Verdict
    reason: str
    layer: str
    matched_test: str | None = None
    # All test files matched to this source (a class/module can have several:
    # unit + integration + e2e). matched_test stays the canonical one for
    # reporting; matched_tests carries the full set for batching.
    matched_tests: tuple[str, ...] = ()


@dataclass
class LayerResult:
    """Result from one layer of analysis.

    Attributes:
        layer: Layer identifier (e.g., "layer1", "layer2", "layer3").
        verdict: Verdict for this layer (PASS/FAIL/WARNING/SKIP).
        details: Human-readable summary of the layer's findings.
        file_verdicts: Per-file verdicts produced by this layer.
        short_circuit: If True, this layer's verdict is final (Layer 2 in AI-disabled mode).
        coverage_details: Per-file coverage percentages (populated by Layer 1).
        unmeasurable_files: Changed source files that ARE in the coverage report
            but have no executable changed lines, so there is nothing to cover
            (populated by Layer 1, consumed by Layer 3's Gate 2).
    """

    layer: str
    verdict: Verdict
    details: str
    file_verdicts: list[FileVerdict]
    short_circuit: bool = False
    coverage_details: dict[str, float] | None = None
    unmeasurable_files: set[str] = field(default_factory=lambda: set[str]())


@dataclass
class Report:
    """Final report aggregating all layers.

    The overall_verdict property implements a priority-based aggregation:
    - If Layer 3 ran and returned non-SKIP, its verdict is authoritative (overrides L1+L2).
    - Otherwise, worst-wins across all layers: FAIL > WARNING > PASS > SKIP.
    """

    # Use default_factory to create a new list per instance (avoid mutable default).
    layers: list[LayerResult] = field(default_factory=lambda: list[LayerResult]())
    summary: str | None = None

    @property
    def overall_verdict(self) -> Verdict:
        """Compute the final verdict by priority: Layer 3 authority, then worst-wins.

        Returns:
            Verdict: FAIL > WARNING > PASS > SKIP (worst-wins if no Layer 3 authority).
        """
        verdicts = [lr.verdict for lr in self.layers]
        if verdicts and all(verdict == Verdict.SKIP for verdict in verdicts):
            return Verdict.SKIP

        # Layer 3 is authoritative: if it ran and returned non-SKIP, use its verdict.
        layer3_results = [lr for lr in self.layers if lr.layer == "layer3"]
        if layer3_results:
            l3_verdict = layer3_results[-1].verdict
            if l3_verdict != Verdict.SKIP:
                return l3_verdict

        # Fallback: worst-wins across all layers (Layer 3 skipped or didn't run).
        non_skip = [v for v in verdicts if v != Verdict.SKIP]
        if not non_skip:
            return Verdict.SKIP
        if Verdict.FAIL in non_skip:
            return Verdict.FAIL
        if Verdict.WARNING in non_skip:
            return Verdict.WARNING
        return Verdict.PASS
