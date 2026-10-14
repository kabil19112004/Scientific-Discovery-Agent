"""Hypothesis Generator Agent.

Turns a LiteratureReview into *proposed* testable hypotheses. Default mode is a transparent,
rule-based template seeded by the review's research gaps / limitations / contrasts (always cited).
With an optional ``llm`` callable, hypotheses are requested as JSON, validated, and any that cite
papers outside the review are discarded. Output is never presented as a proven discovery.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Callable, Literal

from pydantic import BaseModel, Field, ValidationError

from agents.literature_reviewer import EvidenceItem, LiteratureReview

logger = logging.getLogger(__name__)

DISCLAIMER = ("These are PROPOSED IDEAS for investigation, not proven discoveries. Novelty has not been "
              "verified against the full literature; each hypothesis needs expert review before experiments.")


class Hypothesis(BaseModel):
    hypothesis_id: str
    statement: str
    motivation: str
    supporting_evidence: list[EvidenceItem]
    independent_variables: list[str]
    dependent_variables: list[str]
    control_variables: list[str]
    experiment_design: str
    evaluation_metrics: list[str]
    expected_observations: str
    limitations: list[str]
    alternative_explanations: list[str]
    status: Literal["proposed_idea"] = "proposed_idea"


class HypothesisSet(BaseModel):
    question: str
    status: Literal["ok", "insufficient_evidence"]
    message: str
    generation_mode: Literal["rule_based_template", "llm_validated", "none"]
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER


class HypothesisGeneratorAgent:
    def __init__(self, llm: Callable[[str], str] | None = None) -> None:
        self.llm = llm

    def generate(self, review: LiteratureReview, max_hypotheses: int = 3) -> HypothesisSet:
        if max_hypotheses < 1:
            raise ValueError("max_hypotheses must be >= 1")
        if review.status != "ok":
            return HypothesisSet(question=review.question, status="insufficient_evidence", generation_mode="none",
                                 message="The literature review has insufficient evidence; no hypotheses generated.")
        seeds = (review.research_gaps + review.limitations + review.disagreements)[:max_hypotheses]
        if not seeds:
            return HypothesisSet(question=review.question, status="insufficient_evidence", generation_mode="none",
                                 message="The review contains no research gaps, limitations or contrasts to build on.")
        if self.llm:
            hyps = self._from_llm(review, max_hypotheses)
            if hyps:
                return HypothesisSet(question=review.question, status="ok", generation_mode="llm_validated",
                                     hypotheses=hyps, message=f"{len(hyps)} LLM-drafted hypothesis(es) passed validation.")
            logger.warning("LLM output unusable; falling back to rule-based templates")
        hyps = [self._template(i, seed, review) for i, seed in enumerate(seeds, 1)]
        return HypothesisSet(
            question=review.question, status="ok", generation_mode="rule_based_template", hypotheses=hyps,
            message=("Rule-based drafts seeded from cited research gaps/limitations. They are generic and "
                     "need domain-specific refinement of variables and metrics."))

    # ---- rule-based ------------------------------------------------------
    @staticmethod
    def _template(n: int, seed: EvidenceItem, review: LiteratureReview) -> Hypothesis:
        evidence = [seed]
        for f in review.key_findings:
            if f.paper_ids:
                title = next((c.title for c in review.citations if c.paper_id == f.paper_ids[0]), "")
                evidence.append(EvidenceItem(text=f.statement, paper_id=f.paper_ids[0], paper_title=title))
                break
        return Hypothesis(
            hypothesis_id=f"H{n}",
            statement=(f"[Proposed] Explicitly addressing the issue raised in the literature (\"{seed.text}\") "
                       f"will measurably change the primary outcome compared with an unmodified baseline."),
            motivation=(f"'{seed.paper_title}' ({seed.paper_id}) points to this issue; the review found it "
                        f"relevant to the question \"{review.question}\"."),
            supporting_evidence=evidence,
            independent_variables=["Experimental condition: baseline vs. variant that targets the stated issue"],
            dependent_variables=["Primary performance metric used in the cited studies (fix before running)"],
            control_variables=["Dataset and train/test split", "Random seeds", "Hyperparameters and compute budget",
                               "Preprocessing pipeline"],
            experiment_design=("Run baseline and variant on identical data across at least 5 random seeds, "
                               "pre-registering the primary metric and the comparison."),
            evaluation_metrics=["Primary metric from the cited work", "Mean and standard deviation over seeds",
                                "Paired significance test (e.g. Wilcoxon signed-rank) with effect size"],
            expected_observations=("If the hypothesis holds, the variant differs from the baseline by more than "
                                   "seed-to-seed variation; a null or negative result is equally informative."),
            limitations=["Drafted from abstracts only, not full texts", "Keyword-cue extraction can misread context",
                         "Novelty not checked against the wider literature"],
            alternative_explanations=["Gains come from added capacity or tuning rather than the targeted factor",
                                      "Dataset-specific effects that do not generalise", "Random variation"])

    # ---- LLM path ----------------------------------------------------------
    def _from_llm(self, review: LiteratureReview, n: int) -> list[Hypothesis]:
        valid_ids = {c.paper_id for c in review.citations}
        prompt = (
            f"Research question: {review.question}\n\nEvidence (cite only these ids):\n"
            + "\n".join(f"- {i.paper_id}: {i.text}" for i in review.research_gaps + review.limitations
                        + review.disagreements)
            + "\n" + "\n".join(f"- {', '.join(f.paper_ids)}: {f.statement}" for f in review.key_findings)
            + f"\n\nPropose up to {n} testable hypotheses as a JSON list. Each object needs keys: statement, "
              "motivation, supporting_evidence (list of {text, paper_id, paper_title} copied from the evidence), "
              "independent_variables, dependent_variables, control_variables, experiment_design, "
              "evaluation_metrics, expected_observations, limitations, alternative_explanations. "
              "Return JSON only. Do not claim novelty or proven results.")
        try:
            raw = self.llm(prompt)
            data = json.loads(re.search(r"\[.*\]", raw, re.S).group(0))
        except Exception as exc:
            logger.warning("LLM hypothesis call/parse failed: %s", exc)
            return []
        out: list[Hypothesis] = []
        for item in data[:n]:
            try:
                h = Hypothesis(hypothesis_id=f"H{len(out) + 1}", **{k: v for k, v in item.items()
                                                                    if k not in ("hypothesis_id", "status")})
            except (ValidationError, TypeError, AttributeError):
                continue
            if h.supporting_evidence and all(e.paper_id in valid_ids for e in h.supporting_evidence):
                out.append(h)
        return out
