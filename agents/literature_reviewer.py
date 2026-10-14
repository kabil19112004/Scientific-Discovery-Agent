"""Literature Reviewer Agent.

Retrieves papers from the vector database and builds an *extractive*, citation-backed review.
Every finding, method, limitation, disagreement and gap is a verbatim sentence from a retrieved
abstract and carries the paper it came from, so nothing is invented. An optional ``llm`` callable
can add a narrative summary, but it never replaces the extracted evidence.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Callable, Literal

from pydantic import BaseModel, Field

from vector_store.vector_database import SearchResult

logger = logging.getLogger(__name__)

STOPWORDS = frozenset(
    "this that with from which their these those have been were also into than then such using used "
    "paper study studies show shows results result however while within between about their there "
    "other more most over under when where what whose both each many some can may our we are for and "
    "the but not has had its they them will would could should based method methods approach".split())

_CUES = {
    "result": r"show|demonstrat|achiev|improv|outperform|found|find|reveal|increas|reduc|enabl|yield",
    "method": r"we (?:propose|present|introduce|use|develop|train|apply|employ)|based on|approach|framework|"
              r"architecture|algorithm|pipeline|trained|fine-tun|simulation|randomi[sz]ed|cohort",
    "limitation": r"however|limitation|limited|challeng|fail|struggl|drawback|degrad|cannot|computationally|"
                  r"suffer|bias|expensive|restricted",
    "disagreement": r"in contrast|contradict|inconsistent|conflict|whereas|unlike|mixed|controvers|debate|"
                    r"contrary|disagree|discrepan",
    "gap": r"future (?:work|research)|remains? (?:unclear|unknown|unexplored|open)|not well understood|"
           r"poorly understood|open question|little is known|lack of|lacks|unexplored|understudied|"
           r"further (?:research|investigation)|yet to|has not been|have not been",
}
CUE_PATTERNS = {k: re.compile(rf"\b(?:{v})", re.I) for k, v in _CUES.items()}
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

NOTES = [
    "Items are verbatim sentences from retrieved abstracts, selected by keyword cues; full texts were not read.",
    "'supported' means at least two retrieved abstracts report lexically similar results; it is not a "
    "verification of correctness or a meta-analysis.",
    "Disagreement items are contrast cues and may be ordinary comparisons rather than true conflicts.",
]


# ---- schemas ------------------------------------------------------------
class Citation(BaseModel):
    paper_id: str
    title: str
    authors: list[str]
    published: str | None
    url: str
    relevance_score: float


class EvidenceItem(BaseModel):
    text: str
    paper_id: str
    paper_title: str


class Finding(BaseModel):
    statement: str
    evidence_level: Literal["supported", "tentative"]
    paper_ids: list[str]


class LiteratureReview(BaseModel):
    question: str
    status: Literal["ok", "insufficient_evidence"]
    message: str
    citations: list[Citation] = Field(default_factory=list)
    summary: str = ""
    key_findings: list[Finding] = Field(default_factory=list)
    common_methods: list[EvidenceItem] = Field(default_factory=list)
    limitations: list[EvidenceItem] = Field(default_factory=list)
    disagreements: list[EvidenceItem] = Field(default_factory=list)
    research_gaps: list[EvidenceItem] = Field(default_factory=list)
    recurring_themes: list[str] = Field(default_factory=list)
    llm_narrative: str | None = None
    notes: list[str] = Field(default_factory=list)
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_markdown(self) -> str:
        lines = [f"# Literature review: {self.question}", "", f"**Status:** {self.status}. {self.message}", ""]
        if self.summary:
            lines += [self.summary, ""]
        if self.llm_narrative:
            lines += ["## Narrative (LLM-assisted, grounded in the abstracts above)", self.llm_narrative, ""]
        if self.key_findings:
            lines.append("## Key findings")
            lines += [f"- [{f.evidence_level}] {f.statement} ({', '.join(f.paper_ids)})" for f in self.key_findings]
            lines.append("")
        for title, items in (("Common methods", self.common_methods), ("Limitations", self.limitations),
                             ("Contrasts / possible disagreements", self.disagreements),
                             ("Potential research gaps", self.research_gaps)):
            if items:
                lines.append(f"## {title}")
                lines += [f"- {i.text} ({i.paper_id})" for i in items]
                lines.append("")
        if self.citations:
            lines.append("## References")
            for c in self.citations:
                who = (c.authors[0] + (" et al." if len(c.authors) > 1 else "")) if c.authors else "Unknown authors"
                lines.append(f"- {who} ({c.published or 'n.d.'}). {c.title}. {c.url} [{c.paper_id}]")
            lines.append("")
        if self.notes:
            lines.append("## Caveats")
            lines += [f"- {n}" for n in self.notes]
        return "\n".join(lines)


# ---- helpers --------------------------------------------------------------
def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text or "") if s.strip()]


def content_tokens(sentence: str) -> frozenset[str]:
    return frozenset(w for w in re.findall(r"[a-z]{4,}", sentence.lower()) if w not in STOPWORDS)


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


class LiteratureReviewerAgent:
    def __init__(self, vector_db, llm: Callable[[str], str] | None = None, min_papers: int = 3,
                 min_score: float = 0.25, max_items: int = 5, support_threshold: float = 0.2) -> None:
        self.vector_db = vector_db
        self.llm = llm
        self.min_papers = min_papers
        self.min_score = min_score
        self.max_items = max_items
        self.support_threshold = support_threshold

    def review(self, question: str, k: int = 8) -> LiteratureReview:
        question = (question or "").strip()
        if not question:
            raise ValueError("A research question or topic is required")
        results: list[SearchResult] = self.vector_db.search(question, k=k)
        relevant = [r for r in results if r.score >= self.min_score]
        citations = [Citation(paper_id=r.paper.paper_id, title=r.paper.title, authors=r.paper.authors,
                              published=r.paper.published, url=r.paper.url, relevance_score=r.score)
                     for r in relevant]
        if len(relevant) < self.min_papers:
            return LiteratureReview(
                question=question, status="insufficient_evidence", citations=citations,
                message=(f"Only {len(relevant)} sufficiently relevant paper(s) found "
                         f"(need {self.min_papers}; {len(results)} retrieved, relevance threshold "
                         f"{self.min_score}). No findings are reported. Ingest more papers on this topic."),
                notes=NOTES)

        pools: dict[str, list[tuple[SearchResult, str]]] = {k_: [] for k_ in CUE_PATTERNS}
        for r in relevant:
            for sent in split_sentences(r.paper.abstract):
                if len(sent.split()) < 6:
                    continue
                for cat, pat in CUE_PATTERNS.items():
                    if pat.search(sent):
                        pools[cat].append((r, sent))

        findings = self._findings(pools["result"])
        review = LiteratureReview(
            question=question, status="ok", citations=citations, key_findings=findings,
            common_methods=self._items(pools["method"]), limitations=self._items(pools["limitation"]),
            disagreements=self._items(pools["disagreement"]), research_gaps=self._items(pools["gap"]),
            recurring_themes=self._themes(relevant), notes=list(NOTES), message="Review generated from retrieved abstracts.")
        n_sup = sum(f.evidence_level == "supported" for f in findings)
        review.summary = (f"Reviewed {len(relevant)} relevant papers. Extracted {len(findings)} finding(s) "
                          f"({n_sup} supported by 2+ abstracts, {len(findings) - n_sup} tentative), "
                          f"{len(review.research_gaps)} potential gap(s) and {len(review.limitations)} limitation(s).")
        if not findings:
            review.notes.append("No result-style sentences were detected; treat this review as incomplete.")
        if self.llm:
            review.llm_narrative = self._narrate(question, relevant, review)
        return review

    def _items(self, pool: list[tuple[SearchResult, str]]) -> list[EvidenceItem]:
        items, per_paper, seen = [], {}, []
        for r, sent in pool:
            pid = r.paper.paper_id
            toks = content_tokens(sent)
            if per_paper.get(pid, 0) >= 2 or any(jaccard(toks, t) >= 0.5 for t in seen):
                continue
            per_paper[pid] = per_paper.get(pid, 0) + 1
            seen.append(toks)
            items.append(EvidenceItem(text=sent, paper_id=pid, paper_title=r.paper.title))
            if len(items) >= self.max_items:
                break
        return items

    def _findings(self, pool: list[tuple[SearchResult, str]]) -> list[Finding]:
        toks = [content_tokens(s) for _, s in pool]
        scored = []
        for i, (r, sent) in enumerate(pool):
            supporters = [r.paper.paper_id]
            for j, (r2, _) in enumerate(pool):
                pid2 = r2.paper.paper_id
                if pid2 not in supporters and jaccard(toks[i], toks[j]) >= self.support_threshold:
                    supporters.append(pid2)
            scored.append((-len(supporters), i, sent, supporters))
        scored.sort(key=lambda x: (x[0], x[1]))
        findings, chosen = [], []
        for neg, i, sent, supporters in scored:
            if any(jaccard(toks[i], t) >= 0.5 for t in chosen):
                continue
            chosen.append(toks[i])
            findings.append(Finding(statement=sent, paper_ids=supporters,
                                    evidence_level="supported" if len(supporters) >= 2 else "tentative"))
            if len(findings) >= self.max_items + 1:
                break
        return findings

    @staticmethod
    def _themes(relevant: list[SearchResult], top: int = 10) -> list[str]:
        df: dict[str, int] = {}
        for r in relevant:
            for w in content_tokens(f"{r.paper.title}. {r.paper.abstract}"):
                df[w] = df.get(w, 0) + 1
        ranked = sorted(((c, w) for w, c in df.items() if c >= 2), key=lambda x: (-x[0], x[1]))
        return [w for _, w in ranked[:top]]

    def _narrate(self, question: str, relevant: list[SearchResult], review: LiteratureReview) -> str | None:
        evidence = "\n".join(f"[{r.paper.paper_id}] {r.paper.title}: {r.paper.abstract}" for r in relevant)
        prompt = (f"Question: {question}\n\nUsing ONLY the abstracts below, write a short neutral synthesis. "
                  f"Cite paper ids in brackets. Do not mention any paper, number or claim not present below. "
                  f"If evidence is thin, say so.\n\n{evidence}")
        try:
            return self.llm(prompt).strip() or None
        except Exception as exc:  # LLM is optional; never fail the review because of it
            logger.warning("LLM narrative failed: %s", exc)
            review.notes.append(f"LLM narrative unavailable: {exc}")
            return None
