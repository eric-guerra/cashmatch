"""Pairwise match model + decision policy.

The model scores each candidate independently (XGBoost on the features in features.FEATURES). The
policy then turns scores into one of three outcomes per payment:
  auto     — post the match without a human (high score AND a clear margin over the runner-up)
  suggest  — show the top candidates to an AR clerk, pre-selected
  exception — nothing plausible; goes to the unapplied-cash queue
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from xgboost import XGBClassifier

from . import features as F


@dataclass
class Decision:
    payment_id: str
    outcome: str  # auto | suggest | exception
    invoice_ids: tuple[str, ...]
    score: float
    margin: float
    alternatives: list[tuple[tuple[str, ...], float]]


class Matcher:
    def __init__(self, auto_threshold: float = 0.9, auto_margin: float = 0.3, suggest_threshold: float = 0.3):
        self.auto_threshold = auto_threshold
        self.auto_margin = auto_margin
        self.suggest_threshold = suggest_threshold
        self.model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.9,
                                   eval_metric="logloss", n_jobs=1, random_state=0)
        # Business monotonicity: more referenced invoices / closer amount / more similar name never lowers the score.
        self.model.set_params(monotone_constraints="(-1,1,0,1,1,1,1,0,0,1)")

    def fit(self, payments, invoices, truth, prior_payers=None):
        X, y = [], []
        for p in payments:
            for c in F.generate(p, invoices, prior_payers=prior_payers):
                X.append(F.vector(c))
                y.append(int(c.key == truth.get(p.id)))
        self.model.fit(np.array(X), np.array(y))
        return self

    def decide(self, payment, invoices, prior_payers=None) -> Decision:
        cands = F.generate(payment, invoices, prior_payers=prior_payers)
        if not cands:
            return Decision(payment.id, "exception", (), 0.0, 0.0, [])
        scores = self.model.predict_proba(np.array([F.vector(c) for c in cands]))[:, 1]
        ranked = sorted(zip(cands, scores), key=lambda t: -t[1])
        best, s1 = ranked[0]
        s2 = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = float(s1 - s2)
        if s1 >= self.auto_threshold and margin >= self.auto_margin:
            outcome = "auto"
        elif s1 >= self.suggest_threshold:
            outcome = "suggest"
        else:
            outcome = "exception"
        return Decision(payment.id, outcome, best.key, float(s1), margin, [(c.key, float(s)) for c, s in ranked[1:4]])

    def importance(self) -> dict[str, float]:
        return dict(sorted(zip(F.FEATURES, map(float, self.model.feature_importances_)), key=lambda t: -t[1]))


def evaluate(matcher: Matcher, payments, invoices, truth) -> dict:
    stats = {"auto": 0, "auto_correct": 0, "suggest": 0, "suggest_top1": 0, "exception": 0, "top1": 0, "n": 0}
    for p in payments:
        d = matcher.decide(p, invoices)
        stats["n"] += 1
        stats[d.outcome] += 1
        correct = d.invoice_ids == truth[p.id]
        stats["top1"] += correct
        if d.outcome == "auto":
            stats["auto_correct"] += correct
        if d.outcome == "suggest":
            stats["suggest_top1"] += correct
    n = stats["n"]
    return {
        **stats,
        "top1_accuracy": stats["top1"] / n,
        "auto_rate": stats["auto"] / n,
        "auto_precision": stats["auto_correct"] / max(1, stats["auto"]),
    }
