"""FastAPI service: score a batch of bank receipts against open AR; record clerk feedback for retraining."""
from __future__ import annotations

from datetime import date

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import features as F
from .model import Matcher


class InvoiceIn(BaseModel):
    id: str
    number: str
    customer_id: str
    customer_name: str
    amount_cents: int = Field(gt=0)
    due: date
    currency: str = "USD"


class PaymentIn(BaseModel):
    id: str
    amount_cents: int = Field(gt=0)
    received: date
    payer_name: str = ""
    memo: str = ""
    currency: str = "USD"


class MatchRequest(BaseModel):
    payments: list[PaymentIn]
    open_invoices: list[InvoiceIn]


class Feedback(BaseModel):
    payment_id: str
    accepted_invoice_ids: list[str]
    clerk: str


def create_app(matcher: Matcher) -> FastAPI:
    app = FastAPI(title="CashMatch", version="1.0")
    feedback: list[Feedback] = []  # Postgres `match_feedback` in production (see schema.sql)

    @app.post("/match")
    def match(req: MatchRequest):
        invoices = [F.Invoice(**i.model_dump()) if hasattr(i, "model_dump") else F.Invoice(**i.dict()) for i in req.open_invoices]
        results, claimed = [], set()
        # Highest-confidence payments first, so two payments can't both auto-claim the same invoice.
        decisions = [matcher.decide(F.Payment(**(p.model_dump() if hasattr(p, "model_dump") else p.dict())), invoices) for p in req.payments]
        for d in sorted(decisions, key=lambda d: -d.score):
            if d.outcome == "auto" and claimed & set(d.invoice_ids):
                d.outcome = "suggest"
            if d.outcome == "auto":
                claimed |= set(d.invoice_ids)
            results.append({"payment_id": d.payment_id, "outcome": d.outcome, "invoice_ids": list(d.invoice_ids),
                            "score": round(d.score, 4), "margin": round(d.margin, 4),
                            "alternatives": [{"invoice_ids": list(k), "score": round(s, 4)} for k, s in d.alternatives]})
        summary = {o: sum(r["outcome"] == o for r in results) for o in ("auto", "suggest", "exception")}
        return {"summary": summary, "results": sorted(results, key=lambda r: r["payment_id"])}

    @app.post("/feedback", status_code=201)
    def record(fb: Feedback):
        if not fb.accepted_invoice_ids:
            raise HTTPException(422, "accepted_invoice_ids must not be empty")
        feedback.append(fb)
        return {"stored": len(feedback)}

    @app.get("/model")
    def model_info():
        return {"features": F.FEATURES, "importance": matcher.importance(),
                "thresholds": {"auto": matcher.auto_threshold, "margin": matcher.auto_margin, "suggest": matcher.suggest_threshold}}

    return app
