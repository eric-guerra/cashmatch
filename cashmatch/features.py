"""Candidate generation and features for matching bank receipts to open invoices.

A "candidate" is one hypothesis: this payment settles this invoice, or this *set* of invoices from one
customer (customers often pay several invoices in one wire, sometimes short by a bank fee).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from itertools import combinations

from rapidfuzz import fuzz

FEATURES = [
    "amount_diff_ratio", "exact_amount", "within_fee", "invoice_ref_hits", "invoice_ref_all",
    "name_similarity", "customer_ref_hit", "days_from_due", "n_invoices", "prior_payer_match",
]

_STOP = re.compile(r"\b(inc|llc|ltd|corp|co|company|the|gmbh|sa)\b\.?", re.I)


@dataclass(frozen=True)
class Invoice:
    id: str
    number: str
    customer_id: str
    customer_name: str
    amount_cents: int
    due: date
    currency: str = "USD"


@dataclass(frozen=True)
class Payment:
    id: str
    amount_cents: int
    received: date
    payer_name: str
    memo: str
    currency: str = "USD"


@dataclass
class Candidate:
    payment: Payment
    invoices: tuple[Invoice, ...]
    features: dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(i.amount_cents for i in self.invoices)

    @property
    def key(self) -> tuple[str, ...]:
        return tuple(sorted(i.id for i in self.invoices))


def normalize_name(s: str) -> str:
    s = _STOP.sub(" ", s.lower())
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s).split())


def invoice_refs(memo: str) -> set[str]:
    """Pull invoice-number-ish tokens from remittance text: 'INV 10234', 'inv#10235', '10236/10237'."""
    m = memo.upper().replace("INV#", "INV ").replace("INVOICE", "INV")
    return set(re.findall(r"\d{4,8}", m))


def _number_digits(number: str) -> str:
    return re.sub(r"\D", "", number)


def generate(payment: Payment, open_invoices: list[Invoice], *, fee_tolerance_cents: int = 5000,
             max_combo: int = 4, max_per_customer: int = 40, prior_payers: dict[str, set[str]] | None = None) -> list[Candidate]:
    """Single-invoice candidates for plausible invoices, plus multi-invoice combos per customer whose sum
    is within tolerance of the payment (bounded subset search)."""
    prior_payers = prior_payers or {}
    refs = invoice_refs(payment.memo)
    pname = normalize_name(payment.payer_name)
    same_ccy = [i for i in open_invoices if i.currency == payment.currency]

    by_customer: dict[str, list[Invoice]] = {}
    for inv in same_ccy:
        by_customer.setdefault(inv.customer_id, []).append(inv)

    cands: dict[tuple[str, ...], Candidate] = {}

    def add(invs: tuple[Invoice, ...]):
        c = Candidate(payment, invs)
        if c.key not in cands:
            cands[c.key] = c

    for inv in same_ccy:
        referenced = _number_digits(inv.number) in refs
        close = abs(inv.amount_cents - payment.amount_cents) <= max(fee_tolerance_cents, payment.amount_cents * 0.02)
        named = fuzz.token_set_ratio(pname, normalize_name(inv.customer_name)) >= 70
        if referenced or close or (named and inv.amount_cents >= payment.amount_cents * 0.5):
            add((inv,))

    for cust, invs in by_customer.items():
        if len(invs) < 2:
            continue
        name_ok = fuzz.token_set_ratio(pname, normalize_name(invs[0].customer_name)) >= 60
        ref_ok = any(_number_digits(i.number) in refs for i in invs)
        if not (name_ok or ref_ok):
            continue
        # Remittance often lists exactly the invoices paid: try that set first.
        referenced = tuple(i for i in invs if _number_digits(i.number) in refs)
        if len(referenced) >= 2:
            add(referenced)
        # Bounded subset-sum: invoices due closest to the receipt date; deeper combos only on small pools.
        pool = sorted(invs, key=lambda i: abs((payment.received - i.due).days))[:max_per_customer]
        for k in range(2, min(max_combo, len(pool)) + 1):
            if k == 4 and len(pool) > 20:
                break
            for combo in combinations(pool, k):
                short = sum(i.amount_cents for i in combo) - payment.amount_cents
                if -100 <= short <= fee_tolerance_cents:  # exact (± rounding), or short-paid by a bank fee
                    add(tuple(combo))

    out = list(cands.values())
    for c in out:
        c.features = featurize(c, refs, pname, prior_payers)
    return out


def featurize(c: Candidate, refs: set[str], pname: str, prior_payers: dict[str, set[str]]) -> dict[str, float]:
    p = c.payment
    diff = p.amount_cents - c.total
    nums = [_number_digits(i.number) for i in c.invoices]
    hits = sum(n in refs for n in nums)
    cust = c.invoices[0]
    return {
        "amount_diff_ratio": abs(diff) / max(1, c.total),
        "exact_amount": float(diff == 0),
        "within_fee": float(0 < diff <= 5000 or -5000 <= diff < 0),
        "invoice_ref_hits": hits / len(nums),
        "invoice_ref_all": float(hits == len(nums) and hits > 0),
        "name_similarity": fuzz.token_set_ratio(pname, normalize_name(cust.customer_name)) / 100,
        "customer_ref_hit": float(cust.customer_id.lower() in p.memo.lower()),
        "days_from_due": max(-60, min(120, (p.received - min(i.due for i in c.invoices)).days)) / 120,
        "n_invoices": float(len(c.invoices)),
        "prior_payer_match": float(pname in prior_payers.get(cust.customer_id, set())),
    }


def vector(c: Candidate) -> list[float]:
    return [c.features[f] for f in FEATURES]
