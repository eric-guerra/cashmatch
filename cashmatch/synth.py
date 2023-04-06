"""Synthetic AR history with the messiness real bank files have: payer names that differ from the
customer master, typos, missing remittance, fees deducted, several invoices in one payment."""
from __future__ import annotations

import random
from datetime import date, timedelta

from .features import Invoice, Payment

COMPANIES = [
    "Acme Packaging Inc", "Globex Corporation", "Initech LLC", "Umbrella Retail Co", "Stark Fixtures",
    "Wayne Logistics", "Hooli Supply", "Vandelay Industries", "Soylent Foods", "Tyrell Displays",
    "Cyberdyne Systems", "Wonka Confections", "Gringotts Holdings", "Oceanic Freight", "Pied Piper Co",
    "Monarch Paper", "Duff Beverages", "Krusty Foods", "Prestige Worldwide", "Dunder Mifflin",
]


def _typo(rng: random.Random, s: str) -> str:
    if len(s) < 5 or rng.random() > 0.3:
        return s
    i = rng.randrange(1, len(s) - 1)
    return s[:i] + s[i + 1:]


def _payer_name(rng: random.Random, name: str) -> str:
    variants = [name.upper(), name.replace("Inc", "").replace("LLC", "").strip().upper(),
                "ACH CREDIT " + name.split()[0].upper(), name.split()[0].upper() + " " + name.split()[1][:4].upper() + " AP"]
    return _typo(rng, rng.choice(variants))


def generate(seed: int = 7, n_payments: int = 400):
    """Returns (open_invoices, payments, truth) where truth[payment_id] = tuple(sorted invoice ids)."""
    rng = random.Random(seed)
    customers = [(f"C{100 + i}", n) for i, n in enumerate(COMPANIES)]
    invoices: list[Invoice] = []
    start = date(2023, 1, 1)
    for k in range(n_payments * 3):
        cid, cname = rng.choice(customers)
        amt = rng.choice([rng.randint(50_00, 900_00), rng.randint(1_000_00, 25_000_00)])
        if rng.random() < 0.1:
            amt = rng.choice([1_000_00, 2_500_00, 5_000_00])  # round amounts collide across customers
        issued = start + timedelta(days=rng.randint(0, 150))
        invoices.append(Invoice(f"i{k}", f"INV-{10000 + k}", cid, cname, amt, issued + timedelta(days=30)))

    payments, truth, used = [], {}, set()
    by_cust: dict[str, list[Invoice]] = {}
    for inv in invoices:
        by_cust.setdefault(inv.customer_id, []).append(inv)

    for j in range(n_payments):
        cid, cname = rng.choice(customers)
        avail = [i for i in by_cust[cid] if i.id not in used]
        if not avail:
            continue
        k = 1 if rng.random() < 0.75 else rng.randint(2, min(3, len(avail)))
        chosen = rng.sample(avail, min(k, len(avail)))
        used.update(i.id for i in chosen)
        total = sum(i.amount_cents for i in chosen)
        fee = rng.choice([0, 0, 0, 1500, 2500]) if rng.random() < 0.25 else 0
        memo_kind = rng.random()
        if memo_kind < 0.45:
            memo = " ".join(f"INV {i.number[4:]}" for i in chosen)
        elif memo_kind < 0.6:
            memo = f"PMT {cid} " + "/".join(i.number[4:] for i in chosen)
        elif memo_kind < 0.75:
            memo = "PAYMENT THANK YOU"
        else:
            memo = ""
        received = max(i.due for i in chosen) + timedelta(days=rng.randint(-20, 45))
        p = Payment(f"p{j}", total - fee, received, _payer_name(rng, cname), memo)
        payments.append(p)
        truth[p.id] = tuple(sorted(i.id for i in chosen))
    return invoices, payments, truth
