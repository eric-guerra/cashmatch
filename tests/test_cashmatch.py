from datetime import date

import pytest
from fastapi.testclient import TestClient

from cashmatch import features as F
from cashmatch import synth
from cashmatch.api import create_app
from cashmatch.model import Matcher, evaluate


@pytest.fixture(scope="module")
def trained():
    inv_train, pay_train, truth_train = synth.generate(seed=1, n_payments=500)
    m = Matcher().fit(pay_train, inv_train, truth_train)
    inv_test, pay_test, truth_test = synth.generate(seed=99, n_payments=250)
    return m, inv_test, pay_test, truth_test


def test_memo_reference_parsing():
    assert F.invoice_refs("INV 10234 inv#10235, 10236/10237") == {"10234", "10235", "10236", "10237"}
    assert F.invoice_refs("PAYMENT THANK YOU") == set()
    assert F.normalize_name("ACME Packaging, Inc.") == "acme packaging"


def test_candidates_include_multi_invoice_combo_short_by_fee():
    invs = [F.Invoice("a", "INV-1001", "C1", "Acme Packaging Inc", 120_000, date(2023, 3, 1)),
            F.Invoice("b", "INV-1002", "C1", "Acme Packaging Inc", 80_000, date(2023, 3, 5)),
            F.Invoice("c", "INV-1003", "C2", "Globex Corporation", 200_000, date(2023, 3, 1))]
    p = F.Payment("p", 200_000 - 2_500, date(2023, 3, 20), "ACME PACKAGING", "")
    keys = {c.key for c in F.generate(p, invs)}
    assert ("a", "b") in keys
    combo = next(c for c in F.generate(p, invs) if c.key == ("a", "b"))
    assert combo.features["within_fee"] == 1.0 and combo.features["n_invoices"] == 2


def test_other_currency_is_never_a_candidate():
    invs = [F.Invoice("a", "INV-1", "C1", "Acme", 1000, date(2023, 1, 1), currency="CAD")]
    assert F.generate(F.Payment("p", 1000, date(2023, 1, 1), "ACME", "INV 1"), invs) == []


def test_model_quality_on_held_out_data(trained):
    m, inv, pay, truth = trained
    r = evaluate(m, pay, inv, truth)
    # The bar the AR team agreed: auto-posted matches must be essentially always right.
    assert r["auto_precision"] >= 0.98, r
    assert r["auto_rate"] >= 0.5, r
    assert r["top1_accuracy"] >= 0.85, r


def test_feature_importance_is_sensible(trained):
    m = trained[0]
    top3 = list(m.importance())[:3]
    assert {"invoice_ref_all", "exact_amount", "amount_diff_ratio", "invoice_ref_hits", "within_fee"} & set(top3)


def test_api_batch_never_double_claims_an_invoice(trained):
    m, inv, pay, truth = trained
    client = TestClient(create_app(m))
    i = inv[0]
    body = {
        "open_invoices": [{"id": i.id, "number": i.number, "customer_id": i.customer_id, "customer_name": i.customer_name,
                           "amount_cents": i.amount_cents, "due": i.due.isoformat()}],
        "payments": [{"id": f"dup{k}", "amount_cents": i.amount_cents, "received": i.due.isoformat(),
                      "payer_name": i.customer_name.upper(), "memo": f"INV {i.number[4:]}"} for k in range(2)],
    }
    r = client.post("/match", json=body).json()
    autos = [x for x in r["results"] if x["outcome"] == "auto"]
    assert len(autos) <= 1
    assert r["summary"]["auto"] + r["summary"]["suggest"] + r["summary"]["exception"] == 2

    assert client.post("/feedback", json={"payment_id": "dup0", "accepted_invoice_ids": [i.id], "clerk": "kim"}).status_code == 201
    assert client.post("/feedback", json={"payment_id": "dup0", "accepted_invoice_ids": [], "clerk": "kim"}).status_code == 422
    assert client.get("/model").json()["features"] == F.FEATURES


def test_unknown_payer_with_no_reference_is_an_exception(trained):
    m, inv, _, _ = trained
    d = m.decide(F.Payment("x", 123_457, date(2023, 6, 1), "ZZZ UNKNOWN", ""), inv)
    assert d.outcome in ("exception", "suggest")
    assert d.outcome != "auto"
