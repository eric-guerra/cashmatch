-- Postgres schema backing the service.
CREATE TABLE bank_receipts (
    id           TEXT PRIMARY KEY,           -- bank reference (BAI2 / camt.053 entry id)
    amount_cents BIGINT NOT NULL CHECK (amount_cents > 0),
    currency     CHAR(3) NOT NULL,
    received     DATE NOT NULL,
    payer_name   TEXT NOT NULL DEFAULT '',
    memo         TEXT NOT NULL DEFAULT '',
    status       TEXT NOT NULL DEFAULT 'unapplied' CHECK (status IN ('unapplied', 'suggested', 'applied', 'exception'))
);

CREATE TABLE match_decisions (
    receipt_id    TEXT NOT NULL REFERENCES bank_receipts (id),
    model_version TEXT NOT NULL,
    outcome       TEXT NOT NULL,
    invoice_ids   TEXT[] NOT NULL,
    score         REAL NOT NULL,
    margin        REAL NOT NULL,
    decided_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (receipt_id, model_version)
);

-- Clerk corrections become labeled training rows for the weekly retrain.
CREATE TABLE match_feedback (
    receipt_id           TEXT NOT NULL REFERENCES bank_receipts (id),
    accepted_invoice_ids TEXT[] NOT NULL,
    clerk                TEXT NOT NULL,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Prior payer names per customer (a strong feature: "ACH CREDIT VANDELAY" paid C107 before).
CREATE TABLE payer_aliases (
    customer_id TEXT NOT NULL,
    payer_name  TEXT NOT NULL,
    seen        INT  NOT NULL DEFAULT 1,
    PRIMARY KEY (customer_id, payer_name)
);
