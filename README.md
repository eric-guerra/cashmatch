# CashMatch

Portfolio reference implementation of a 2023 project, built to show **cash application**: automatically matching incoming bank receipts to open invoices, so AR clerks only handle the ambiguous ones.

**Stack:** Python 3.11, XGBoost, rapidfuzz, FastAPI, PostgreSQL (schema in `schema.sql`).

## Why it's hard
Bank receipts rarely look like the invoices they pay:
- The payer name differs from the customer master ("ACH CREDIT VANDELAY" vs "Vandelay Industries"), sometimes with typos.
- Remittance text is often missing or just says "PAYMENT THANK YOU".
- A single wire can pay **several invoices**, sometimes **short by a bank fee**.
- Round amounts ($5,000.00) collide across customers.

## Approach
1. **Candidate generation** (`features.py`). The currency must match, then a candidate is either:
   - a single invoice that is referenced in the memo, close in amount, or from a customer with a similar name; or
   - a **multi-invoice combination** for one customer. The search first tries the exact set the memo lists, then a bounded subset-sum over the invoices due nearest the receipt date, allowing a short pay up to the fee tolerance.
2. **Features:**
   - Amount difference ratio, exact match, within-fee.
   - Share of the candidate's invoices referenced in the memo, and whether all of them are.
   - Fuzzy name similarity (token-set ratio after stripping Inc/LLC/Co).
   - Customer id in the memo, days from due, number of invoices, prior payer alias.
3. **XGBoost** pairwise classifier with **monotone constraints**, so a closer amount or more references can never lower the score. That keeps the model explainable to finance.
4. **Decision policy:**
   - **auto** when the score is ≥ 0.9 *and* beats the runner-up by ≥ 0.3.
   - **suggest** (pre-selected for a clerk) when the score is ≥ 0.3.
   - Otherwise **exception** (unapplied cash).
   - In a batch, the highest-confidence payments claim first, and two payments can never auto-claim the same invoice.

## Results (synthetic held-out set, 250 receipts)
| Metric | Value |
|---|---|
| Top-1 accuracy | **97.6%** |
| Auto-applied | **67%** of receipts |
| Auto precision | **100%** (168/168) |

Errors analysis drove the candidate generator. The first version found only 77% top-1, because multi-invoice payments were never proposed: the subset search had the short-pay sign backwards and only looked at the 12 oldest invoices.

## Run
```bash
pip install -r requirements.txt
pytest -q                          # 7 tests incl. the precision/auto-rate bar on held-out data
uvicorn "cashmatch.serve:app"      # see cashmatch/api.py:create_app (trains on synthetic data at startup for the demo)
```
