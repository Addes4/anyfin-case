# /// script
# requires-python = ">=3.10"
# dependencies = ["anthropic==2.14.0"]
# ///
"""Edge-case checks that need no API key:  uv run test_classify.py"""
import classify as C

_n = 0


def row(cust, acct, name, typ, day, amount, text):
    global _n
    _n += 1
    return {"id": f"x{_n:04d}", "customer": cust, "bank": "Test", "account": acct, "account_name": name,
            "account_type": typ, "date": f"2026-05-{day:02d}", "amount_sek": f"{amount:.2f}", "text": text}


def rules_only(rows):
    """Labels from pairing and rules; None where the model would be asked."""
    names, pairs = C.own_names(rows), C.pair_transfers(rows)
    sav = C.savings_accounts(rows)
    return {r["id"]: pairs.get(r["id"]) or C.rule(r, sav, names) for r in rows}


def test_unrelated_payments_are_not_paired():
    a = row("c", "c-main", "Personkonto", "checking", 3, -399, "Autogiro TELIA SVERIGE AB")
    b = row("c", "c-hush", "Hushållskonto", "checking", 4, 399, "Swish inbetalning ANNA B")
    lab = rules_only([a, b])
    assert lab[a["id"]] is None, lab            # left to the model (Telia is housing)
    assert lab[b["id"]] == "transfer.person", lab


def test_real_savings_pair():
    a = row("c", "c-main", "Personkonto", "checking", 25, -3000, "Överföring Sparkonto 3300 41 22871")
    b = row("c", "c-spar", "Sparkonto", "savings", 25, 3000, "Överföring Från 3300 41 18204")
    lab = rules_only([a, b])
    assert lab[a["id"]] == lab[b["id"]] == "transfer.savings", lab


def test_card_bill_pair():
    a = row("c", "c-main", "Privatkonto", "checking", 20, -4246.02, "SEB KORT BG")
    b = row("c", "c-card", "Kreditkort", "credit_card", 24, 4246.02, "INBETALNING")
    lab = rules_only([a, b])
    assert lab[a["id"]] == "credit.loan" and lab[b["id"]] == "transfer.own", lab


def test_sparse_checking_accounts_are_not_savings():
    a = row("c", "c-a", "Vardag", "checking", 10, -500, "Överföring 1234-5678")
    b = row("c", "c-b", "Extra", "checking", 10, 500, "Överföring 8765-4321")
    assert C.savings_accounts([a, b]) == set()
    lab = rules_only([a, b])
    assert lab[a["id"]] == lab[b["id"]] == "transfer.own", lab


def test_ambiguous_pair_is_skipped():
    a = row("c", "c-a", "Personkonto", "checking", 10, -500, "Överföring 1234")
    b = row("c", "c-b", "Konto B", "checking", 9, 500, "Överföring 5678")
    d = row("c", "c-d", "Konto D", "checking", 11, 500, "Överföring 9999")
    assert C.pair_transfers([a, b, d]) == {}


def test_unowned_transfer_goes_to_model():
    a = row("c", "c-main", "Personkonto", "checking", 10, -2000, "Överföring 8327-9,912 345 678-1")
    assert rules_only([a])[a["id"]] is None


def test_buffer_named_checking_account_is_savings():
    a = row("c", "c-main", "Lönekonto", "checking", 6, 2500, "ÖVERFÖRING 8327-9,912 345 670-3")
    b = row("c", "c-buf", "Buffert", "checking", 6, -2500, "ÖVERFÖRING 8327-9,912 345 678-1")
    lab = rules_only([a, b])
    assert lab[a["id"]] == lab[b["id"]] == "transfer.savings", lab


def test_seb_and_foreign_suffixes_normalise():
    assert C.norm("COOP TRIANGELN /26-10-22") == "COOP TRIANGELN"
    assert C.norm("Kortköp 260912 PINGO DOCE LISBOA EUR 21.98") == "PINGO DOCE LISBOA"
    assert C.norm("REMA OSLO NOK 120,00 /26-07-08") == "REMA OSLO"


def test_swish_names_never_reach_the_model():
    r = row("c", "c-main", "Personkonto", "checking", 1, -455, "Swish betalning OSCAR L mat")
    it = C.item_of(r, set())
    assert "OSCAR" not in it["text"] and it["text"] == "SWISH PRIVATE PERSON MAT", it


def test_api_failure_is_reported_not_silent():
    class Boom(Exception):
        pass

    def fail(*a, **k):
        raise Boom("simulated outage")

    orig, C.classify_batch, C.RETRY_SLEEP = C.classify_batch, fail, 0
    orig_cache = C.CACHE
    C.CACHE = "/dev/null/never"  # nothing cached, nothing written
    C.FAILED_ROWS.clear()
    try:
        rows = [row("c", "c-main", "Personkonto", "checking", 2, -129, "Kortköp 260501 SOME NEW SHOP")]
        train = [dict(row("t", "t-main", "Personkonto", "checking", 2, -50, "Kortköp 260501 ICA NARA"),
                      category="spend.groceries")]
        try:
            C.save_cache = lambda c: None
            pred = C.predict(rows, train)
        finally:
            pass
        assert pred[rows[0]["id"]] == "unknown"
        assert C.FAILED_ROWS == [rows[0]["id"]], C.FAILED_ROWS
    finally:
        C.classify_batch, C.CACHE = orig, orig_cache


if __name__ == "__main__":
    import os
    os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"{len(tests)} passed")
