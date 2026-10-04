"""Regenerate the synthetic corpus and eval set deterministically.

All content is SYNTHETIC: a fictional retailer, authored for this repository.

    python build_corpus.py            # writes corpus/docs/*.txt and corpus/eval.jsonl
    python build_corpus.py --out DIR  # write somewhere else

Same code, same bytes: every random choice comes from a string-seeded
random.Random, never from hash order or the clock.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import textwrap
from pathlib import Path
from typing import Sequence

import gate

SEED = "synthetic-rag-groundedness-v1"
PER_CLASS = 15
GROUNDED_SINGLE, GROUNDED_SAME_DOC, GROUNDED_TWO_DOCS = 52, 26, 26
GROUNDED_RESPELLED, GROUNDED_LISTS, GROUNDED_INVISIBLE = 20, 15, 10

# (doc id, title, four fact sentences). Fictional company: Larkspur and Pine.
# Rules for authors: no double quotes, no inner sentence periods except decimals,
# at least one number, money amount, percent or date per fact, none in the first
# twelve characters.
DOCS: list[tuple[str, str, list[str]]] = [
    ("POL-01", "Return Window", [
        "Unused items may be returned within 30 days of delivery for a full refund.",
        "Items marked final sale cannot be returned after 48 hours from checkout.",
        "Customers pay a restocking fee of 10% on opened electronics.",
        "Returns started after March 31, 2026 follow the extended spring window.",
    ]),
    ("POL-02", "Refund Timing", [
        "Approved refunds reach the original payment method within 7 business days.",
        "Refunds under $15.00 are issued as store credit instead of a card refund.",
        "Customers receive a refund confirmation email within 24 hours of approval.",
        "Refunds for orders paid in cash at a store are capped at $500 per visit.",
    ]),
    ("POL-03", "Standard Shipping", [
        "Standard shipping costs $5.99 and arrives in 3 to 5 business days.",
        "Orders over $75 ship free within the continental region.",
        "Orders placed after 2 PM local time ship on the next business day.",
        "Standard parcels weigh no more than 40 pounds per box.",
    ]),
    ("POL-04", "Express Shipping", [
        "Express shipping costs $14.99 and arrives within 2 business days.",
        "Overnight delivery is available for $29.99 on orders placed before noon.",
        "Express orders over $200 receive a shipping discount of 50%.",
        "Express shipping is paused from December 24, 2026 through December 26, 2026.",
    ]),
    ("POL-05", "International Shipping", [
        "International orders ship in 10 to 14 business days after payment.",
        "Customers are responsible for import duties above $800 per parcel.",
        "International shipping starts at $24.00 for parcels under 5 pounds.",
        "The international carrier cutoff for holiday delivery is November 20, 2026.",
    ]),
    ("POL-06", "Price Adjustment", [
        "Customers may request a price adjustment within 14 days of purchase.",
        "Adjustments are limited to differences greater than $5.00 per item.",
        "Price adjustments are not available on clearance items marked 70% off.",
        "Adjustment requests are reviewed by the pricing team within 3 business days.",
    ]),
    ("POL-07", "Gift Cards", [
        "Gift cards are issued in amounts from $10 to $500.",
        "Gift cards never expire but lose a dormancy fee of $2.00 after 24 months.",
        "Lost gift cards can be replaced only with proof of purchase within 90 days.",
        "Gift cards cannot be redeemed for cash except where balances fall below $5.",
    ]),
    ("POL-08", "Loyalty Program", [
        "Members earn 5% back in points on every eligible order.",
        "Points expire after 18 months of account inactivity.",
        "A member reaches Gold status after spending $1,200 in a calendar year.",
        "Double points weekends run on April 11, 2026 and April 12, 2026.",
    ]),
    ("POL-09", "Product Warranty", [
        "Every appliance carries a limited warranty of 12 months from delivery.",
        "Warranty claims require a receipt dated within the last 365 days.",
        "The warranty does not cover damage from repairs attempted by owners in the first 90 days.",
        "Replacement units ship within 5 business days of an approved claim.",
    ]),
    ("POL-10", "Extended Warranty", [
        "The extended plan adds 24 months of coverage for a one time fee of $89.",
        "Extended plans must be purchased within 30 days of the original order.",
        "The extended plan includes a deductible of $25 per claim.",
        "Plans purchased before June 1, 2026 renew at a discount of 15%.",
    ]),
    ("POL-11", "Repairs", [
        "Repair requests are acknowledged within 2 business days.",
        "Bench repairs cost a flat fee of $45 plus parts.",
        "Repairs take about 10 business days once the item reaches the depot.",
        "Items left at the depot for more than 60 days are recycled.",
    ]),
    ("POL-12", "Order Cancellation", [
        "Orders can be cancelled at no cost within 2 hours of checkout.",
        "Cancellations after shipment incur a handling fee of $8.50.",
        "Cancelled orders are refunded within 5 business days.",
        "Custom engraved orders cannot be cancelled after 24 hours.",
    ]),
    ("POL-13", "Order Changes", [
        "Shipping addresses can be edited for up to 4 hours or until the order status reads packed.",
        "Address changes after packing carry a fee of $6.00.",
        "Item swaps are allowed within 1 hour of checkout.",
        "Orders edited after June 15, 2026 may lose their promotional price.",
    ]),
    ("POL-14", "Backorders", [
        "Backordered items ship within 21 days of the original order date.",
        "Customers may cancel a backordered item at any time before 30 days elapse.",
        "Backordered items are charged only when they ship, up to a limit of $1,000.",
        "The seasonal garden line returns to stock on May 4, 2026.",
    ]),
    ("POL-15", "Payment Methods", [
        "The store accepts 4 major card brands and 2 digital wallets.",
        "Card payments are authorized for up to $5,000 per order.",
        "Digital wallet orders over $2,500 require an extra verification step.",
        "Declined payments are retried automatically after 48 hours.",
    ]),
    ("POL-16", "Installment Plans", [
        "Orders above $300 can be split into 4 equal installments.",
        "Installment plans carry no interest when paid within 6 weeks.",
        "A late installment adds a fee of $7.00 after a grace period of 5 days.",
        "New installment plans are paused between November 27, 2026 and November 30, 2026.",
    ]),
    ("POL-17", "Late Fees", [
        "Business invoices are due 30 days after the invoice date.",
        "Overdue balances accrue a late fee of 1.5% per month.",
        "Accounts more than 60 days overdue are placed on hold.",
        "The minimum late fee is $10 for any overdue invoice.",
    ]),
    ("POL-18", "Sales Tax", [
        "Sales tax of up to 9% is calculated at checkout using the delivery address.",
        "Tax exempt customers must upload a certificate valid through December 31, 2026.",
        "Exemption certificates are reviewed within 3 business days.",
        "Tax is refunded in full when an order is returned within 30 days.",
    ]),
    ("POL-19", "Account Security", [
        "Passwords must contain at least 12 characters.",
        "Accounts lock for 15 minutes after 5 failed sign in attempts.",
        "Customers are asked to confirm their identity every 90 days.",
        "Security notices are emailed within 1 hour of a suspicious sign in.",
    ]),
    ("POL-20", "Data Retention", [
        "Order records are kept for 7 years to meet bookkeeping rules.",
        "Support chat transcripts are deleted after 180 days.",
        "Marketing preferences are retained until the customer unsubscribes or 36 months pass.",
        "Backups are rotated every 30 days and removed on a schedule ending June 30, 2026.",
    ]),
    ("POL-21", "Privacy Requests", [
        "Customers may request a copy of their data once every 12 months.",
        "Privacy requests are answered within 30 days of identity verification.",
        "Deletion requests remove marketing data within 14 days.",
        "Verified requests received before January 15, 2026 follow the earlier process.",
    ]),
    ("POL-22", "Store Hours", [
        "Stores open at 9 AM and close at 8 PM from Monday to Saturday.",
        "Sunday hours run from 11 AM to 5 PM.",
        "The flagship store stays open until 10 PM on 6 peak weekends each year.",
        "Curbside pickup orders are held for 3 days.",
    ]),
    ("POL-23", "Holiday Schedule", [
        "All stores close on December 25, 2026 and January 1, 2027.",
        "Support is available for 4 hours on Thanksgiving Day.",
        "Holiday shipping deadlines are posted by November 1, 2026.",
        "Stores open 2 hours early on the day after Thanksgiving.",
    ]),
    ("POL-24", "Support Response Times", [
        "Email support replies within 24 hours on business days.",
        "Phone support answers calls within 5 minutes on average.",
        "Chat support is staffed from 8 AM to 10 PM on weekdays.",
        "Priority customers receive a reply within 4 hours.",
    ]),
    ("POL-25", "Escalations", [
        "Unresolved cases escalate to a supervisor after 3 business days.",
        "Supervisors contact the customer within 1 business day of escalation.",
        "Compensation above $100 requires manager approval.",
        "Escalation records are reviewed every quarter, starting July 1, 2026.",
    ]),
    ("POL-26", "Bulk Orders", [
        "Orders of 50 units or more qualify for bulk pricing.",
        "Bulk discounts start at 8% and reach 20% at 500 units.",
        "Bulk orders ship in 15 business days after a signed quote.",
        "Quotes are valid for 30 days from the issue date.",
    ]),
    ("POL-27", "Business Accounts", [
        "Business accounts can set a credit limit of up to $25,000.",
        "Net terms of 45 days require a credit review lasting 5 business days.",
        "Each business account may add up to 10 authorized buyers.",
        "Annual business contracts renew on January 1 unless cancelled by October 31, 2026.",
    ]),
    ("POL-28", "Referral Rewards", [
        "A referred friend receives $15 off a first order of $60 or more.",
        "The referrer earns a $10 credit after the first order ships.",
        "Referral credits expire after 12 months.",
        "Each member can earn rewards for at most 20 referrals per year.",
    ]),
    ("POL-29", "Damaged Goods", [
        "Damage must be reported within 48 hours of delivery with photos.",
        "Damaged items are replaced or refunded in 3 business days after review.",
        "Claims for items worth over $250 require an inspection.",
        "Packaging is kept for 14 days so carriers can inspect it.",
    ]),
    ("POL-30", "Recycling Program", [
        "Customers can return old appliances for recycling at no cost within 45 days of a new delivery.",
        "Recycling pickup is free for orders above $150 within 30 days of delivery.",
        "Each recycled appliance earns a credit of $20 toward the next order.",
        "The program accepts up to 3 appliances per customer each year.",
    ]),
]

INVENTED = [
    "Customers who mention this policy receive an automatic credit of $40 on any order",
    "Returns are accepted for 120 days when the original packaging has been discarded",
    "Every order ships free for loyalty members regardless of order value or weight",
    "Warranty claims are honored for 60 months at no charge for any appliance",
    "Refunds are issued within 1 hour to any payment method on request",
    "Gift cards can be exchanged for cash at any store up to $1,000",
    "Price adjustments are granted for 90 days on every item including clearance",
    "Express shipping is free on all orders placed during the month of May",
    "Late fees are waived entirely for accounts held for more than 6 months",
    "Support replies to every message within 15 minutes around the clock",
    "Bulk pricing of 35% applies to any order of 10 units or more",
    "Referral rewards are paid as $100 in cash after a friend signs up",
    "Backordered items ship free with a bonus credit of $25 per week of delay",
    "Customers may cancel any order at no cost until the day it is delivered",
    "Damaged items are replaced within 12 hours without photos or a report",
]

SWAPS = [
    (r"\bCustomers\b", "Shoppers"), (r"\bcustomers\b", "shoppers"), (r"\bwithin\b", "inside"),
    (r"\bmay\b", "can"), (r"\bmust\b", "need to"), (r"\bitems\b", "products"),
    (r"\bOrders\b", "Purchases"), (r"\borders\b", "purchases"), (r"\bapproved\b", "accepted"),
    (r"\breceive\b", "get"), (r"\bearn\b", "collect"), (r"\bcosts\b", "is priced at"),
    (r"\barrives\b", "is delivered"), (r"\bsupport\b", "service"), (r"\brequires\b", "needs"),
]
LEAD_INS = ["", "Per our policy: ", "Our policy states: "]

# Numberless claims about things no policy doc says (used by unrelated_citation).
UNRELATED = [
    "Gold members enjoy a private lounge with fresh coffee",
    "The flagship store hosts a weekly jazz evening",
    "Employees receive a company bicycle after their first year",
    "Our headquarters runs entirely on wind power",
    "Visitors can tour the roastery on Saturday mornings",
    "The mobile app supports a dark theme",
    "Every new location plants a tree on opening day",
    "Staff wear green aprons during the holiday season",
    "The founder started the company in a small garage",
    "Gift wrapping uses recycled paper and linen ribbon",
    "The loyalty club hosts an annual picnic by the lake",
    "New locations feature local art on the walls",
    "The warehouse cats keep the packing floor quiet",
    "Seasonal playlists are curated by the managers",
    "Cooking classes run in the downtown kitchen",
]

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
         "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety"}
_HOMOGLYPH = {"o": "\u043e", "e": "\u0435", "a": "\u0430", "i": "\u0456"}


def int_words(n: int) -> str:
    """0..99 in English words ("twenty-five")."""
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens] + (f"-{_ONES[ones]}" if ones else "")


def disguise(word: str) -> str:
    """Swap the first Latin letter that has a Cyrillic twin for it."""
    for i, ch in enumerate(word):
        if ch in _HOMOGLYPH:
            return word[:i] + _HOMOGLYPH[ch] + word[i + 1:]
    raise AssertionError(word)


def rng_for(name: str) -> random.Random:
    return random.Random(f"{SEED}:{name}")


def strip_period(fact: str) -> str:
    return fact[:-1] if fact.endswith(".") else fact


def cite(doc: str, quote: str) -> str:
    return f'[doc:{doc} "{quote}"]'


def claim(lead: str, text: str, doc: str, quote: str) -> str:
    return f"{lead}{text} {cite(doc, quote)}."


def doc_texts() -> dict[str, str]:
    out = {}
    for doc_id, title, facts in DOCS:
        body = textwrap.fill(" ".join(facts), width=70)
        out[doc_id] = f"[SYNTHETIC] {title}\n\n{body}\n"
    return out


def mutate_token(token: str, k: int) -> str:
    """Return a different number of the same shape (money stays money, date stays date)."""
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", token):
        y, m, d = token.split("-")
        return f"{y}-{m}-{(int(d) + k - 1) % 28 + 1:02d}"
    if re.search(r"[A-Za-z]", token):
        return re.sub(r"\b\d{1,2}\b", lambda m: str((int(m.group()) + k - 1) % 28 + 1), token, count=1)

    def bump(m: re.Match[str]) -> str:
        raw = m.group(0).replace(",", "")
        if "." in raw:
            return f"{float(raw) + k:.{len(raw.split('.')[1])}f}"
        value = int(raw) + k
        return f"{value:,}" if "," in m.group(0) else str(value)

    return re.sub(r"\d+(?:,\d{3})*(?:\.\d+)?", bump, token, count=1)


def first_number_span(fact: str) -> re.Match[str]:
    m = gate.NUM_RE.search(fact)
    assert m, fact
    return m


def pick_facts(rng: random.Random, n: int, docs: list[tuple[str, str, list[str]]]) -> list[tuple[str, str]]:
    pool = [(d, f) for d, _, facts in docs for f in facts]
    return rng.sample(pool, n)


def grounded_records(rng: random.Random) -> list[dict]:
    docs = DOCS
    ids = [d for d, _, _ in docs]
    recs: list[dict] = []

    def retrieved_with(rng: random.Random, needed: list[str]) -> list[str]:
        distractors = rng.sample([i for i in ids if i not in needed], 2)
        return sorted(set(needed) | set(distractors))

    def sentence(rng: random.Random, doc: str, fact: str) -> str:
        return claim(rng.choice(LEAD_INS), strip_period(fact), doc, strip_period(fact))

    n = 0
    for _ in range(GROUNDED_SINGLE):
        (doc, fact), = pick_facts(rng, 1, docs)
        n += 1
        recs.append(dict(id=f"G-{n:03d}", split="grounded", **{"class": "grounded"},
                         retrieved=retrieved_with(rng, [doc]), answer=sentence(rng, doc, fact),
                         expect_reason=None, bad_sentence=None))
    for _ in range(GROUNDED_SAME_DOC):
        doc, _, facts = rng.choice(docs)
        a, b = rng.sample(facts, 2)
        n += 1
        recs.append(dict(id=f"G-{n:03d}", split="grounded", **{"class": "grounded"},
                         retrieved=retrieved_with(rng, [doc]),
                         answer=f"{sentence(rng, doc, a)} {sentence(rng, doc, b)}",
                         expect_reason=None, bad_sentence=None))
    for _ in range(GROUNDED_TWO_DOCS):
        all_facts = [(d, f) for d, _, fs in docs for f in fs]
        (d1, f1), (d2, f2) = rng.sample(all_facts, 2)
        while d1 == d2:
            (d1, f1), (d2, f2) = rng.sample(all_facts, 2)
        n += 1
        recs.append(dict(id=f"G-{n:03d}", split="grounded", **{"class": "grounded"},
                         retrieved=retrieved_with(rng, [d1, d2]),
                         answer=f"{sentence(rng, d1, f1)} {sentence(rng, d2, f2)}",
                         expect_reason=None, bad_sentence=None))

    # Variants that a naive hardening would wrongly block: kept in the grounded split so
    # false positives on them are reported, not hidden.
    pool = [(d, f) for d, _, fs in docs for f in fs]

    def variant(name: str, doc_list: list[str], answer: str) -> None:
        nonlocal n
        n += 1
        recs.append(dict(id=f"G-{n:03d}", split="grounded", **{"class": "grounded"}, variant=name,
                         retrieved=retrieved_with(rng, doc_list), answer=answer,
                         expect_reason=None, bad_sentence=None))

    def respelled(fact: str) -> str | None:
        m = first_number_span(fact)
        tok = m.group(0)
        if re.fullmatch(r"\d{1,2}", tok):
            new = int_words(int(tok))
        elif re.fullmatch(r"\$\d{1,3}", tok):
            new = f"{tok[1:]} dollars"
        elif re.fullmatch(r"\d{1,2}%", tok):
            new = f"{tok[:-1]} percent"
        else:
            return None
        return fact[: m.start()] + new + fact[m.end():]

    made = 0
    for doc, fact in rng.sample(pool, len(pool)):
        text = respelled(fact)
        if text is None:
            continue
        variant("number_respelled", [doc], claim(rng.choice(LEAD_INS), strip_period(text), doc, strip_period(fact)))
        made += 1
        if made == GROUNDED_RESPELLED:
            break
    assert made == GROUNDED_RESPELLED

    for i in range(GROUNDED_LISTS):
        (d1, f1), (d2, f2) = rng.sample(pool, 2)
        a = claim("", strip_period(f1), d1, strip_period(f1))
        b = claim("", strip_period(f2), d2, strip_period(f2))
        kind = i % 3
        answer = (f"- {a}\n- {b}", f"1. {a}\n2. {b}", f"{a}\n{b}")[kind]
        variant(("bulleted_list", "numbered_list", "line_per_claim")[kind], [d1, d2], answer)

    for doc, fact in rng.sample(pool, GROUNDED_INVISIBLE):
        quote = strip_period(fact).replace(" ", " \u200b", 1)
        variant("invisible_char_in_quote", [doc], claim(rng.choice(LEAD_INS), strip_period(fact), doc, quote))
    return recs


def ungrounded_records() -> list[dict]:
    docs = DOCS
    ids = [d for d, _, _ in docs]
    facts_of = {d: f for d, _, f in docs}
    text_of = {d: " ".join(f) for d, _, f in docs}
    recs: list[dict] = []

    def good_prefix(rng: random.Random, avoid: str) -> tuple[str, str]:
        """A grounded sentence from another doc, used as a leading sentence in compound answers."""
        d = rng.choice([i for i in ids if i != avoid])
        f = rng.choice(facts_of[d])
        return d, claim(rng.choice(LEAD_INS), strip_period(f), d, strip_period(f))

    def emit(cls: str, rng: random.Random, i: int, doc: str, bad: str, retrieved: list[str]) -> None:
        prefix_doc = None
        answer, bad_idx = bad, 0
        if i % 2 == 1:  # compound answer: a grounded sentence first, the defect second
            prefix_doc, good = good_prefix(rng, doc)
            answer, bad_idx = f"{good} {bad}", 1
            retrieved = retrieved + [prefix_doc]
        recs.append(dict(id=f"{cls}-{i + 1:02d}", split="ungrounded", **{"class": cls},
                         retrieved=sorted(set(retrieved)), answer=answer,
                         expect_reason=gate.CLASS_REASON[cls], bad_sentence=bad_idx))

    def with_distractor(rng: random.Random, doc: str) -> list[str]:
        return [doc, rng.choice([i for i in ids if i != doc])]

    # fabricated_quote: an invented quote for a real, retrieved doc
    rng = rng_for("fabricated_quote")
    for i, text in enumerate(INVENTED[:PER_CLASS]):
        doc = rng.choice(ids)
        emit("fabricated_quote", rng, i, doc,
             claim(rng.choice(LEAD_INS), text, doc, text), with_distractor(rng, doc))

    # paraphrase_drift: a real fact with words swapped, quoted as if verbatim
    rng = rng_for("paraphrase_drift")
    pool = [(d, f) for d, _, fs in docs for f in fs]
    rng.shuffle(pool)
    made = 0
    for doc, fact in pool:
        drifted = strip_period(fact)
        for pattern, repl in SWAPS:
            drifted = re.sub(pattern, repl, drifted)
        if drifted == strip_period(fact) or drifted in text_of[doc]:
            continue
        emit("paraphrase_drift", rng, made, doc,
             claim(rng.choice(LEAD_INS), drifted, doc, drifted), with_distractor(rng, doc))
        made += 1
        if made == PER_CLASS:
            break
    assert made == PER_CLASS

    # wrong_number: real quote, claim changes a number to one that appears nowhere in the doc
    rng = rng_for("wrong_number")
    pool = [(d, f) for d, _, fs in docs for f in fs]
    for i, (doc, fact) in enumerate(rng.sample(pool, PER_CLASS)):
        m = first_number_span(fact)
        k = rng.randint(2, 9)
        while gate.number_in_text(gate._norm_number(mutate_token(m.group(0), k)), text_of[doc]):
            k += 1
        text = strip_period(fact[: m.start()] + mutate_token(m.group(0), k) + fact[m.end():])
        emit("wrong_number", rng, i, doc,
             claim(rng.choice(LEAD_INS), text, doc, strip_period(fact)), with_distractor(rng, doc))

    # missing_citation: a claim with no citation at all
    rng = rng_for("missing_citation")
    for i, (doc, fact) in enumerate(rng.sample(pool, PER_CLASS)):
        emit("missing_citation", rng, i, doc,
             f"{rng.choice(LEAD_INS)}{strip_period(fact)}.", with_distractor(rng, doc))

    # unknown_doc: real quote attributed to a doc ID that does not exist
    rng = rng_for("unknown_doc")
    for i, (doc, fact) in enumerate(rng.sample(pool, PER_CLASS)):
        ghost = f"POL-{rng.randint(31, 99)}"
        emit("unknown_doc", rng, i, doc,
             claim(rng.choice(LEAD_INS), strip_period(fact), ghost, strip_period(fact)),
             with_distractor(rng, doc))

    # out_of_retrieval: real doc, real quote, but the doc was not retrieved for this answer
    rng = rng_for("out_of_retrieval")
    for i, (doc, fact) in enumerate(rng.sample(pool, PER_CLASS)):
        others = rng.sample([x for x in ids if x != doc], 2)
        emit("out_of_retrieval", rng, i, doc,
             claim(rng.choice(LEAD_INS), strip_period(fact), doc, strip_period(fact)), others)

    # number_outside_quote: the number is in the doc, but the quote stops before it
    rng = rng_for("number_outside_quote")
    candidates = [(d, f) for d, f in pool if len(f[: first_number_span(f).start()].strip()) >= 12]
    for i, (doc, fact) in enumerate(rng.sample(candidates, PER_CLASS)):
        quote = fact[: first_number_span(fact).start()].strip()
        emit("number_outside_quote", rng, i, doc,
             claim(rng.choice(LEAD_INS), strip_period(fact), doc, quote), with_distractor(rng, doc))

    # number_word_evasion: the claim spells a different number as a word; the quote has the real one
    rng = rng_for("number_word_evasion")
    plain = [(d, f) for d, f in pool if re.fullmatch(r"\d{1,2}", first_number_span(f).group(0))]
    for i, (doc, fact) in enumerate(rng.sample(plain, PER_CLASS)):
        m = first_number_span(fact)
        k = rng.randint(2, 9)
        while (int(m.group(0)) + k > 99
               or gate.number_in_text(f"num:{int(m.group(0)) + k}", text_of[doc])):
            k += 1
        text = strip_period(fact[: m.start()] + int_words(int(m.group(0)) + k) + fact[m.end():])
        emit("number_word_evasion", rng, i, doc,
             claim(rng.choice(LEAD_INS), text, doc, strip_period(fact)), with_distractor(rng, doc))

    # number_kind_mismatch: the digits match but the unit or sign does not ($ vs euro, % vs $, +5 vs -5)
    rng = rng_for("number_kind_mismatch")
    kinds = [(d, f) for d, f in pool if re.fullmatch(r"\$\d[\d,]*(?:\.\d+)?|\d+%", first_number_span(f).group(0))]
    for i, (doc, fact) in enumerate(rng.sample(kinds, PER_CLASS)):
        m = first_number_span(fact)
        tok = m.group(0)
        if tok.endswith("%"):
            new = f"${tok[:-1]}"
        elif i % 2 == 0:
            new = "\u20ac" + tok[1:]
        else:
            new = "-" + tok
        text = strip_period(fact[: m.start()] + new + fact[m.end():])
        emit("number_kind_mismatch", rng, i, doc,
             claim(rng.choice(LEAD_INS), text, doc, strip_period(fact)), with_distractor(rng, doc))

    # short_quote_laundering: a two-word quote that is technically in the doc
    rng = rng_for("short_quote_laundering")
    for i, (doc, fact) in enumerate(rng.sample(pool, PER_CLASS)):
        quote = " ".join(strip_period(fact).split()[:2])
        emit("short_quote_laundering", rng, i, doc,
             claim(rng.choice(LEAD_INS), strip_period(fact), doc, quote), with_distractor(rng, doc))

    # quote_spans_sentences: one quote stitched from two neighboring sentences of the same doc
    rng = rng_for("quote_spans_sentences")
    stitched = [(d, fs[k], fs[k + 1]) for d, _, fs in docs for k in range(len(fs) - 1)]
    for i, (doc, a, b) in enumerate(rng.sample(stitched, PER_CLASS)):
        emit("quote_spans_sentences", rng, i, doc,
             claim(rng.choice(LEAD_INS), strip_period(a), doc, f"{a} {b}"), with_distractor(rng, doc))

    # subword_quote: the quote starts in the middle of a word, so it is a raw substring of the doc
    rng = rng_for("subword_quote")
    for i, (doc, fact) in enumerate(rng.sample(pool, PER_CLASS)):
        emit("subword_quote", rng, i, doc,
             claim(rng.choice(LEAD_INS), strip_period(fact), doc, strip_period(fact)[1:]),
             with_distractor(rng, doc))

    # unrelated_citation: a real quote attached to a claim that has nothing to do with it
    rng = rng_for("unrelated_citation")
    for i, text in enumerate(UNRELATED[:PER_CLASS]):
        doc, fact = rng.choice(pool)
        emit("unrelated_citation", rng, i, doc,
             claim("", text, doc, strip_period(fact)), with_distractor(rng, doc))

    # hidden_uncited_sentence: an uncited claim made to look like part of a cited one
    rng = rng_for("hidden_uncited_sentence")
    joiners = [("- {u}\n- {g}"), ("1. {u}\n2. {g}"), ("{u}\n{g}"), ("{u}; {g}"), ("{u}.{g}")]
    for i, text in enumerate(INVENTED[:PER_CLASS]):
        doc, fact = rng.choice(pool)
        good = claim("", strip_period(fact), doc, strip_period(fact))
        recs.append(dict(id=f"hidden_uncited_sentence-{i + 1:02d}", split="ungrounded",
                         **{"class": "hidden_uncited_sentence"}, retrieved=sorted(with_distractor(rng, doc)),
                         answer=joiners[i % len(joiners)].format(u=text, g=good),
                         expect_reason=gate.CLASS_REASON["hidden_uncited_sentence"], bad_sentence=0))

    # citation_only: citations with no claim text at all
    rng = rng_for("citation_only")
    for i, (doc, fact) in enumerate(rng.sample(pool, PER_CLASS)):
        one = cite(doc, strip_period(fact))
        answer = (one, f"{one}.", f"{one} {one}")[i % 3]
        recs.append(dict(id=f"citation_only-{i + 1:02d}", split="ungrounded", **{"class": "citation_only"},
                         retrieved=sorted(with_distractor(rng, doc)), answer=answer,
                         expect_reason=gate.CLASS_REASON["citation_only"], bad_sentence=0))

    # mixed_script_number: the number is a word with one Cyrillic letter, so it is not read as a number
    rng = rng_for("mixed_script_number")
    for i, (doc, fact) in enumerate(rng.sample(plain, PER_CLASS)):
        m = first_number_span(fact)
        text = strip_period(fact[: m.start()] + disguise(int_words(int(m.group(0)))) + fact[m.end():])
        emit("mixed_script_number", rng, i, doc,
             claim(rng.choice(LEAD_INS), text, doc, strip_period(fact)), with_distractor(rng, doc))
    return recs


# Accurate rewordings of real doc sentences: the claim text differs from the cited quote, the
# meaning does not. Reported as its own split so the gate's false positives on paraphrase are
# visible and not hidden behind copy-paste answers. known_blocked rows are paraphrases the gate
# still blocks (synonyms it cannot match); they are listed in the README Limitations.
PARAPHRASES = [
    ("hyphen_percent", "POL-01", "Customers pay a restocking fee of 10-percent on opened electronics",
     "Customers pay a restocking fee of 10% on opened electronics", False),
    ("month_abbreviation_ordinal", "POL-05", "The international carrier cutoff for holiday delivery is Nov. 20th, 2026",
     "The international carrier cutoff for holiday delivery is November 20, 2026", False),
    ("amount_without_symbol", "POL-03", "Standard shipping costs 5.99 and arrives in 3 to 5 business days",
     "Standard shipping costs $5.99 and arrives in 3 to 5 business days", False),
    ("lowercase_first_letter_quote", "POL-03", "Standard shipping costs $5.99 and arrives in 3 to 5 business days",
     "standard shipping costs $5.99 and arrives in 3 to 5 business days", False),
    ("clock_abbreviation", "POL-03", "Orders placed after 2 p.m. local time ship on the next business day",
     "Orders placed after 2 PM local time ship on the next business day", False),
    ("number_word_dropped_adjective", "POL-02", "Refunds reach the original payment method within seven business days",
     "Approved refunds reach the original payment method within 7 business days", False),
    ("verb_swap_number_word", "POL-04", "Express shipping is $14.99 and arrives in two business days",
     "Express shipping costs $14.99 and arrives within 2 business days", False),
    ("date_reformat", "POL-08", "Double points weekends fall on April 11 and April 12, 2026",
     "Double points weekends run on April 11, 2026 and April 12, 2026", False),
    ("range_connector", "POL-07", "Gift cards come in amounts from $10 up to $500",
     "Gift cards are issued in amounts from $10 to $500", False),
    ("amount_trailing_zeros", "POL-07", "A dormancy fee of $2 applies to gift cards after 24 months",
     "Gift cards never expire but lose a dormancy fee of $2.00 after 24 months", False),
    ("plural_noun", "POL-02", "Refund confirmation emails go out within 24 hours of approval",
     "Customers receive a refund confirmation email within 24 hours of approval", False),
    ("one_new_word", "POL-07", "Lost gift cards can be replaced if you show proof of purchase within 90 days",
     "Lost gift cards can be replaced only with proof of purchase within 90 days", False),
    ("preposition_swap", "POL-03", "Orders over $75 ship free to the continental region",
     "Orders over $75 ship free within the continental region", False),
    ("synonyms", "POL-08", "Members get 5% of every purchase back as loyalty points",
     "Members earn 5% back in points on every eligible order", True),
    ("synonym_above", "POL-06", "Price adjustments are limited to differences above $5 per item",
     "Adjustments are limited to differences greater than $5.00 per item", True),
]


def build_paraphrase() -> list[dict]:
    return [dict(id=f"P-{i:03d}", split="grounded", **{"class": "paraphrase"}, variant=name, retrieved=[doc],
                 answer=claim("", text, doc, quote), known_blocked=known, expect_reason=None,
                 bad_sentence=None, synthetic=True)
            for i, (name, doc, text, quote, known) in enumerate(PARAPHRASES, 1)]


def build_eval() -> list[dict]:
    recs = grounded_records(rng_for("grounded")) + ungrounded_records()
    for r in recs:
        r["synthetic"] = True
    return recs


def write_all(out_dir: str | Path) -> None:
    out = Path(out_dir)
    (out / "docs").mkdir(parents=True, exist_ok=True)
    for doc_id, text in doc_texts().items():
        (out / "docs" / f"{doc_id}.txt").write_text(text, encoding="utf-8", newline="\n")
    lines = [json.dumps(r, ensure_ascii=True, sort_keys=True) for r in build_eval()]
    (out / "eval.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    lines = [json.dumps(r, ensure_ascii=True, sort_keys=True) for r in build_paraphrase()]
    (out / "paraphrase.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Regenerate the synthetic corpus and eval set deterministically.")
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "corpus"),
                   help="output directory (default: ./corpus)")
    args = p.parse_args(argv)
    write_all(args.out)
    recs = build_eval()
    print(f"wrote {len(DOCS)} docs and {len(recs)} eval records to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
