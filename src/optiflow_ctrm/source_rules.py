"""Small grammars for explicit commercial syntax, with original-text coordinates.

These rules operate on a scoped source, not on arbitrary model guesses. They do
not convert quantities, compute FX or infer terms from typical product specs.
"""
from __future__ import annotations

from datetime import datetime
import re

from optiflow_ctrm.provenance import add_source_span, physical_spans, locate_wrapped_evidence

WORDS = dict(zip(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety".split(),
    list(range(20)) + list(range(20, 100, 10)),
))
NUMBER = r"(?:\d+(?:\.\d+)?|" + "|".join(WORDS) + r")"
UNIT = r"(?:MT|BBL|metric\s+tonnes?|barrels?)"
BL = r"B\s*/?\s*L"
OFFSET = rf"(?:[+-]|plus|minus)\s*{NUMBER}"


def numeric(text):
    text = text.casefold().strip()
    return WORDS[text] if text in WORDS else float(text.replace(",", ""))


def signed_offset(text):
    match = re.fullmatch(rf"([+-]|plus|minus)\s*({NUMBER})", text, re.I)
    return numeric(match[2]) * (-1 if match[1].lower() in {"-", "minus"} else 1)


def parse_bl(source):
    pattern = rf"\b{BL}\s*(?P<a>{OFFSET}|date)\s*(?:/|\.\.|to\b|thru\b|through\b)\s*(?:{BL}\s*)?(?P<b>{OFFSET})\b"
    matches = list(re.finditer(pattern, source, re.I))
    if not matches:
        match = re.search(rf"\bon\s+the\s+{BL}\s+date\s+only\b", source, re.I)
        if not match:
            return None
        a = b = 0
    else:
        # Multiple differing periods require amendment interpretation by the model.
        values = {(0 if m['a'].lower() == 'date' else signed_offset(m['a']), signed_offset(m['b'])) for m in matches}
        if len(values) != 1:
            return None
        a, b = values.pop()
        if int(a) != a or int(b) != b:
            return None
        a, b = int(a), int(b)
        match = matches[0]
    return ({"kind": "bl_relative", "expression": f"BL{a:+d}..BL{b:+d}",
             "from_offset_days": a, "to_offset_days": b}, match.span())


def evidence(spans, field, source, match, group=0):
    for span in physical_spans(source, *match.span(group)):
        add_source_span(spans, field, span)


def explicit_rules(deal, source, spans):
    quality = re.search(r'\b([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+)\s+quality\b', source)
    if quality:
        deal['quality_spec'] = quality.group()
        evidence(spans, '/deal/quality_spec', source, quality)
    pricing = deal.get("pricing")
    if isinstance(pricing, dict):
        parse_prices(pricing, source, spans)
        statistic = pricing.get("statistic")
        if statistic and not re.search(r"\b(?:mean|high|low|average)\b", source, re.I):
            pricing["statistic"] = None
    parse_fx(deal, source, spans)
    parse_delivery(deal, source, spans)
    parse_inspection(deal, source, spans)
    quantity = deal.get("quantity")
    if isinstance(quantity, dict):
        if re.search(r"\boption\s+(?:not\s+(?:recorded|agreed|specified)|TBC)\b", source, re.I):
            quantity["tolerance_option"] = None
        multiple = re.search(rf"\b{NUMBER}\s*(?:or(?:\s+maybe)?\s+|-)\s*{NUMBER}\s+(?:stems|cargoes)\b[^\r\n]*?\b(?:each|apiece)\b", source, re.I)
        if multiple:
            quantity["value"] = None
            if isinstance(deal.get("notes"), list) and multiple.group() not in deal["notes"]:
                deal["notes"].append(multiple.group())
            evidence(spans, "/deal/notes", source, multiple)
    clean_notes(deal, source)


def parse_prices(pricing, source, spans):
    # A signed amount and its explicit currency/unit form a closed grammar.
    pattern = rf"(?P<sign>minus|less|plus|discount(?:\s+of)?|premium|[+-])\s*(?:USD|\$)\s*(?P<n>\d[\d,]*(?:\.\d+)?)\s*(?:/|per\s+)(?P<u>{UNIT})\b"
    candidates = []
    for m in re.finditer(pattern, source, re.I):
        negative = m['sign'].lower().startswith(('minus', 'less', 'discount', '-'))
        unit = 'USD/BBL' if m['u'].lower().startswith(('bbl', 'barrel')) else 'USD/MT'
        candidates.append(((-1 if negative else 1) * float(m['n'].replace(',', '')), unit, m))
    spoken = rf'\b(?P<sign>premium|discount|plus|minus)\s+(?P<d>{NUMBER})\s+dollars?\s+(?P<c>{NUMBER})\s+cents?\s+per\s+(?P<u>{UNIT})\b'
    for m in re.finditer(spoken, source, re.I):
        cents = numeric(m['c'])
        if not 0 <= cents < 100:
            continue
        negative = m['sign'].lower() in {'discount', 'minus'}
        amount = (numeric(m['d']) + cents / 100) * (-1 if negative else 1)
        unit = 'USD/BBL' if m['u'].lower().startswith(('bbl', 'barrel')) else 'USD/MT'
        candidates.append((amount, unit, m))
    if candidates:
        # Do not impose textual ordering across forwarded messages. A single value
        # is unambiguous; multiple values remain the model's amendment decision.
        values = {(amount, unit) for amount, unit, _ in candidates}
        current = pricing.get('differential')
        if isinstance(current, dict):
            if len(values) == 1:
                amount, unit = values.pop()
                current.update(value=amount, unit=unit)
            for amount, unit, m in candidates:
                if current.get('value') == amount and current.get('unit') == unit:
                    evidence(spans, '/deal/pricing/differential', source, m)


def parse_fx(deal, source, spans):
    pattern = r"\b(?:fixed(?:\s+agreed)?(?:\s+rate(?:\s+of)?)?)\s+(?P<n>\d+\.\d+)\s+(?P<a>USD|EUR)\s*(?:/|per\s+|for\s+each\s+)(?P<b>USD|EUR)\b"
    matches = list(re.finditer(pattern, source, re.I))
    if len(matches) != 1:
        return
    m = matches[0]
    deal['fx'] = {'rate': float(m['n']), 'quote': f"{m['a'].upper()}/{m['b'].upper()}", 'basis': 'fixed_agreed'}
    evidence(spans, '/deal/fx', source, m)


def parse_delivery(deal, source, spans):
    term = deal.get('delivery_term')
    if isinstance(term, dict) and isinstance(term.get('as_written'), str):
        wording = term['as_written']
        code = re.match(r"^(EXW|FAS|FOB|CFR|CIF|DAP|DDP)\b", wording, re.I)
        if not code:
            term['incoterm'] = None
        # Exclude quotation names when locating an agreed delivery term.
        matches = [m for m in re.finditer(re.escape(wording), source, re.I)
                   if not re.search(r'\b(?:Platts|Argus)\b', source[source.rfind('\n', 0, m.start())+1:m.start()], re.I)]
        if len(matches) == 1:
            evidence(spans, '/deal/delivery_term', source, matches[0])
    # Numeric day/month ranges: only interpret unambiguous day-first dates.
    m = re.search(r"\b(?:DATES|laycan|window)\s*[:—-]?\s*(\d{1,2}/\d{1,2}/(?:\d{4}|\d{2}))\s*[-–]\s*(\d{1,2}/\d{1,2}/(?:\d{4}|\d{2}))\b", source, re.I)
    if m and any(int(s.split('/')[0]) > 12 for s in (m[1], m[2])):
        try:
            dates = [datetime.strptime(s, '%d/%m/%Y' if len(s.split('/')[-1]) == 4 else '%d/%m/%y').date().isoformat() for s in (m[1], m[2])]
        except ValueError:
            return
        deal['delivery_window'] = {'from': dates[0], 'to': dates[1]}
        evidence(spans, '/deal/delivery_window', source, m)


def parse_inspection(deal, source, spans):
    inspection = deal.get('inspection')
    if not isinstance(inspection, dict):
        return
    # Identify company/location grammar; never derive appointment from cost sharing.
    m = re.search(r"(?:Inspection:\s*|(?<=\. ))(?P<company>[A-Z][\w-]*)\s+at\s+(?P<location>load(?:\s+port)?|discharge)\b", source)
    if m:
        inspection['location'] = m['location']
        evidence(spans, '/deal/inspection/location', source, m, 'location')
        # Explicit replacement wins even when the source is newest-first.
        replacement = re.search(r"(?:use\s+|inspector\s+is\s+)([A-Z][\w-]*)\s+(?:instead\s+of|not)\s+" + re.escape(m['company']) + r"\b", source)
        if replacement:
            inspection['company'] = replacement[1]
            evidence(spans, '/deal/inspection/company', source, replacement, 1)
            evidence(spans, '/deal/inspection/company', source, m, 'company')
        else:
            inspection['company'] = m['company']
            evidence(spans, '/deal/inspection/company', source, m, 'company')
    roles = list(re.finditer(r"\b(?:appointed(?:\s+and\s+paid)?\s+by\s+(buyer|seller|jointly)|(buyer|seller)\s+appoints)\b", source, re.I))
    pronoun_appointment = re.search(r'\bappointed\s+by\s+us\b', source, re.I)
    if roles and not pronoun_appointment:
        values = {(x[1] or x[2]).lower() for x in roles}
        if len(values) == 1:
            role = values.pop()
            inspection['appointed_by'] = 'joint' if role == 'jointly' else role
            for role_match in roles:
                evidence(spans, '/deal/inspection/appointed_by', source, role_match)
    elif not roles and not pronoun_appointment:
        inspection['appointed_by'] = None
    equal = re.search(r'\b(?:costs?\s+(?:equally\s+shared|50/50)|split\s+inspection\s+cost\s+50/50)\b', source, re.I)
    if equal:
        inspection['cost_split'] = '50/50'
        evidence(spans, '/deal/inspection/cost_split', source, equal)
    else:
        payer = re.search(r'\b(buyer|seller)\s+appoints\s+and\s+pays\b', source, re.I)
        if payer:
            inspection['cost_split'] = payer[1].lower()
            evidence(spans, '/deal/inspection/cost_split', source, payer)


def duplicate_tolerance(note, quantity):
    """Recognize a closed tolerance clause, not arbitrary text sharing evidence."""
    if not isinstance(quantity, dict):
        return False
    lowered = note.casefold()
    percentage = re.search(rf'\b({NUMBER})\s*(?:%|percent\b|per\s+cent\b|pct\b)', lowered)
    parties = re.findall(r'\b(buyer|seller)\b', lowered)
    if len(set(parties)) != 1:
        return False
    if parties[0] != quantity.get('tolerance_option'):
        return False
    if percentage:
        if numeric(percentage[1]) != quantity.get('tolerance_pct'):
            return False
        remainder = lowered[:percentage.start()] + lowered[percentage.end():]
    else:
        if not re.search(r'\b(?:choice|option)\b', lowered):
            return False
        remainder = lowered
    allowed = {'more', 'less', 'plus', 'minus', 'at', 'in', 'the', 's', 'buyer', 'seller',
               'choice', 'option', 'on', 'volume', 'quantity', 'with', 'tolerance'}
    return set(re.findall(r'\w+', remainder)) <= allowed


def clean_notes(deal, source):
    notes = deal.get('notes')
    if not isinstance(notes, list):
        return
    result = []
    for note in notes:
        if note not in source and not locate_wrapped_evidence(note, source):
            continue
        # Administrative acknowledgements are not independent commercial terms.
        if re.fullmatch(r"(?:All remaining terms stand\.\s*)?Confirmed(?:\s+(?:tonight,?|on)\s+[^.;]+)?[.!]?", note, re.I):
            continue
        if duplicate_tolerance(note, deal.get('quantity')):
            continue
        result.append(note)
    deal['notes'] = result
