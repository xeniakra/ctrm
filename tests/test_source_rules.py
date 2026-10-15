"""Independent syntax/safety regressions; never examples in model prompts."""
import pytest

from optiflow_ctrm.validation import normalize_generated_output
from optiflow_ctrm.provenance import locate_wrapped_evidence


def normalize(source, **deal):
    value, spans = normalize_generated_output({"deal": deal}, source)
    return value["deal"], spans


@pytest.mark.parametrize("text", ["BL-4/+6", "B/L minus four through B/L plus six", "BL-4 to BL+6"])
def test_bl_grammar_preserves_offsets(text):
    deal, spans = normalize("Price: mean Example " + text, pricing={"period": None, "differential": {}})
    assert deal["pricing"]["period"]["expression"] == "BL-4..BL+6"
    assert spans["/deal/pricing/period"]


def test_wrapped_evidence_has_original_physical_spans():
    source = "é prefix\n> fixed 1.2345 USD\n> for each EUR."
    spans = locate_wrapped_evidence("fixed 1.2345 USD for each EUR", source)
    assert [source[a:b] for a, b in spans] == ["fixed 1.2345 USD", "for each EUR"]
    assert locate_wrapped_evidence("fixed 9.9999 USD for each EUR", source) == []


def test_unrelated_tail_cannot_override_price():
    source = "Price: mean Example BL+0/+4 minus USD 2.75/MT.\nUNRELATED NOTE — do not merge:\nOther price plus USD 99/MT."
    deal, _ = normalize(source, pricing={"period": None, "differential": {"value": 0, "unit": None}})
    assert deal["pricing"]["differential"] == {"value": -2.75, "unit": "USD/MT"}


def test_no_incoterm_or_statistic_inference():
    deal, _ = normalize("Delivery: delivered Hamburg. Price: Example BL+0/+2.",
        delivery_term={"as_written": "delivered Hamburg", "incoterm": "DAP", "place": "Hamburg"},
        pricing={"statistic": "mean", "period": None, "differential": {}})
    assert deal["delivery_term"]["incoterm"] is None
    assert deal["pricing"]["statistic"] is None


@pytest.mark.parametrize('count', ['two or three', 'four or five', '6-8'])
def test_missing_option_and_per_cargo_amount_remain_unknown(count):
    deal, _ = normalize(f"Qty: {count} cargoes, about 7,200 MT each; option not recorded.",
        quantity={"value": 7200, "unit": "MT", "tolerance_option": "seller"})
    assert deal["quantity"]["value"] is None
    assert deal["quantity"]["tolerance_option"] is None


def test_equal_cost_does_not_imply_joint_appointment():
    deal, _ = normalize("Inspection: Acme at load, costs equally shared", inspection={
        "company": "Acme", "location": "at load", "cost_split": "equally shared", "appointed_by": "joint"})
    assert deal["inspection"] == {"company": "Acme", "location": "load", "cost_split": "50/50", "appointed_by": None}


def test_inline_currency_and_explicit_quality():
    source = "It's diesel, ABC-7 quality. Premium USD 3/MT. USD."
    deal, spans = normalize(source, currency=None, quality_spec=None)
    assert deal['currency'] == 'USD'
    assert deal['quality_spec'] == 'ABC-7 quality'
    assert source[slice(*spans['/deal/currency'][0])] == 'USD'


def test_conflicting_prices_do_not_default_to_last():
    deal, _ = normalize('Price minus USD 2/MT. Earlier price plus USD 4/MT.',
                        pricing={'period': None, 'differential': {'value': None, 'unit': None}})
    assert deal['pricing']['differential']['value'] is None


@pytest.mark.parametrize('bl', ['BL', 'B/L', 'B / L'])
@pytest.mark.parametrize('separator', ['/', '..', ' to ', ' through '])
@pytest.mark.parametrize('offsets', [('-4', '+6'), ('minus four', 'plus six')])
def test_compositional_bl_variants(bl, separator, offsets):
    text = f'{bl} {offsets[0]}{separator}{bl} {offsets[1]}'
    deal, _ = normalize(text, pricing={'period': None, 'differential': {}})
    assert deal['pricing']['period']['expression'] == 'BL-4..BL+6'


def test_wrapping_does_not_accept_ambiguous_or_separate_paragraphs():
    assert locate_wrapped_evidence('rate USD per EUR', 'rate USD\n\nper EUR') == []
    assert locate_wrapped_evidence('rate USD', 'rate USD; rate USD') == []
    assert locate_wrapped_evidence('rate usd', 'rate USD') == []


def test_inspector_change_keeps_explicit_appointment():
    source = ('Inspection: Acme at load, appointed and paid by seller.\n'
              'use VerifyCo instead of Acme and split inspection cost 50/50.')
    deal, spans = normalize(source, inspection={
        'company': 'Acme', 'location': 'at load', 'appointed_by': 'buyer', 'cost_split': 'seller'})
    assert deal['inspection'] == {'company': 'VerifyCo', 'location': 'load',
                                 'appointed_by': 'seller', 'cost_split': '50/50'}
    assert len(spans['/deal/inspection/company']) == 2


def test_notes_keep_unstructured_dependency():
    text = 'pricing dates to follow once terminal confirms'
    deal, _ = normalize(text, notes=[text], pricing={'period': None, 'differential': {}})
    assert deal['notes'] == [text]


def test_only_verified_tolerance_duplicate_is_removed():
    quantity = {'value': 12000, 'unit': 'MT', 'tolerance_pct': 7, 'tolerance_option': 'buyer'}
    for text in ["7% more/less at buyer's choice", "seven per cent tolerance in buyer's option"]:
        deal, _ = normalize(text, notes=[text], quantity=quantity)
        assert deal['notes'] == []
    for text in ["8% more/less at buyer's choice", "7% at buyer's choice subject to terminal approval"]:
        deal, _ = normalize(text, notes=[text], quantity=quantity)
        assert deal['notes'] == [text]


@pytest.mark.parametrize('note', ["buyer's option", "at buyer's choice"])
def test_option_only_note_is_removed_only_for_the_same_quantity_option(note):
    deal, _ = normalize(note, notes=[note], quantity={'tolerance_option': 'buyer'})
    assert deal['notes'] == []
    deal, _ = normalize(note, notes=[note], quantity={'tolerance_option': 'seller'})
    assert deal['notes'] == [note]


def test_notes_do_not_accept_unsourced_additions():
    deal, _ = normalize('two or three cargoes', notes=['terminal approval tomorrow'])
    assert deal['notes'] == []


def test_shared_evidence_does_not_delete_a_residual_condition():
    from optiflow_ctrm.extractor import _remove_structured_duplicate_notes
    source = 'SGS at load, subject to terminal approval'
    deal = {'inspection': {'company': 'SGS', 'location': 'load',
                           'appointed_by': 'buyer', 'cost_split': 'buyer'}, 'notes': [source]}
    spans = {'/deal/inspection': [(0, len(source))], '/deal/notes': [(0, len(source))]}
    _remove_structured_duplicate_notes(deal, source, spans)
    assert deal['notes'] == [source]


def test_fractional_bl_offsets_are_not_silently_truncated():
    deal, _ = normalize('BL+0/+2.5', pricing={'period': None, 'differential': {}})
    assert deal['pricing']['period'] is None


@pytest.mark.parametrize('sign, amount', [('premium', 8.3), ('discount', -8.3)])
def test_spoken_price_with_cents(sign, amount):
    text = f'{sign} eight dollars\nthirty cents per metric tonne'
    deal, spans = normalize(text, pricing={'period': None, 'differential': {'value': None, 'unit': None}})
    assert deal['pricing']['differential'] == {'value': amount, 'unit': 'USD/MT'}
    assert len(spans['/deal/pricing/differential']) == 2


def test_failed_model_evidence_cannot_erase_a_parser_proven_leaf():
    from optiflow_ctrm.extractor import _reject_unproven_fields
    deal = {'inspection': {'company': 'Acme', 'location': 'load',
                           'appointed_by': 'buyer', 'cost_split': 'buyer'}}
    _reject_unproven_fields(deal, {'/deal/inspection'},
                            {'/deal/inspection/company': [(0, 4)]})
    assert deal['inspection'] == {'company': 'Acme', 'location': None,
                                  'appointed_by': None, 'cost_split': None}


def test_pronoun_amendment_is_not_overwritten_by_old_explicit_party():
    source = 'Inspection: Acme at load, appointed by buyer.\nSeller replies: now appointed by us.'
    deal, spans = normalize(source, inspection={'company': 'Acme', 'location': 'load',
                                               'appointed_by': 'seller', 'cost_split': None})
    assert deal['inspection']['appointed_by'] == 'seller'
    assert '/deal/inspection/appointed_by' not in spans
