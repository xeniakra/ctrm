"""Pydantic request/response models and the canonical JSON Schema.

Keep the externally returned deal shape here. The extractor and MCP server must
use these types; no endpoint should maintain a competing ad-hoc dictionary.
"""

from __future__ import annotations

from datetime import date
from collections.abc import Iterable
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


NonEmptyText = Annotated[str, StringConstraints(min_length=1)]
CurrencyCode = Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")]
JsonPointer = Annotated[
    str, StringConstraints(pattern=r"^/(?:[^/~]|~[01])+(?:/(?:[^/~]|~[01])*)*$")
]
ClarificationField = Literal[
    "/deal/recap_date",
    "/deal/buyer",
    "/deal/seller",
    "/deal/product/as_written",
    "/deal/quantity/value",
    "/deal/quantity/unit",
    "/deal/quantity/tolerance_pct",
    "/deal/quantity/tolerance_option",
    "/deal/delivery_term/incoterm",
    "/deal/delivery_term/place",
    "/deal/delivery_window/from",
    "/deal/delivery_window/to",
    "/deal/pricing/quote_code",
    "/deal/pricing/statistic",
    "/deal/pricing/period",
    "/deal/pricing/differential/value",
    "/deal/pricing/differential/unit",
    "/deal/payment_terms",
    "/deal/currency",
    "/deal/fx/rate",
    "/deal/fx/basis",
]
EvidenceField = Literal[
    "/deal/recap_date",
    "/deal/buyer",
    "/deal/seller",
    "/deal/product/as_written",
    "/deal/quality_spec",
    "/deal/quantity",
    "/deal/delivery_term",
    "/deal/delivery_window",
    "/deal/pricing/quote_code",
    "/deal/pricing/statistic",
    "/deal/pricing/period",
    "/deal/pricing/differential",
    "/deal/payment_terms",
    "/deal/currency",
    "/deal/fx",
    "/deal/inspection",
    "/deal/demurrage",
    "/deal/law",
    "/deal/arbitration",
    "/deal/vessel",
    "/deal/notes",
]

QuantityUnit = Literal["MT", "BBL"]
ToleranceOption = Literal["seller", "buyer"]
Incoterm = Literal["EXW", "FAS", "FOB", "CFR", "CIF", "DAP", "DDP"]
PricingStatistic = Literal["mean", "high", "low", "mean_of_high"]
DifferentialUnit = Literal["USD/MT", "USD/BBL"]
QuoteSourceColumn = Literal["mean", "high", "low"]


class StrictModel(BaseModel):
    """Shared external-model policy: unknown keys are never silently ignored."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Product(StrictModel):
    as_written: NonEmptyText | None
    product_code: NonEmptyText | None
    reference_name: NonEmptyText | None


class Quantity(StrictModel):
    value: Annotated[float, Field(gt=0)] | None
    unit: QuantityUnit | None
    tolerance_pct: Annotated[float, Field(ge=0, le=100)] | None
    tolerance_option: ToleranceOption | None


class DeliveryTerm(StrictModel):
    # Keeps trader shorthand even when it cannot safely be classified.
    as_written: NonEmptyText | None
    incoterm: Incoterm | None
    place: NonEmptyText | None


class DeliveryWindow(StrictModel):
    from_date: date | None = Field(alias="from")
    to_date: date | None = Field(alias="to")

    @model_validator(mode="after")
    def dates_are_ordered(self) -> DeliveryWindow:
        if (
            self.from_date is not None
            and self.to_date is not None
            and self.from_date > self.to_date
        ):
            raise ValueError("delivery window start must not be after its end")
        return self


class BlRelativePricingPeriod(StrictModel):
    kind: Literal["bl_relative"]
    expression: NonEmptyText
    from_offset_days: int
    to_offset_days: int

    @model_validator(mode="after")
    def offsets_are_ordered(self) -> BlRelativePricingPeriod:
        if self.from_offset_days > self.to_offset_days:
            raise ValueError("pricing-period start offset must not be after its end offset")
        return self


class ExplicitPricingPeriod(StrictModel):
    kind: Literal["date_range"]
    expression: NonEmptyText
    from_date: date
    to_date: date

    @model_validator(mode="after")
    def dates_are_ordered(self) -> ExplicitPricingPeriod:
        if self.from_date > self.to_date:
            raise ValueError("pricing-period start must not be after its end")
        return self


PricingPeriod = Annotated[
    BlRelativePricingPeriod | ExplicitPricingPeriod,
    Field(discriminator="kind"),
]


class PriceDifferential(StrictModel):
    value: float | None
    unit: DifferentialUnit | None


class Pricing(StrictModel):
    quote_code: NonEmptyText | None
    statistic: PricingStatistic | None
    period: PricingPeriod | None
    differential: PriceDifferential


class PaymentPortion(StrictModel):
    """The explicitly stated amount represented by one payment."""

    kind: Literal["percentage", "balance", "full"] | None
    percentage: Annotated[float, Field(gt=0, le=100)] | None

    @model_validator(mode="after")
    def percentage_matches_kind(self) -> PaymentPortion:
        if self.kind == "percentage" and self.percentage is None:
            raise ValueError("percentage payment portion requires percentage")
        if self.kind != "percentage" and self.percentage is not None:
            raise ValueError("percentage is only valid for a percentage payment portion")
        return self


class PaymentDue(StrictModel):
    """A deadline without silently treating ordinary days as banking days."""

    qualifier: Literal["within", "net", "no_later_than"] | None
    days: Annotated[int, Field(gt=0)] | None
    day_basis: Literal["days", "banking_days", "business_days"] | None
    relation: Literal["from", "after", "before", "at", "of"] | None
    event: Literal[
        "bill_of_lading",
        "final_pricing",
        "loading",
        "lifting",
        "each_lifting",
        "sight",
    ] | None

    @model_validator(mode="after")
    def days_and_basis_are_paired(self) -> PaymentDue:
        if (self.days is None) != (self.day_basis is None):
            raise ValueError("payment due days and day_basis must be supplied together")
        return self


class PaymentDocumentCondition(StrictModel):
    relation: Literal["against"]
    document_type: Literal[
        "documents",
        "scanned_documents",
        "original_documents",
        "shipping_documents",
    ]


class PaymentInstallment(StrictModel):
    portion: PaymentPortion
    stage: Literal["provisional", "final"] | None
    instrument: Literal["cash", "prepayment", "letter_of_credit"] | None
    irrevocable: bool | None
    due: PaymentDue | None
    document_condition: PaymentDocumentCondition | None


class LateInterest(StrictModel):
    rate_pct: Annotated[float, Field(gt=0)]
    basis: Literal["per_annum"]


class PaymentTerms(StrictModel):
    """Original wording plus a machine-readable payment schedule."""

    as_written: NonEmptyText
    payments: Annotated[list[PaymentInstallment], Field(min_length=1, max_length=4)]
    late_interest: LateInterest | None


class PriceQuotePeriod(StrictModel):
    from_date: date = Field(alias="from")
    to_date: date = Field(alias="to")

    @model_validator(mode="after")
    def dates_are_ordered(self) -> PriceQuotePeriod:
        if self.from_date > self.to_date:
            raise ValueError("quote-period start must not be after its end")
        return self


class PriceQuoteObservation(StrictModel):
    date: date
    low: float
    high: float
    mean: float
    selected_value: float


class PriceQuoteResult(StrictModel):
    """Canonical JSON-safe response returned by deterministic quote lookup."""

    quote_code: NonEmptyText
    quote_name: NonEmptyText
    publication: NonEmptyText
    unit: NonEmptyText
    statistic: PricingStatistic
    period: PriceQuotePeriod
    observations: Annotated[list[PriceQuoteObservation], Field(min_length=1)]
    observation_count: Annotated[int, Field(gt=0)]
    aggregation: Literal["arithmetic_mean"]
    source_column: QuoteSourceColumn
    value: float
    formula: NonEmptyText

    @model_validator(mode="after")
    def observation_count_matches(self) -> PriceQuoteResult:
        if self.observation_count != len(self.observations):
            raise ValueError("observation_count must equal the number of observations")
        return self


class FxTerms(StrictModel):
    rate: Annotated[float, Field(gt=0)] | None
    quote: NonEmptyText | None
    basis: NonEmptyText | None


class Inspection(StrictModel):
    company: NonEmptyText | None
    location: NonEmptyText | None
    appointed_by: Literal["seller", "buyer", "joint"] | None
    cost_split: NonEmptyText | None


class Demurrage(StrictModel):
    rate_per_day: Annotated[float, Field(ge=0)] | None
    currency: CurrencyCode | None
    basis: NonEmptyText | None


class Deal(StrictModel):
    recap_date: date | None
    buyer: NonEmptyText | None
    seller: NonEmptyText | None
    product: Product
    quantity: Quantity
    delivery_term: DeliveryTerm
    delivery_window: DeliveryWindow
    pricing: Pricing
    payment_terms: PaymentTerms | None
    currency: CurrencyCode | None
    fx: FxTerms | None
    quality_spec: NonEmptyText | None
    inspection: Inspection | None
    demurrage: Demurrage | None
    law: NonEmptyText | None
    arbitration: NonEmptyText | None
    vessel: NonEmptyText | None
    notes: list[NonEmptyText]


class ClarificationQuestion(StrictModel):
    field: ClarificationField
    question: NonEmptyText


class SourceSpan(StrictModel):
    kind: Literal["source"]
    start: Annotated[int, Field(ge=0)]
    end: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def end_is_after_start(self) -> SourceSpan:
        if self.end <= self.start:
            raise ValueError("source span end must be greater than start")
        return self


class ReferenceEvidence(StrictModel):
    kind: Literal["reference"]
    file: NonEmptyText
    key: NonEmptyText


Evidence = Annotated[SourceSpan | ReferenceEvidence, Field(discriminator="kind")]


class ProvenanceEntry(StrictModel):
    field: JsonPointer
    evidence: Annotated[list[Evidence], Field(min_length=1)]


class ModelEvidenceClaim(StrictModel):
    """Compact internal evidence: one verbatim quote can support many fields."""

    quote: Annotated[str, StringConstraints(min_length=3, max_length=500)]
    fields: Annotated[list[EvidenceField], Field(min_length=1, max_length=12)]


# A mandatory field can be null, because abstention is part of the contract.
# It then needs a question whose pointer names exactly that missing field.
MANDATORY_PATHS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("/deal/recap_date", ("recap_date",)),
    ("/deal/buyer", ("buyer",)),
    ("/deal/seller", ("seller",)),
    ("/deal/product/as_written", ("product", "as_written")),
    ("/deal/quantity/value", ("quantity", "value")),
    ("/deal/quantity/unit", ("quantity", "unit")),
    ("/deal/quantity/tolerance_pct", ("quantity", "tolerance_pct")),
    ("/deal/quantity/tolerance_option", ("quantity", "tolerance_option")),
    ("/deal/delivery_term/incoterm", ("delivery_term", "incoterm")),
    ("/deal/delivery_term/place", ("delivery_term", "place")),
    ("/deal/delivery_window/from", ("delivery_window", "from_date")),
    ("/deal/delivery_window/to", ("delivery_window", "to_date")),
    ("/deal/pricing/quote_code", ("pricing", "quote_code")),
    ("/deal/pricing/statistic", ("pricing", "statistic")),
    ("/deal/pricing/period", ("pricing", "period")),
    ("/deal/pricing/differential/value", ("pricing", "differential", "value")),
    ("/deal/pricing/differential/unit", ("pricing", "differential", "unit")),
    ("/deal/payment_terms", ("payment_terms",)),
    ("/deal/currency", ("currency",)),
)


class ExtractionPayload(StrictModel):
    deal: Deal
    questions: Annotated[list[ClarificationQuestion], Field(max_length=21)]

    @model_validator(mode="after")
    def enforce_safe_abstention(self) -> ExtractionPayload:
        question_paths = [question.field for question in self.questions]
        duplicates = {path for path in question_paths if question_paths.count(path) > 1}
        if duplicates:
            raise ValueError(f"duplicate clarification questions for: {sorted(duplicates)}")

        missing = [
            pointer for pointer, path in MANDATORY_PATHS if _read_path(self.deal, path) is None
        ]
        if self.deal.currency not in {None, "USD"}:
            if self.deal.fx is None or self.deal.fx.rate is None:
                missing.append("/deal/fx/rate")
            if self.deal.fx is None or self.deal.fx.basis is None:
                missing.append("/deal/fx/basis")

        unanswered = [pointer for pointer in missing if pointer not in question_paths]
        if unanswered:
            raise ValueError(f"missing mandatory values require questions for: {unanswered}")
        return self


class GeneratedExtraction(ExtractionPayload):
    """Internal result produced by the commercial-field inference call."""


class ModelEvidenceResponse(StrictModel):
    evidence: Annotated[
        dict[
            EvidenceField,
            Annotated[
                list[Annotated[str, StringConstraints(min_length=3, max_length=500)]],
                Field(min_length=1, max_length=8),
            ],
        ],
        Field(min_length=1, max_length=21),
    ]


class ExtractionResult(ExtractionPayload):
    provenance: list[ProvenanceEntry]


def _read_path(value: Any, path: tuple[str, ...]) -> Any:
    for part in path:
        value = getattr(value, part)
    return value


# Convenient canonical schema for llama.cpp constrained decoding.  Callers may
# also regenerate it from ExtractionResult to pick up intentional model changes.
def extraction_json_schema() -> dict[str, Any]:
    """Schema sent to llama.cpp; deterministic code attaches provenance."""

    return GeneratedExtraction.model_json_schema(by_alias=True, mode="validation")


def evidence_json_schema(fields: Iterable[EvidenceField] | None = None) -> dict[str, Any]:
    """Compact schema whose evidence keys can be fixed to the requested fields."""

    schema = ModelEvidenceResponse.model_json_schema(by_alias=True, mode="validation")
    if fields is None:
        return schema
    requested = sorted(set(fields))
    schema["properties"]["evidence"] = {
        "type": "object",
        "properties": {
            field: {
                "type": "array",
                "items": {"type": "string", "minLength": 3, "maxLength": 500},
                "minItems": 1,
                "maxItems": 8,
            }
            for field in requested
        },
        "required": requested,
        "additionalProperties": False,
        "minProperties": len(requested),
        "maxProperties": len(requested),
    }
    return schema


def canonical_json_schema() -> dict[str, Any]:
    """Complete externally returned schema, including source provenance."""

    return ExtractionResult.model_json_schema(by_alias=True, mode="validation")


# Descriptive aliases keep integrations readable while preserving one schema.
DealExtraction = ExtractionResult
RecapExtraction = ExtractionResult
