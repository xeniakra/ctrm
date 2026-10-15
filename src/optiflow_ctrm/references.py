"""Deterministic readers and resolvers for products.csv and price_quotes.csv."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re


DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True, slots=True)
class ProductReference:
    code: str
    name: str


@dataclass(frozen=True, slots=True)
class QuoteReference:
    code: str
    name: str


@dataclass(frozen=True, slots=True)
class ReferenceCatalog:
    products: tuple[ProductReference, ...]
    quotes: tuple[QuoteReference, ...]

    @classmethod
    def load(cls, data_dir: Path = DATA_DIR) -> ReferenceCatalog:
        with (data_dir / "products.csv").open(newline="", encoding="utf-8") as stream:
            products = tuple(
                ProductReference(row["product_code"], row["product_name"])
                for row in csv.DictReader(stream)
            )
        quotes_by_code: dict[str, QuoteReference] = {}
        with (data_dir / "price_quotes.csv").open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                quotes_by_code.setdefault(
                    row["quote_code"], QuoteReference(row["quote_code"], row["quote_name"])
                )
        return cls(products=products, quotes=tuple(quotes_by_code.values()))

    def resolve_product(self, value: str | None) -> ProductReference | None:
        return _resolve_product(value, self.products)

    def resolve_quote(self, value: str | None) -> QuoteReference | None:
        return _resolve(value, self.quotes)


def _resolve_product(
    value: str | None, choices: tuple[ProductReference, ...]
) -> ProductReference | None:
    if not value:
        return None
    for choice in choices:
        if value.casefold() == choice.code.casefold():
            return choice
    value_tokens = _reference_tokens(value)
    code_matches = [
        choice
        for choice in choices
        if _is_code_prefix(_reference_tokens(choice.code), value_tokens)
    ]
    if len(code_matches) == 1:
        return code_matches[0]
    needle = _normalize(value)
    matches = [
        choice
        for choice in choices
        if needle == _normalize(choice.name)
        or (len(needle) >= 6 and needle in _normalize(choice.name))
        or (len(_normalize(choice.name)) >= 6 and _normalize(choice.name) in needle)
    ]
    return matches[0] if len(matches) == 1 else None


def _resolve(value: str | None, choices):
    if not value:
        return None
    for choice in choices:
        if value.casefold() == choice.code.casefold():
            return choice
    needle = _normalize(value)
    matches = [
        choice
        for choice in choices
        if needle == _normalize(choice.name)
        or (len(needle) >= 6 and needle in _normalize(choice.name))
        or (len(_normalize(choice.name)) >= 6 and _normalize(choice.name) in needle)
    ]
    return matches[0] if len(matches) == 1 else None


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _reference_tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z]+|\d+", value.casefold()))


def _is_code_prefix(code_tokens: tuple[str, ...], value_tokens: tuple[str, ...]) -> bool:
    return (
        len("".join(code_tokens)) >= 5
        and len(value_tokens) >= len(code_tokens)
        and value_tokens[: len(code_tokens)] == code_tokens
    )
