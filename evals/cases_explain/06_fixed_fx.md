# Fixed FX

**Special:** A voice-note transcription states numbers in words, uses the daily `high`, limits
pricing to B/L day, and supplies a fixed USD/EUR conversion.

**Perturbations:** Conversational dictation, wrapped sentences, words instead of digits,
uppercase emphasis, and a spoken decimal amount.

**Failure caught:** Mapping `high` to `mean_of_high`, expanding a one-day pricing period,
reversing the FX quote, or missing buyer-paid inspection.
