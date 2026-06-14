"""Pydantic schemas for structured LLM translation responses."""

from pydantic import BaseModel
from pydantic import Field


class TranslationResponse(BaseModel):
    """Structured translation response from the model."""

    translated_text: str = Field(description="The translated text.")


class BatchTranslationItem(BaseModel):
    id: int = Field(description="The id of the input paragraph.")
    output: str = Field(description="The translated text for this paragraph.")


class BatchTranslationResponse(BaseModel):
    """Structured response for batch paragraph translation."""

    items: list[BatchTranslationItem] = Field(
        description="Translated paragraphs in the same order as the input."
    )


class ExtractedTerm(BaseModel):
    src: str = Field(description="Source language term.")
    tgt: str = Field(description="Translated term in the target language.")


class TermExtractionResponse(BaseModel):
    """Structured response for automatic term extraction."""

    terms: list[ExtractedTerm] = Field(
        description="Extracted term pairs from the source text."
    )
