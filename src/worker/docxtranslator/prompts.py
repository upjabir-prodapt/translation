"""Batch translation prompt for Word paragraphs.

Mirrors the PDF batch prompt contract (stable ids, one output per input, no
merging) and adds the inline ``<gN>`` run-span rules the DOCX write-back needs
in order to put each translated span back into its original run.
"""

from string import Template

DOCX_PROMPT_TEMPLATE = Template(
    """$role_block

## Structure Rules
1. Return **exactly one output per input item**, with the same "id".
2. Each item is one paragraph of a Word document — a heading, a body
   paragraph, a table cell, a text box, a header, a footer, or a note.
   → Treat every item as an **independent, fixed unit**.
   → Do NOT merge items, split items, or move content between items.
3. Translate ALL human-readable content into $lang_out.
4. Never return an empty output for a non-empty input.

## Inline Run Tags
Some inputs are split into spans by tags like `<g0>…</g0><g1>…</g1>`.
Each span is a differently formatted piece of the same paragraph (bold, a
hyperlink, a different font).
- Keep **every tag, with the same index, exactly once**, in ascending order.
- Do NOT add, drop, renumber, or nest tags.
- Put each translated word inside the span that carries its meaning. You MAY
  move words across span boundaries when $lang_out word order requires it.
- A span may end up empty (`<g1></g1>`) if the target word order demands it.
- Preserve leading and trailing spaces inside spans; they are real spacing in
  the document.

## Do NOT Modify
- Placeholders: `{v1}`, `{name}`, `%s`, `%d`, `[[...]]`, `%%...%%` — keep exactly unchanged.
- Data-masking tokens matching `__DLP_TOKEN_NNNN__` — copy character-for-character.
- URLs, file paths, email addresses, product codes, and version strings.

$glossary_usage_rules_block
## Output Format
Return JSON with an "items" array of the same length as the input.
For each item, keep the same "id" and add "output" with the translated text only.
No extra text, no ```json blocks.

### Example
Input:
[
    {
    "id": 0,
    "input": "<g0>Payment </g0><g1>within 30 days</g1>"
    }
]
Output:
{
    "items": [
    {
    "id": 0,
    "output": "<g0>Pago </g0><g1>en un plazo de 30 días</g1>"
    }
    ]
}

## Style
- Produce fluent, professional $lang_out.
- Match the register of the source (contractual, technical, marketing, UI).
- Preserve punctuation unless target-language fluency requires otherwise.

$glossary_tables_block

## Here is the input:

$json_input_str"""
)
