"""WordprocessingML namespace constants and low-level element helpers.

The DOCX path edits Word documents in place: only ``w:t`` text nodes are
rewritten, and every other part of the OPC package — styles, numbering,
section properties, drawings, and crucially the ``w:tbl`` structures that make
tables editable in Word — is carried through untouched.
"""

from __future__ import annotations

from lxml import etree

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_NS = "http://www.w3.org/XML/1998/namespace"

NSMAP = {"w": W_NS}


def w(tag: str) -> str:
    """Return the fully-qualified WordprocessingML tag name."""
    return f"{{{W_NS}}}{tag}"


W_P = w("p")
W_R = w("r")
W_T = w("t")
W_TBL = w("tbl")
W_TC = w("tc")
W_RPR = w("rPr")
XML_SPACE = f"{{{XML_NS}}}space"

# Children that carry no translatable text and whose subtree must not be walked.
# ``rPr``/``pPr`` hold formatting only; the field-code and tracked-deletion tags
# hold text Word owns, which must never be sent to the model or rewritten.
SKIP_SUBTREE_TAGS: frozenset[str] = frozenset(
    {
        W_RPR,
        w("pPr"),
        w("sectPr"),
        w("tblPr"),
        w("tcPr"),
        w("trPr"),
        w("instrText"),
        w("delText"),
        w("delInstrText"),
    }
)

# Inline elements that occupy a position in the text flow. Text on either side
# of one must stay in its own run so tabs, breaks, images, and note anchors keep
# their exact position after translation.
FLOW_BREAK_TAGS: frozenset[str] = frozenset(
    {
        w("br"),
        w("cr"),
        w("tab"),
        w("ptab"),
        w("sym"),
        w("noBreakHyphen"),
        w("softHyphen"),
        w("drawing"),
        w("pict"),
        w("object"),
        w("fldChar"),
        w("footnoteReference"),
        w("endnoteReference"),
        w("commentReference"),
        w("footnoteRef"),
        w("endnoteRef"),
        w("separator"),
        w("continuationSeparator"),
        w("pgNum"),
        w("lastRenderedPageBreak"),
    }
)


def owning_run(text_node: etree._Element) -> etree._Element | None:
    """Return the ``w:r`` ancestor of a ``w:t`` node, if any."""
    parent = text_node.getparent()
    while parent is not None:
        if parent.tag == W_R:
            return parent
        if parent.tag == W_P:
            return None
        parent = parent.getparent()
    return None


def run_format_key(text_node: etree._Element) -> str:
    """Build a stable grouping key for the formatting applied to a text node.

    The key combines the run's serialized ``w:rPr`` with the identity of the
    inline containers between the run and its paragraph (hyperlinks, tracked
    insertions, smart tags). Runs only merge when both match, so a link never
    absorbs the plain text next to it.
    """
    run = owning_run(text_node)
    if run is None:
        return "no-run"

    rpr = run.find(W_RPR)
    if rpr is None:
        fmt = ""
    else:
        fmt = etree.tostring(rpr, encoding="unicode")

    containers: list[str] = []
    parent = run.getparent()
    while parent is not None and parent.tag != W_P:
        containers.append(f"{parent.tag}#{id(parent)}")
        parent = parent.getparent()
    return f"{fmt}|{'>'.join(reversed(containers))}"


def set_text(text_node: etree._Element, value: str) -> None:
    """Write text into a ``w:t`` node, preserving significant whitespace."""
    text_node.text = value
    if value != value.strip():
        text_node.set(XML_SPACE, "preserve")
    elif XML_SPACE in text_node.attrib:
        del text_node.attrib[XML_SPACE]
