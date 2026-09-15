"""Who a given LLM call is being made on behalf of.

The Apigee `llm` gateway meters and rate-limits on two asserted headers,
`x-colt-user-oid` and `x-colt-user-department` (AICOE-Terraform `docs/23` S4.2).
Every LLM call this service makes must carry them, or it lands in the gateway's
`system`/`unattributed` bucket and neither per-user nor per-department quotas
mean anything.

Why a ContextVar rather than function parameters
------------------------------------------------
Identity enters the worker at exactly one point -- `TranslateTaskHandler.handle`
re-reads the job row from BigQuery, because the Cloud Tasks payload is
deliberately PII-free -- and is needed at four LLM client construction sites
sitting behind three separate call chains (domain classifier, translator
factory, quality judge). Threading it explicitly would mean new parameters on
roughly ten functions, and every future call site would have to remember to
pass it; forgetting degrades silently to `unattributed` with no error. A
ContextVar read inside the gateway helper makes attribution the default instead
of an opt-in, which is the whole point of centralising on that helper.

`asyncio.to_thread` copies the current context, so the thread offloads in this
package (e.g. the blocking domain classifier) see the identity. Bespoke
executor pools do NOT necessarily -- see the note in
`src/worker/doctranslator/utils/priority_thread_pool_executor.py`. No LLM
client is constructed inside one today; if that changes, that pool must copy
the context.

Trust model
-----------
These values are ASSERTED by this backend, not derived from a verified token --
at worker time the user's Entra token is long gone. A compromised or buggy
backend could misattribute usage or evade a per-user quota. That is the same
hybrid-trust compromise already accepted for `department` on the `int` proxy,
and it is acceptable because callers are trusted internal services gated by API
key and network isolation. It does mean these quotas are a **cost-control
mechanism, not a security boundary** -- do not present them as the latter.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass

__all__ = ["LlmIdentity", "current_llm_identity", "use_llm_identity"]


@dataclass(frozen=True, slots=True)
class LlmIdentity:
    """The user an LLM call is attributed to.

    `oid` is the Entra object id where available. Translation persists the
    submitting user's *email* as `cost_attribution.user_id` and has no
    top-level user column, so the email is the fallback -- see
    `use_llm_identity`.
    """

    oid: str = ""
    department: str = ""


_current: ContextVar[LlmIdentity | None] = ContextVar("llm_identity", default=None)


def current_llm_identity() -> LlmIdentity | None:
    """The identity in scope, or None when there is no user behind this call."""
    return _current.get()


@contextlib.contextmanager
def use_llm_identity(oid: str | None, department: str | None) -> Iterator[None]:
    """Attribute every LLM call made inside this block to one user.

    Blank values are kept as blanks rather than invented sentinels: the gateway
    omits the corresponding header, and Apigee's own `AM-Identity` policy
    applies its documented `system` / `unattributed` defaults. Deciding that
    here as well would give two places to disagree about what "no user" means.

    The token is always reset, so a later job on the same Cloud Run instance
    can never inherit the previous job's user -- these containers are reused.
    """
    token = _current.set(
        LlmIdentity(oid=(oid or "").strip(), department=(department or "").strip())
    )
    try:
        yield
    finally:
        _current.reset(token)
