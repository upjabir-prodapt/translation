# Colt internal CA

`colt-internal-ca.pem` is the two-certificate chain (Internal COLT Issuing CA2 V3 -> Internal COLT
Root CA V3, **no leaf**) that signs `llm.aicoedev-int.colt.net` — the Apigee LLM gateway this service
routes every model call through — and `aihub-api.aicoedev-int.colt.net`.

This is public CA material with no private key. It is safe to commit, and it is the same file
`shared_ui/aihub-ui/certs/colt-internal-ca.pem` carries.

## Why it is needed

Python's default trust store does not know this CA, so every call to the gateway fails with
`CERTIFICATE_VERIFY_FAILED: self-signed certificate in certificate chain`. That was the live failure
that took this service down once the gateway was enabled.

## How it is installed, and the two traps

Both Dockerfiles copy it to `/usr/local/share/ca-certificates/colt-internal-ca.crt` and run
`update-ca-certificates`.

1. **The filename must end in `.crt`** even though the content is PEM. `update-ca-certificates`
   silently skips any other extension — no error, just an untrusted CA.
2. **`update-ca-certificates` alone is not enough.** It updates the OpenSSL/system store, but `httpx`
   (used by `google-genai` and `anthropic`) and `requests` both default to `certifi`'s own bundle and
   would keep failing. The Dockerfiles therefore also set `SSL_CERT_FILE` and `REQUESTS_CA_BUNDLE` to
   the system bundle. This is the half of the fix that is easy to miss, and it presents as a TLS error
   only at the first outbound call in a deployed container.

## Renewing

Replace this file if Colt rotates the Issuing CA or the Root CA. A leaf renewal without a CA rotation
needs no change here.
