# Redis (Memorystore for Redis Cluster) via Private Service Connect — Infra Setup

This document is for the **infra/platform team**. It is not executed by
any CI pipeline in this repo — it's a manual, one-time (per environment)
setup guide for provisioning the Redis instance backing the LLM
translation cache (`src/worker/doctranslator/translator/translation_cache.py`).

> **Product note:** GCP's PSC-based connectivity for Memorystore is only
> available today through the **Memorystore for Redis Cluster** product
> (`gcloud redis clusters ...`), not the classic single-instance
> Memorystore for Redis product (`gcloud redis instances ...`). The two
> use different provisioning APIs, different networking models, and —
> critically — different auth models:
>
> - Classic Memorystore for Redis: `--connect-mode=PRIVATE_SERVICE_CONNECT`
>   + `--auth-enabled` gives you a rotatable AUTH password string.
> - Memorystore for Redis Cluster: PSC connectivity is set up via a
>   **Network Connectivity Service Connection Policy** (a policy that
>   lets a consumer VPC/subnet auto-connect to the `gcp-memorystore-redis`
>   PSC service class), and `--auth-mode` only accepts `disabled`
>   (default) or `iam-auth` — **there is no password/AUTH-string mode**.
>
> This guide documents the **Memorystore for Redis Cluster** flow, which
> is what is actually provisioned for this project (verified working in
> `aicoesandox`: cluster `translation-llm-cache`, region `europe-west1`).
> Security is provided by PSC network isolation (only the configured
> subnet can reach the cluster) plus TLS in transit
> (`--transit-encryption-mode=SERVER_AUTHENTICATION`) — there is
> intentionally **no `REDIS_PASSWORD`** in this setup.

The application only needs two things from this setup, delivered as
Cloud Run env vars (see `.gitlab-ci.yml` / `azure-pipelines.yml` worker
deploy steps):

| App setting | Source |
|---|---|
| `REDIS_HOST` | Cluster discovery endpoint IP — see step 4 |
| `REDIS_PORT` | Port the discovery endpoint listens on (default `6379`) |

Cloud Run worker/API services already egress into the target VPC/subnet
via `--network=$CLOUD_RUN_NETWORK --subnet=$CLOUD_RUN_SUBNET` (Direct VPC
egress — see the `gcloud beta run deploy` calls in `.gitlab-ci.yml`), so
no additional VPC connector is required; the Service Connection Policy
(step 2) is what allows that same subnet to reach the Redis Cluster's PSC
endpoint.

---

## 0. Enable required APIs (one-time per project)

```bash
PROJECT_ID="aicoesandox"          # or aicoeprod for the prod environment
REGION="europe-west1"

gcloud services enable \
  networkconnectivity.googleapis.com \
  servicenetworking.googleapis.com \
  --project="$PROJECT_ID"
```

## 1. Confirm the PSC service class

Memorystore for Redis Cluster publishes its PSC service under the
service class `gcp-memorystore-redis`. This is what the Service
Connection Policy in step 2 targets.

## 2. Create a Network Connectivity Service Connection Policy

This policy is what authorizes the consumer VPC/subnet (the same
network/subnet the Cloud Run worker uses for Direct VPC egress) to
establish PSC connections to Memorystore for Redis Cluster instances.

```bash
CLOUD_RUN_NETWORK="aicoesandox-vpc"
CLOUD_RUN_SUBNET="aicoesandox-subnet"

gcloud network-connectivity service-connection-policies create translation-redis-scp \
  --project="$PROJECT_ID" \
  --region="$REGION" \
  --network="$CLOUD_RUN_NETWORK" \
  --service-class=gcp-memorystore-redis \
  --subnets="projects/${PROJECT_ID}/regions/${REGION}/subnetworks/${CLOUD_RUN_SUBNET}"

# Verify:
gcloud network-connectivity service-connection-policies describe translation-redis-scp \
  --project="$PROJECT_ID" --region="$REGION"
```

## 3. Create the Memorystore for Redis Cluster

```bash
CLUSTER_ID="translation-llm-cache"
ZONE="europe-west1-b"

gcloud redis clusters create "$CLUSTER_ID" \
  --project="$PROJECT_ID" \
  --region="$REGION" \
  --zone-distribution-mode=single-zone \
  --zone="$ZONE" \
  --replica-count=0 \
  --shard-count=1 \
  --node-type=redis-standard-small \
  --network="projects/${PROJECT_ID}/global/networks/${CLOUD_RUN_NETWORK}" \
  --transit-encryption-mode=server-authentication \
  --auth-mode=disabled
```

Notes:
- `--replica-count=0` / single shard / `redis-shared-core-nano` = the
  smallest possible footprint. Acceptable here because the cache is used
  with a fail-open pattern in application code (a Redis outage degrades
  to "no cache hits", never a hard failure). Scale `--shard-count` /
  `--node-type` / add replicas for HA if the cache becomes
  business-critical.
- `--transit-encryption-mode=server-authentication` enables TLS; the
  application sets `REDIS_TLS_ENABLED=true` to match.
- `--auth-mode=disabled` is the default and only non-IAM option. If
  stronger app-layer auth is required later, `--auth-mode=iam-auth`
  requires the client to present a GCP IAM access token per connection
  (different `redis-py` connection semantics — not currently implemented
  in `translation_cache.py`) rather than a static password.
- Cluster creation takes roughly 20–30 minutes. Poll with:
  ```bash
  gcloud redis clusters describe "$CLUSTER_ID" --project="$PROJECT_ID" \
    --region="$REGION" --format='value(state)'
  # Wait for: ACTIVE
  ```

## 4. Get the discovery endpoint (REDIS_HOST)

```bash
gcloud redis clusters describe "$CLUSTER_ID" \
  --project="$PROJECT_ID" --region="$REGION" \
  --format='value(discoveryEndpoints)'
```

This returns the PSC discovery endpoint IP (e.g. `192.168.1.12`) and
port (`6379` by default) reachable from any resource in
`$CLOUD_RUN_NETWORK` / `$CLOUD_RUN_SUBNET`, thanks to the Service
Connection Policy from step 2 — no manual forwarding-rule or static IP
reservation is needed (that manual PSC-endpoint step only applies to
classic Memorystore for Redis, not Redis Cluster).

## 5. Firewall rules

Direct VPC egress from Cloud Run already places the worker inside
`$CLOUD_RUN_NETWORK`/`$CLOUD_RUN_SUBNET`. Confirm (or add) an egress
firewall rule allowing TCP/6379 from that subnet's range to the
discovery endpoint IP if your VPC firewall policy defaults to deny:

```bash
gcloud compute firewall-rules create allow-cloudrun-to-redis-cluster \
  --project="$PROJECT_ID" \
  --network="$CLOUD_RUN_NETWORK" \
  --direction=EGRESS \
  --action=ALLOW \
  --rules=tcp:6379 \
  --destination-ranges="<DISCOVERY_ENDPOINT_IP>/32"
```

## 6. Wire into CI pipeline variables

Add `REDIS_HOST` and `REDIS_PORT` as plain **CI/CD pipeline variables**
(GitLab CI/CD → Variables, or an Azure DevOps pipeline/variable group) —
**not** Secret Manager entries. They are not sensitive: with
`--auth-mode=disabled` there is no password to protect, and the
discovery endpoint IP is only reachable from inside
`$CLOUD_RUN_NETWORK`/`$CLOUD_RUN_SUBNET` (useless outside that VPC).
They're already threaded into `--set-env-vars` (not `--set-secrets`) in
both `.gitlab-ci.yml` and `azure-pipelines.yml`.

### Current values by environment

| Environment | Project | `REDIS_HOST` | `REDIS_PORT` | Cluster / state |
|---|---|---|---|---|
| sandbox | `aicoesandox` | `192.168.1.12` | `6379` | `translation-llm-cache` (europe-west1) — `ACTIVE` |
| dev | *(not yet provisioned)* | — | — | — |
| prod | *(not yet provisioned)* | — | — | — |

Re-verify the current discovery endpoint at any time with:

```bash
gcloud redis clusters describe translation-llm-cache \
  --project=aicoesandox --region=europe-west1 \
  --format='value(discoveryEndpoints,state)'
```

There is no `REDIS_PASSWORD_SECRET_NAME` variable for this setup —
`--auth-mode=disabled` means the app connects with TLS only, no AUTH
string. `REDIS_PASSWORD` in `src/config/constants.py` stays empty/unused
and `translation_cache.py` passes `password=None` to the Redis client in
that case. (`REDIS_PASSWORD`/`REDIS_PASSWORD_SECRET_NAME` have been
removed from the `--set-secrets` wiring in `.gitlab-ci.yml` and
`azure-pipelines.yml`.)

## 7. Verify connectivity

From a host with network access to the same VPC/subnet (e.g. the Cloud
Run revision itself, or a VM/notebook attached to
`$CLOUD_RUN_NETWORK`/`$CLOUD_RUN_SUBNET`), confirm a working Redis
handshake over TLS:

```bash
python3 -c "
import redis
r = redis.Redis(host='<REDIS_HOST>', port=6379, ssl=True, ssl_cert_reqs=None,
                 socket_timeout=5, socket_connect_timeout=5, decode_responses=True)
print('PING:', r.ping())
r.set('smoketest-key', 'hello-world', ex=60)
print('GET:', r.get('smoketest-key'))
"
# Expect: PING: True / GET: hello-world
```

(`redis-cli`'s `--tls` support is inconsistent across versions for
Redis Cluster's TLS-only mode; the `redis-py` snippet above is the
recommended smoke test and matches what `translation_cache.py` does.)

## 8. Repeat per environment

Run steps 0–5 once per GCP project/environment that runs the worker
(e.g. once for `aicoesandox` dev/test, once for `aicoeprod`), each with
its own `CLUSTER_ID`, Service Connection Policy, discovery endpoint, and
CI variable values scoped to that environment/branch.

## 9. Multi-tenant sharing (other services using this same cluster)

`translation-llm-cache` was originally provisioned as a dedicated cache
for the translation worker, but it can be shared by other GCP services
in the same project as long as the two constraints below are respected.
This section is the onboarding contract for any team that wants to point
another service at this cluster instead of provisioning their own.

### 9.1 Key-prefix isolation (required)

Memorystore for Redis **Cluster mode** has no DB/AUTH-based tenant
isolation: only DB 0 exists (`SELECT` is disabled) and, per this setup,
`--auth-mode=disabled` means there is no AUTH password to scope access
either. **The only isolation mechanism available is an application-level
key-prefix convention.** Every consuming service must namespace 100% of
the keys it writes with its own prefix so that no two services can ever
read/overwrite/evict each other's keys or accidentally collide on TTLs.

- This service's convention: `translation_cache.py` prefixes every cache
  key with `settings.REDIS_KEY_PREFIX` (`REDIS_KEY_PREFIX` in
  `src/config/constants.py`, default `"translation-cache:"`), applied in
  `build_cache_key()` before the key is ever passed to `get`/`set`.
- **Any new service onboarding onto this cluster must do the same**:
  pick a unique prefix (e.g. `sales-research:`, `contract-mgmt:`),
  configure it via that service's own equivalent of `REDIS_KEY_PREFIX`,
  and apply it to every key the service writes — never write unprefixed
  keys or reuse another service's prefix.
- Keep a registry of prefixes in use (table below) and check it before
  picking a new one, to guarantee prefixes never collide.

| Prefix | Service | Notes |
|---|---|---|
| `translation-cache:` | `translation-worker-service` | LLM translation result cache, TTL 7 days (`REDIS_CACHE_TTL_SECONDS`) |

### 9.2 Network reachability (required)

Redis Cluster's PSC endpoint is only reachable from the VPC/subnet(s)
authorized by the Service Connection Policy created in step 2
(`translation-redis-scp`, scoped to `aicoesandox-vpc` /
`aicoesandox-subnet`). For another service to reach the cluster it must
either:

- Already run with Direct VPC egress into `aicoesandox-vpc` /
  `aicoesandox-subnet` (e.g. `sales-research-application` already does,
  confirmed via its Cloud Run revision config — no extra network change
  needed), **or**
- Have Direct VPC egress added to its Cloud Run deploy
  (`--network=aicoesandox-vpc --subnet=aicoesandox-subnet`) if it
  currently has no VPC network interface configured at all (e.g.
  `mcp-toolbox`, `supplier-mcp` as of this writing), **or**
- Add its own subnet to the `translation-redis-scp` Service Connection
  Policy (`--subnets=...`) if it must stay on a different subnet.

Firewall egress rules (step 5) must also permit the new service's
subnet range to reach the discovery endpoint IP on TCP/6379.

### 9.3 Capacity guidance by environment

Sharing the cluster means every consuming service's memory, CPU, and
connection usage now comes out of the same shared budget (see node-type
spec table referenced in step 3). Sizing guidance agreed for this
cluster:

- **Sandbox (`aicoesandox`)**: stay on **`redis-shared-core-nano`**
  (current tier — 0.5 vCPU, 1.12 GB usable, no SLA, max ~1,000
  connections). Acceptable for light/sandbox multi-service traffic given
  every consuming service's client (including `translation_cache.py`)
  already fails open on Redis errors. Revisit sizing if real multi-service
  traffic causes evictions, latency, or connection-limit errors.
- **Production (`aicoeprod`, not yet provisioned)**: provision at
  **`redis-standard-small`** (2 vCPU, 6.5 GB usable, has SLA) with
  `--replica-count=0` as the baseline. This gives production workloads an
  SLA-backed tier without the added cost of standby replicas; revisit
  `--replica-count` (resizable live via `gcloud redis clusters update`,
  no downtime) if automatic failover becomes a requirement once more
  services depend on this cluster in prod.
- Node-type and replica-count can both be resized in place at any time
  with `gcloud redis clusters update --node-type=... --replica-count=...
  --project=<PROJECT_ID> --region=<REGION>` — no data loss, no need to
  recreate the cluster. Note: once you move off `redis-shared-core-nano`
  to any other tier, you cannot move back down to
  `redis-shared-core-nano` again (it's explicitly a dev/test-only floor
  tier, no downgrade path).

### 9.4 Onboarding checklist for a new consuming service

1. Confirm/add network reachability (9.2).
2. Pick a unique key prefix and register it in the table in 9.1.
3. Namespace every key the new service writes/reads with that prefix.
4. Wire `REDIS_HOST`/`REDIS_PORT` into the new service the same way as
   step 6 (plain CI/CD variables, not secrets — nothing sensitive here).
5. Coordinate with the translation-worker team before assuming
   higher-than-`redis-shared-core-nano` capacity is available in
   sandbox; escalate a resize request if needed (see 9.3).
