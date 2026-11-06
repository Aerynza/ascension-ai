# Aerynza native intelligence — product relationship and public evidence index

**Status (2026-11-06):** architecture mapping and review index, **not** a production-certification, security attestation, patent claim, or legal conclusion. This repository is public. No private keys, customer data, proprietary corpus samples, unpublished invention claims, or confidential counsel material belong here.

## Canonical relationship

AerynzaLife (LifeOS) is the foundational human-coordination product and authority for user records, identity, privacy permissions, entitlements, and executed actions. Aerynza AI is the **master native intelligence product** created from that architecture. Neither is merely an incidental feature of the other.

Conceptual design, consistent with [LifeOS architecture/IP issue #41](https://github.com/Aerynza/Aerynza/issues/41):

1. Human-directed goals, signals, and corrections enter authorized LifeOS domains.
2. LifeOS / Human Model / domain Vortexes prepare only the context the caller may share with the requested AI shell.
3. Aerynza AI reasons over that supplied context and proposes responses or possible actions. The model is **not** a principal with independent rights over the user's data.
4. The authenticated product may stage a proposal for review or execute a permitted action, after the necessary approval and checks, and generates verifiable execution receipts.
5. Only separately permitted interaction or outcome evidence may return to LifeOS to improve later context. A model's description of an action is never a receipt.

The toroidal/Vortex terminology describes the product's intended circulating coordination architecture; this index does not assert that every conceptual return path is implemented or validated. A user's refusal of a data share must stop the corresponding path.

## Product mapping (protocol identifiers unchanged)

| Existing AI shell enum | Product boundary | Required limit |
| --- | --- | --- |
| `ap` | Individual AP | One person's private context only |
| `lifeos` | AerynzaLife domain intelligence | Only caller-authorized domain context |
| `nexus_home` | **FamilyOS**, home/household | Explicitly shared household data; no unapproved private AP context |
| `nexus_family` | **Nexus**, extended family/friends/enterprise network | Explicit member consent; network membership is not account authority |
| `sprout` | **AerynzaSprout** | Guardian permissions, age-appropriate privacy, no child-authorized membership removal |
| `core` | Neutral model/tier context | Not a new owner, permission principal, or separate relationship shell |
| `creation` | Creation workflow | No execution or publication without product confirmation |

The existing enum strings are compatibility interfaces; changing product names in prompts or documents does not rename an API request or modify entitlements. Permission enforcement must happen in the authenticated calling product.

## Public source pointers for independent engineering review

| Mechanism / claim to examine | Source reference | Evidence limitation |
| --- | --- | --- |
| Shell vocabulary and model response instructions | `src/core/contracts.py` | Prompt boundaries are not proof of server-side authorization |
| Context scope and reasoning orchestration | `src/core/orchestrator.py` | Inspect caller-supplied context and integration behavior separately |
| Proposal/action versus execution/receipt contract | `src/core/action_runtime.py` | A model action catalog does not establish a successful provider write |
| Signed HTTP envelopes and trusted issuer | `src/serving/http_envelope.py`, `src/serving/envelope_gateway.py` | Cross-account/resource ownership still requires end-to-end validation |
| Replay prevention | `src/serving/replay_guard.py` | Unit/fake nonce tests do **not** prove real distributed Redis or Postgres deployment |
| Native HTTP serving and request guards | `src/serving/api.py` | Proxy buffering, load, cancellation, body limits and error handling need deployed testing |
| Pinned model profiles | `config/model_profiles.json` | A configured/checksummed checkpoint is not a quality certification |
| Native release evidence gate | `src/core/release_readiness.py` | Release stays held until signed-admission and checkpoint-bound evidence gates pass |
| Existing model-free assessment | `scripts/replacement_readiness_eval.py`, `scripts/replacement_contract_eval.py` | Passes do not substitute for inference trials on the deployed checkpoint |

### Current unresolved acceptance

- [AI admission and five-shell evidence issue #10](https://github.com/Aerynza/ascension-ai/issues/10): distributed replay tests, bounded concurrent traffic, cancellation/timeout recovery, per-resource owner/consent checks, and real checkpoint-bound native quality/latency/safety tests are still necessary.
- [LifeOS coordination issue #40](https://github.com/Aerynza/Aerynza/issues/40): maintain API ownership, private context, action receipts, test evidence, and release discipline across the two repositories.
- Outside-AI fallback remains available. **Do not** mark native replacement ready or imply that merely switching shell labels has solved tenant isolation.

## Disclosure and counsel handoff

The source paths above identify *potential implementation evidence*. Counsel should review invention concepts, priority/ownership history, publication timelines, disclosures, alternative embodiments, third-party licenses, and what must remain confidential. This public document intentionally excludes claim-concept specifics, secret data pipelines, unreviewed private design variants, legal conclusions, and any invented conception or reduction-to-practice dates. Use actual commit history and release records as the evidence source, not this index as an attestation.

No paid infrastructure, model change, deployment, new credential, or permission expansion is authorized by this document.
