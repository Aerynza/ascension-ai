# Aerynza AI identity migration

## Canonical brand

The intelligence engine is **Aerynza AI**, shared by **AerynzaLife**, **AerynzaSprout**, **FamilyOS** (household), and **Nexus** (extended network). AP remains the personal assistant shell. `core` is a tier/capability classification where the product contract defines it, not a new product identity.

GitHub target repository slug: `Aerynza/aerynza-ai` (currently `Aerynza/ascension-ai`).

## Safe migration sequence

1. Apply and test the brand-only source changes, including package metadata, lockfile, public labels and contract regression tests. Do not change existing shell values `nexus_home`, `nexus_family`, `sprout`, `ap`, or `lifeos` in this migration: they are established API values.
2. Rename the existing GitHub repository **in place** under **Settings → General → Repository name**. Do **not** create a new repository and copy the code: keep commits, releases, issues, pull requests and provenance. Ensure the target slug is available, then verify the new URL and old redirect. The repository rename is an owner/admin settings operation, not a source commit.
3. Update explicit clone URLs, GitHub Actions cross-repository references, service source/repo connections, package scripts, notebooks and internal links to `Aerynza/aerynza-ai` in a separate audited pass. Do not rely indefinitely on redirects.
4. On Render, confirm the **actual** Aerynza AI service's source, branch, custom domains and current deployment before altering a service name or URL. A `render.yaml` name change might provision a **new** service, rather than rename the current one. Keep the serving endpoint functioning while all callers migrate.
5. Preserve existing production environment variable names (`ASCENSION_AI_*`), secret values, auth compatibility, model storage/checksums and API routes until a separately tested dual-read alias migration covers every consumer. The cosmetic rename does **not** rotate credentials or promise native inference promotion.
6. Verify health, signed server-to-server authentication, model readiness, AerynzaLife routing, AP chat, Sprout access controls, FamilyOS household isolation, and Nexus opt-in disclosure boundaries before cutting over any endpoint. Keep external-model fallback and rollback available until production evidence justifies removal.

## Non-goals of this source PR

This source change does **not** rename the GitHub repository, Render services, DNS or environment variable keys; does not deploy, replace models, train, change access permissions or incur new hosting costs. A separate verified operations change is required for the externally visible repository and service rename.

Run `python -m unittest discover -s tests -p test_aerynza_ai_brand_contract.py` to check the source identity contract. This is a brand/integration compatibility check, not a production deployment or model-safety certification.
