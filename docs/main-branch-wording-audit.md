# Main-Branch Public Wording Audit

Date: 2026-06-09

## Scope

This audit covers the current public main branch after the public-benefit and
repo-credibility pass. It focuses on wording, documentation, funding language, grant
readiness, donation readiness, and repository hygiene. It does not validate GPU runtime
behavior, Docker builds, ROCm builds, model downloads, or benchmark reproduction.

## Current Public Technical Claims Found

- [README.md](../README.md) states the public mission, current GFX906 / MI50-class
  focus, prebuilt Docker Hub runtime image tag, Docker Hub digest, vLLM serving command,
  archive hash, and reference fixed-token TPS results.
- [qwen36-gfx906/README.md](../qwen36-gfx906/README.md) documents the Qwen3.6-35B-A3B
  runtime profile, build pins, Docker archive SHA-256 values, Docker Hub image identity,
  runtime defaults, disk-space checks, GPU VRAM checks, smoke-test flow, and scoped TPS
  results.
- [qwen36-gfx906/manifest.json](../qwen36-gfx906/manifest.json) describes the runtime
  profile, command, hardware scope, runtime environment, and bundled file expectations.
- [qwen36-gfx906/deploy.sh](../qwen36-gfx906/deploy.sh),
  [qwen36-gfx906/smoke-test.sh](../qwen36-gfx906/smoke-test.sh), and
  [qwen36-gfx906/run_qwen36_live_tps.py](../qwen36-gfx906/run_qwen36_live_tps.py)
  provide the committed script evidence for deployment, health checks, and the current
  benchmark harness.
- [qwen36-gfx906/media/](../qwen36-gfx906/media/) contains public benchmark media
  referenced by the README.

## Safe To Say Based On Committed Main

- LocalAIServers is an IRS-recognized 501(c)(3) public charity focused on public
  education around locally hosted AI servers.
- The repo maintains open documentation, reproducible local AI deployment notes,
  benchmark reporting, and hardware verification methodology.
- Current committed technical work focuses on GFX906 / AMD Instinct MI50-class
  documentation and a Qwen3.6-35B-A3B runtime profile.
- The repo includes a prebuilt Docker Hub image tag and digest for the current public
  runtime notes.
- The repo documents reference fixed-token single-request TPS results for the documented
  runtime lane.
- The repo supports public education, open-source maintenance, hardware verification
  standards, benchmark reporting, affordable AI research access, and community AI
  research access.

## Should Not Be Said Without More Evidence

- Do not claim official ROCm support, AMD endorsement, or vendor certification.
- Do not generalize benchmark figures beyond the documented model, hardware, command,
  runtime, and workload.
- Do not claim hardware inventory, program capacity, participant counts, releases,
  validation logs, or funding results that are not in committed public files.
- Do not describe unreleased dense source work or unreleased mixed prompt/decode tooling
  as current repo capability.
- Do not imply donations provide hardware, preferential treatment, discounts,
  procurement access, program participation, or other private benefits.
- Do not add legal filing details, tax identifiers, addresses, private payment
  collection details, banking details, participant data, supplier records, or private
  operations records.
- Do not use coordinated-purchasing, commercial procurement, or private-benefit framing.

## Donation And Grant Wording Gaps

Before this pass, the repo had a technical README but did not have a public funding
page, donation boundary page, grant concept note, public impact page, funding
placeholder, or communication templates.

This pass adds:

- [docs/impact.md](impact.md)
- [docs/funding.md](funding.md)
- [docs/program-boundaries.md](program-boundaries.md)
- [grants/README.md](../grants/README.md)
- [grants/concept-note-gfx906-community-ai-access.md](../grants/concept-note-gfx906-community-ai-access.md)
- [.github/FUNDING.yml.example](../.github/FUNDING.yml.example)
- [communications/](../communications/)

Remaining human review: confirm donation-page wording, tax receipt wording, grant budget
wording, and any non-donation program-payment language before public fundraising use.

## Privacy Risks

The original public technical README included private-looking local validation host/path
references. Those were replaced with public descriptions such as "two independent GFX906
hosts" and "clean per-run working directories" while preserving archive hashes, image
digests, benchmark figures, and reproduction commands.

The repo now documents redaction expectations in [README.md](../README.md),
[CONTRIBUTING.md](../CONTRIBUTING.md), [SECURITY.md](../SECURITY.md), issue templates,
and communication templates.

Private operations artifacts are ignored in [.gitignore](../.gitignore), including
runner binaries, private ops directories, participant/payment CSVs, labels, shipping
labels, and QC result directories.

## Overclaim Risks

- "Winner contract" wording was replaced with neutral "reference runtime profile"
  wording in [qwen36-gfx906/README.md](../qwen36-gfx906/README.md) and
  [qwen36-gfx906/manifest.json](../qwen36-gfx906/manifest.json).
- Official support and endorsement disclaimers were added to public-facing docs.
- Benchmark results are now described as scoped reference measurements, not universal
  performance guarantees.
- Future grant and impact language is framed as public documentation, open-source
  maintenance, QC methodology, and community AI research access.

## Missing Repo Credibility Items

This pass adds or updates:

- [LICENSE](../LICENSE) using Apache-2.0 for code.
- [docs/LICENSE-DOCS.md](LICENSE-DOCS.md) as a documentation-license placeholder for
  human review.
- [CONTRIBUTING.md](../CONTRIBUTING.md)
- [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md)
- [SECURITY.md](../SECURITY.md)
- [.github/FUNDING.yml.example](../.github/FUNDING.yml.example)
- [.github/ISSUE_TEMPLATE/](../.github/ISSUE_TEMPLATE/)
- [.github/pull_request_template.md](../.github/pull_request_template.md)
- [issues/](../issues/) contributor backlog.

## Recommended Wording Improvements

- Use "public education around locally hosted AI servers" for mission framing.
- Use "reproducible local AI deployment notes" instead of broad deployment guarantees.
- Use "GFX906 / MI50-class hardware documentation" instead of official support language.
- Use "hardware verification standards" and "QC methodology" for validation work.
- Use "benchmark reporting" and "scoped reference measurements" for performance
  material.
- Use "affordable AI research access" and "community AI research access" for
  public-benefit framing.
- Keep donation language separate from hardware verification, pass-through costs, and
  fulfillment.

## Ignore Rule Confirmation

[.gitignore](../.gitignore) includes the required protections:

- `run_codex_unshare.sh`
- `codex-x86_64-unknown-linux-musl-*`
- `ops/private/`
- `ops/out/`
- `*.private.csv`
- `participants*.csv`
- `payments*.csv`
- private payment-provider CSVs
- `labels*.pdf`
- `shipping-labels/`
- `qc-results/`

The local runner artifacts are intended to remain untracked. Final validation commands
should confirm that `run_codex_unshare.sh` and `codex-x86_64-unknown-linux-musl-*` are
ignored and not tracked.
