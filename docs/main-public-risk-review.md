# Main Public Risk Review

Date: 2026-06-09

## Summary

This review checks the public branch for private/legal/payment/participant data, unsafe
donation language, unsupported technical claims, official-support overclaims, and local
runner artifacts. It covers tracked files and the new public-benefit documentation added
in this pass. It does not run GPU workloads, Docker builds, ROCm builds, model
downloads, or long-running validation.

No tax identifier, filing detail, personal address, banking detail, real payment
collection detail, participant dataset, supplier record, live pricing, or tracked local
Codex runner binary was found in the committed branch review.

## High Severity Findings

1. Private validation host and path references in public technical docs.

   Initial review found local network host references and local working paths in
   [qwen36-gfx906/README.md](../qwen36-gfx906/README.md). These were not necessary for
   public reproducibility because the relevant public evidence is the archive hash,
   image digest, command, and validation outcome.

   Status: fixed.

## Medium Severity Findings

1. Overclaim-prone "winner contract" wording.

   [qwen36-gfx906/README.md](../qwen36-gfx906/README.md) and
   [qwen36-gfx906/manifest.json](../qwen36-gfx906/manifest.json) used internal-sounding
   "winner contract" language. This could read as a broader claim than the repo evidence
   supports.

   Status: fixed.

2. Internal source-run naming in manifest metadata.

   [qwen36-gfx906/manifest.json](../qwen36-gfx906/manifest.json) included an internal
   source-run path. It was not needed for public use and could expose internal workflow
   naming.

   Status: fixed.

3. Donation and program-boundary gaps.

   The original README did not clearly separate charitable donations from hardware
   verification, pass-through costs, or fulfillment. That gap could create
   donor-readiness and private-benefit ambiguity.

   Status: fixed with README funding text, [docs/funding.md](funding.md), and
   [docs/program-boundaries.md](program-boundaries.md).

## Low Severity Findings

1. Third-party attribution comments in generated script content.

   A broad scan found public upstream attribution email addresses embedded in generated
   source comments inside [qwen36-gfx906/deploy.sh](../qwen36-gfx906/deploy.sh). These
   appear to be third-party public attribution comments, not LocalAIServers private
   contact data.

   Status: no action taken.

2. Tracked media artifact.

   [qwen36-gfx906/media/qwen36_ref20_machiavelli_100_to_1000_tps_720p.mp4](../qwen36-gfx906/media/qwen36_ref20_machiavelli_100_to_1000_tps_720p.mp4)
   is a tracked public benchmark media artifact. It is not a privacy finding, but it is
   large and should remain intentional.

   Status: no action taken.

3. Placeholder funding URL.

   The repo intentionally uses `https://REPLACE-WITH-DONATION-PAGE` in public funding
   documentation and in [.github/FUNDING.yml.example](../.github/FUNDING.yml.example).
   The active `.github/FUNDING.yml` file was removed so GitHub does not expose an active
   placeholder donation button.

   Status: fixed for active GitHub funding config; human review still required before
   replacing the placeholder.

4. Generated MoE config identifier.

   The committed deploy script uses a generated MoE config directory name that includes
   an internal-looking suffix. This is a relative generated runtime path, not a network
   hostname, private IP, username, or home-directory path. It is retained in the
   committed deploy script and technical README to preserve exact reproduction context
   and avoid invalidating the documented `deploy.sh` SHA-256 evidence.

   Status: no action taken.

## Recommended Fixes

- Replace private validation labels with public descriptions and keep hashes/digests as
  evidence.
- Replace internal "winner" language with neutral runtime-profile wording.
- Keep benchmark claims scoped to documented model, hardware, runtime, command, and
  workload.
- Keep official support and endorsement disclaimers in public-facing docs.
- Keep charitable donations separate from hardware verification, pass-through costs, and
  fulfillment.
- Keep private operations, participant, payment, label, and QC artifacts ignored.
- Keep active `.github/FUNDING.yml` absent until a reviewed public donation page exists.

## Release Blockers

No release blockers remain after remediation, assuming human review accepts the
Apache-2.0 code license, documentation-license placeholder, funding example, and
donation/program-payment wording.

## Non-Blocking Cleanup

- Human legal/accounting review of [docs/funding.md](funding.md),
  [docs/program-boundaries.md](program-boundaries.md), and
  [docs/LICENSE-DOCS.md](LICENSE-DOCS.md).
- Review whether GitHub Pages should expose [docs/index.md](index.md) directly.
- Consider whether the tracked benchmark video should remain in the git repository or
  move to release assets in a future maintenance pass.

## Remediation Performed

- Added mandatory [.gitignore](../.gitignore) protections for local runners, private ops
  directories, participant/payment CSVs, labels, shipping labels, and QC results.
- Moved the placeholder GitHub funding config to
  [.github/FUNDING.yml.example](../.github/FUNDING.yml.example) and removed active
  `.github/FUNDING.yml`.
- Rewrote [README.md](../README.md) as a public-benefit landing page with current
  technical focus, funding boundaries, program boundaries, public-benefit framing,
  benchmark limitations, and safety language.
- Redacted private validation host/path references in
  [qwen36-gfx906/README.md](../qwen36-gfx906/README.md).
- Replaced "winner contract" wording with "reference runtime profile" wording in
  [qwen36-gfx906/README.md](../qwen36-gfx906/README.md) and
  [qwen36-gfx906/manifest.json](../qwen36-gfx906/manifest.json).
- Replaced internal source-run metadata in
  [qwen36-gfx906/manifest.json](../qwen36-gfx906/manifest.json) with public main-branch
  wording.
- Added impact, funding, program-boundary, grant, communication, GitHub settings,
  community health, and contributor backlog documentation.
