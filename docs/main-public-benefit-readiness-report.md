# Main Public-Benefit Readiness Report

Date: 2026-06-09

## Summary

This branch is ready for human PR review as a public-benefit, grant-readiness, donation-readiness, and repo-credibility pass. It rewrites the public README, adds impact/funding/program-boundary/grant documentation, adds community health files, creates public communication templates, adds a contributor issue backlog, documents GitHub settings recommendations, and records public wording/risk reviews.

## Readiness Evaluation

1. README clarity: yes. [README.md](../README.md) now explains the mission, current technical focus, repo contents, public benefit, funding boundaries, program boundaries, and safety limitations.
2. Technical reproduction content preserved: yes. Current commands, Docker image tag, digest, archive hash, vLLM command, and reference TPS figures remain visible.
3. Donation/grant support: yes. Added [docs/impact.md](impact.md), [docs/funding.md](funding.md), [docs/program-boundaries.md](program-boundaries.md), and grant concept material.
4. Donation/program-payment boundaries: yes. README and boundary docs state that charitable donations are separate from hardware verification, pass-through costs, and fulfillment.
5. Private/legal/payment/participant data: no high/medium remaining findings after remediation. Real payment links, addresses, filing documents, tax identifiers, participant data, and vendor quotes were not added.
6. Unreleased dense source claims: none added.
7. Group-buy/resale/private-benefit language: no promotional or benefit-offer language added. Risk terms appear only in avoidance/review contexts where needed.
8. Local runner artifacts ignored and untracked: yes, subject to final command confirmation.
9. GitHub settings recommendations: ready in [docs/github-repo-settings.md](github-repo-settings.md).
10. PR readiness: ready for human review before merge.
11. Push readiness: safe to push after final validation; no active placeholder `.github/FUNDING.yml` exists.
12. Merge/tag/release status: no merge, tag, or GitHub release was performed.

## Files Changed

Modified:

- [.gitignore](../.gitignore)
- [README.md](../README.md)
- [qwen36-gfx906/README.md](../qwen36-gfx906/README.md)
- [qwen36-gfx906/manifest.json](../qwen36-gfx906/manifest.json)

Created:

- [LICENSE](../LICENSE)
- [CONTRIBUTING.md](../CONTRIBUTING.md)
- [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md)
- [SECURITY.md](../SECURITY.md)
- [.github/FUNDING.yml.example](../.github/FUNDING.yml.example)
- [.github/ISSUE_TEMPLATE/bug_report.yml](../.github/ISSUE_TEMPLATE/bug_report.yml)
- [.github/ISSUE_TEMPLATE/documentation_issue.yml](../.github/ISSUE_TEMPLATE/documentation_issue.yml)
- [.github/ISSUE_TEMPLATE/benchmark_report.yml](../.github/ISSUE_TEMPLATE/benchmark_report.yml)
- [.github/pull_request_template.md](../.github/pull_request_template.md)
- [docs/index.md](index.md)
- [docs/impact.md](impact.md)
- [docs/funding.md](funding.md)
- [docs/program-boundaries.md](program-boundaries.md)
- [docs/github-repo-settings.md](github-repo-settings.md)
- [docs/main-branch-wording-audit.md](main-branch-wording-audit.md)
- [docs/main-public-risk-review.md](main-public-risk-review.md)
- [docs/main-public-benefit-readiness-report.md](main-public-benefit-readiness-report.md)
- [docs/LICENSE-DOCS.md](LICENSE-DOCS.md)
- [grants/README.md](../grants/README.md)
- [grants/concept-note-gfx906-community-ai-access.md](../grants/concept-note-gfx906-community-ai-access.md)
- [communications/README.md](../communications/README.md)
- [communications/donation-announcement-template.md](../communications/donation-announcement-template.md)
- [communications/github-project-update-template.md](../communications/github-project-update-template.md)
- [communications/subreddit-technical-update-template.md](../communications/subreddit-technical-update-template.md)
- [issues/README.md](../issues/README.md)
- [issues/001-docs-gfx906-background.md](../issues/001-docs-gfx906-background.md)
- [issues/002-benchmark-report-template-improvements.md](../issues/002-benchmark-report-template-improvements.md)
- [issues/003-qwen36-gfx906-reproducibility-notes.md](../issues/003-qwen36-gfx906-reproducibility-notes.md)
- [issues/004-mi50-hardware-verification-methodology.md](../issues/004-mi50-hardware-verification-methodology.md)
- [issues/005-docs-cleanup-good-first-issue.md](../issues/005-docs-cleanup-good-first-issue.md)
- [issues/006-github-pages-doc-index.md](../issues/006-github-pages-doc-index.md)
- [issues/007-funding-doc-review.md](../issues/007-funding-doc-review.md)

## Checks Run

- `git status --short`: shows the intended modified tracked files plus new public docs/community-health directories. `.tmp/` no longer appears because it is ignored.
- `git diff --stat`: tracked-file diff is 4 files changed, 131 insertions, 33 deletions before staging. New untracked files are listed by `git status --short` until staged.
- `git check-ignore -v run_codex_unshare.sh || true`: matched `.gitignore`.
- `git check-ignore -v codex-x86_64-unknown-linux-musl-test || true`: matched `.gitignore`.
- `git ls-files | grep -E '(^|/)run_codex_unshare\.sh$|(^|/)codex-x86_64-unknown-linux-musl-' || true`: no output; runner artifacts are not tracked.
- `git ls-files | grep -E '(^|/)participants.*\.csv$|(^|/)payments.*\.csv$|(^|/)wise.*\.csv$|(^|/)labels.*\.pdf$|(^|/)shipping-labels/|(^|/)qc-results/|(^|/)\.tmp/' || true`: no output; private ops artifacts are not tracked.
- `python3 -m json.tool qwen36-gfx906/manifest.json >/dev/null`: passed.
- `find . -path './.git' -prune -o -path './.tmp' -prune -o -name "*.sh" -print0 | xargs -0 -r bash -n`: passed with no output.
- `find . -path './.git' -prune -o -path './.tmp' -prune -o -name "*.py" -print0 | xargs -0 -r python3 -m py_compile`: passed with no output.
- `git diff --check`: passed with no output.
- Risk-term grep requested in the task: reviewed. Remaining public branch-relevant matches are avoidance/boundary language, ignore rules, Apache-2.0 license text, and benign technical substrings such as `einops`, `reinstall`, and `otherwise`. The exact grep also reports binary matches from an ignored local runner binary and the tracked benchmark preview image; neither is a tracked private data finding.

## Remaining Human Review Items

- Replace the donation placeholder only after donation-page review.
- Review [docs/LICENSE-DOCS.md](LICENSE-DOCS.md) and replace with a finalized documentation license notice if desired.
- Confirm funding, donation receipt, and program-payment wording with appropriate human advisors.
- Decide whether the existing tracked benchmark video should remain in git or move to release assets in a future maintenance pass.

## High/Medium Risks Fixed

- Private validation host/path references were removed from public technical docs.
- Overclaim-prone "winner contract" wording was replaced with neutral runtime-profile language.
- Internal source-run manifest metadata was replaced with public main-branch wording.
- Active placeholder GitHub funding config was disabled by moving it to `.github/FUNDING.yml.example`.
- Donation/program-payment boundaries were added across README and funding docs.

## PR Readiness Recommendation

The branch is PR-ready and safe to push after final validation. Do not squash in private operations artifacts, payment details, participant data, legal filing documents, or local runner files. No merge, tag, or GitHub release was performed.
