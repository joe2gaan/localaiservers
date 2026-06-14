# Contributing

Contributions should improve public documentation, reproducibility, benchmark clarity,
repository hygiene, source-level GFX906 notes, or hardware verification methodology.

## Good Contribution Areas

- Clarify GFX906 / MI50-class deployment notes.
- Improve reproducibility reports.
- Improve benchmark reports and benchmark methodology.
- Add redaction guidance for logs and screenshots.
- Improve hardware verification and QC methodology.
- Report source-level GFX906 runtime, kernel, graph, or collective communication issues.
- Fix broken links, typos, and stale instructions.
- Add evidence-backed notes for committed scripts or published images.

## Evidence Standard

Do not add technical claims that are not evidenced in committed files, public image
identities, reproducible commands, benchmark output, source records, or clearly labeled
future plans. Do not claim official ROCm support or AMD endorsement.

## Privacy Rules

Do not commit or post:

- Secrets, tokens, private keys, or credentials.
- Participant data, payment data, banking details, shipping records, or labels.
- Private hostnames, private IPs, home-directory paths, or private logs.
- Legal filing documents, filing screenshots, addresses, or tax identifiers.
- Supplier pricing, supplier records, private procurement notes, or private operational
  records.
- Coordinated purchasing, resale, or private-benefit access content.

Redact logs and screenshots before attaching them to issues or pull requests.

## Benchmark Reports

Benchmark reports should include:

- Model.
- Hardware.
- Runtime image or build.
- Command used.
- Context length.
- Backend decode performance and client wall-time performance as separate metrics.
- VRAM usage.
- Temperature or power notes when available.
- Redacted logs or screenshots.

## Source-Level GFX906 Reports

Source-level reports should identify the affected runtime path, kernel or graph area,
reproduction context, evidence type, and whether the result is promoted, rejected,
diagnostic, a source milestone, or an active lane.

## Local Checks

Run the relevant checks before submitting:

```bash
find . -path './.git' -prune -o -path './.tmp' -prune -o -name "*.sh" -print0 | xargs -0 -r bash -n
find . -path './.git' -prune -o -path './.tmp' -prune -o -name "*.py" -print0 | xargs -0 -r python3 -m py_compile
```

Do not run GPU workloads, Docker builds, ROCm builds, or model downloads unless the
issue or pull request specifically requires them and you have the right local
environment.
