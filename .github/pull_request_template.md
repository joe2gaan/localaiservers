## Summary

Describe the public documentation, reproducibility, benchmark, or code change.

## Evidence

Link to committed files, commands, image tags/digests, benchmark output, or issue
context that supports the change.

## Privacy and Public-Benefit Check

- [ ] No secrets, tokens, private keys, or credentials.
- [ ] No participant, payment, shipping, vendor, or private operational data.
- [ ] No private hostnames, private IPs, home-directory paths, or private logs.
- [ ] No legal filing documents, filing screenshots, addresses, or tax identifiers.
- [ ] No donation wording that implies hardware, access, discounts, program
      participation, or private benefits.
- [ ] No unsupported claims of official ROCm support, AMD endorsement, benchmark
      generality, or unreleased work.

## Checks Run

```bash
find . -name "*.sh" -not -path "./.git/*" -print0 | xargs -0 -r bash -n
find . -name "*.py" -not -path "./.git/*" -print0 | xargs -0 -r python3 -m py_compile
```
