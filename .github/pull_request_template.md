## Summary

Describe the public documentation, reproducibility, benchmark, QC, or source-level
GFX906 change.

## Evidence

Link to committed files, commands, image tags/digests, benchmark output, source records,
or issue context that supports the change.

## Public-Benefit Check

- [ ] Improves public documentation, reproducibility, benchmark clarity, QC methodology,
      hardware verification, or source-level GFX906 understanding.
- [ ] Does not imply direct public machine access or public cloud hosting.
- [ ] Does not claim official ROCm support or AMD endorsement.
- [ ] Does not add unsupported benchmark, hardware inventory, funding, or partnership
      claims.

## Privacy Check

- [ ] No secrets, tokens, private keys, or credentials.
- [ ] No participant, payment, shipping, supplier, or private operational data.
- [ ] No private hostnames, private IPs, home-directory paths, or private logs.
- [ ] No legal filing documents, filing screenshots, addresses, or tax identifiers.
- [ ] No donation wording that implies hardware, discounts, procurement access,
      preferential treatment, or private benefits.

## Checks Run

```bash
find . -path './.git' -prune -o -path './.tmp' -prune -o -name "*.sh" -print0 | xargs -0 -r bash -n
find . -path './.git' -prune -o -path './.tmp' -prune -o -name "*.py" -print0 | xargs -0 -r python3 -m py_compile
```
