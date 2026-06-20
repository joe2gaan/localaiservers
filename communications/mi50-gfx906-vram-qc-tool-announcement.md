# MI50 / GFX906 VRAM QC Field-Check Tool Added

LocalAIServers has added a maintained MI50 / GFX906 VRAM QC field-check tool to GitHub.

Tool location:

https://github.com/joe2gaan/localaiservers/tree/main/tools/gfx906-mi50-vram-qc

The tool helps users check ROCm/HIP visibility, kernel-reported VRAM totals,
GFX906/Vega20-style evidence where available, and whether selected devices can allocate
and verify a large VRAM region.

This is part of LocalAIServers' public hardware-verification and QC methodology for
affordable local AI systems.

Important boundaries:

- This is educational QC methodology.
- It is not certification.
- It is not a warranty.
- It is not procurement or resale support.
- It is not a hardware discount, allocation, or subsidy program.
- It does not flash BIOS or VBIOS.
- It does not modify hardware state.
- Passing the check does not guarantee AI workload performance.

The goal is to help people learn how to inspect and reason about affordable local AI
hardware before relying on product-name strings, resale claims, or benchmark claims.

Canonical project links:

- Website: https://localaiservers.com
- GitHub: https://github.com/joe2gaan/localaiservers
- GitHub Releases: https://github.com/joe2gaan/localaiservers/releases
- Hardware verification standards: https://github.com/joe2gaan/localaiservers/blob/main/docs/hardware-verification-standards.md
- QC methodology: https://github.com/joe2gaan/localaiservers/blob/main/docs/qc-methodology.md

Do not post this externally. It is a draft announcement for owner review.
