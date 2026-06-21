# Title

Document MI50 hardware verification methodology

## Context

The deployment notes include GPU count, VRAM, and free-VRAM gates. A separate
methodology page could explain the public QC standard for verifying MI50-class local AI
server readiness.

## Desired Outcome

Create a documentation page that describes public hardware verification standards for
MI50-class systems, including GPU identity, VRAM, free-VRAM, smoke tests, and redacted
evidence expectations.

## Acceptance Criteria

- Documents GPU identity and VRAM checks.
- Documents free-VRAM checks before expensive model launches.
- Documents smoke-test expectations.
- Explains how to share redacted logs/screenshots.
- Avoids publishing inventory, participant, payment, logistics, or private operations
  data.

## Suggested Labels

`documentation`, `hardware-verification`, `qc`, `mi50`

## Difficulty

Intermediate

## Privacy Reminder

Hardware verification logs can expose hostnames, paths, serial-like data, and local
network details. Redact before sharing.
