# Model Format Probe

`model-format-probe` is a small binary classifier for the vNext reproduction
launcher. It inspects model bytes instead of trusting filenames.

Detected formats:

- `gguf`: file starts with `GGUF`.
- `hf_safetensors`: safetensors header length is sane and the JSON header
  contains tensor entries with `dtype`, `shape`, and `data_offsets`.
- `hf_torch_legacy`: PyTorch zip or pickle-like legacy weight file.
- `unknown`: no known weight signature was found.

Directory inputs are scanned recursively to a bounded depth. If more than one
model format is detected under a directory, the probe fails closed with
`confidence=conflict`.
