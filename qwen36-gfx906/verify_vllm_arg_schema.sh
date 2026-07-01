#!/bin/sh
set -eu

usage() {
    cat >&2 <<'EOF'
usage: verify_vllm_arg_schema.sh <vnext-launch-run-dir> <vllm-serve-help-file>

Checks the generated vLLM argv from vnext_repro_launcher.sh against a captured
`vllm serve --help` file from the target runtime image. This catches flag drift
before a benchmark run starts.
EOF
}

die() {
    printf '%s\n' "error: $*" >&2
    exit 2
}

[ "$#" -eq 2 ] || {
    usage
    exit 2
}

run_dir=$1
help_file=$2
argv_file="${run_dir}/vllm_argv.txt"

[ -f "$argv_file" ] || die "missing generated argv file: $argv_file"
[ -f "$help_file" ] || die "missing vLLM help file: $help_file"

tmp="${TMPDIR:-/tmp}/vnext-vllm-args.$$"
trap 'rm -f "$tmp"' EXIT HUP INT TERM

sed -n 's/^\(--[A-Za-z0-9][A-Za-z0-9-]*\)\(=.*\)\{0,1\}$/\1/p' "$argv_file" |
    LC_ALL=C sort -u >"$tmp"

while IFS= read -r opt; do
    [ -n "$opt" ] || continue
    if ! grep -Eq "(^|[[:space:],])${opt}([[:space:],=]|$)" "$help_file"; then
        die "generated vLLM option not present in runtime help: $opt"
    fi
done <"$tmp"

printf '%s\n' "vllm_arg_schema=valid"
printf 'run_dir=%s\n' "$run_dir"
printf 'help_file=%s\n' "$help_file"
