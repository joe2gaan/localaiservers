#!/bin/sh
set -eu

usage() {
    cat >&2 <<'EOF'
usage:
  hash_vnext_patch_bundle.sh create <bundle-dir>
  hash_vnext_patch_bundle.sh verify <bundle-dir> <expected-manifest-sha256>

The create mode writes <bundle-dir>/sha256sum.txt with deterministic relative
paths and prints the sha256 of that manifest. The manifest excludes itself.
EOF
}

[ "$#" -ge 2 ] || {
    usage
    exit 2
}

mode=$1
bundle_dir=$2
manifest="${bundle_dir}/sha256sum.txt"

[ -d "$bundle_dir" ] || {
    printf '%s\n' "error: bundle directory not found: $bundle_dir" >&2
    exit 2
}

case "$mode" in
    create)
        tmp="${manifest}.tmp.$$"
        (
            cd "$bundle_dir"
            find . -type f ! -name sha256sum.txt ! -name 'sha256sum.txt.tmp.*' | LC_ALL=C sort |
            while IFS= read -r file; do
                rel=${file#./}
                sha256sum "$rel"
            done
        ) >"$tmp"
        mv "$tmp" "$manifest"
        sha256sum "$manifest" | awk '{print $1}'
        ;;
    verify)
        [ "$#" -eq 3 ] || {
            usage
            exit 2
        }
        expected=$3
        [ -f "$manifest" ] || {
            printf '%s\n' "error: manifest missing: $manifest" >&2
            exit 2
        }
        actual=$(sha256sum "$manifest" | awk '{print $1}')
        [ "$actual" = "$expected" ] || {
            printf '%s\n' "error: manifest hash mismatch: actual=$actual expected=$expected" >&2
            exit 2
        }
        (
            cd "$bundle_dir"
            sha256sum -c sha256sum.txt
        )
        ;;
    *)
        usage
        exit 2
        ;;
esac
