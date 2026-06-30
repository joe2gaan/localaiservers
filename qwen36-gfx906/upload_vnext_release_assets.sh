#!/bin/sh
set -eu

usage() {
  cat >&2 <<'EOF'
Usage:
  upload_vnext_release_assets.sh <release-tag> <release-asset-dir>

Dry-run by default. Verifies the vNext 69-file release asset bundle and prints
the exact upload plan for an existing GitHub Release.

Set UPLOAD_VNEXT_RELEASE_ASSETS=1 and GH_RELEASE_REPO=owner/repo to upload
with gh to a repository you control. Uploading to joe2gaan/localaiservers is
maintainer-only and also requires ALLOW_LOCALAISERVERS_RELEASE_UPLOAD=1.
This script does not create a tag, create a release, edit release notes,
delete assets, or use --clobber.

Environment:
  UPLOAD_VNEXT_RELEASE_ASSETS=1  Actually run gh release upload. Default: dry-run.
  GH_RELEASE_REPO=owner/repo     Required for real upload. Use your own repo
                                unless you are a LocalAIServers maintainer.
  ALLOW_LOCALAISERVERS_RELEASE_UPLOAD=1
                                Maintainer-only guard for joe2gaan/localaiservers.
  GH=gh                         GitHub CLI command. Default: gh.
  VERIFY_PUBLIC_URLS_AFTER_UPLOAD=0
                                Skip post-upload URL verification. Default: 1.
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

[ "$#" -eq 2 ] || {
  usage
  exit 2
}

release_tag=$1
asset_dir=$2
gh_bin=${GH:-gh}
upload=${UPLOAD_VNEXT_RELEASE_ASSETS:-0}
verify_after=${VERIFY_PUBLIC_URLS_AFTER_UPLOAD:-1}
release_repo=${GH_RELEASE_REPO:-}
allow_localaiservers_upload=${ALLOW_LOCALAISERVERS_RELEASE_UPLOAD:-0}

case "$release_tag" in
  ''|'<release-tag>'|*'<'*|*'>'*|*' '*)
    die "release tag must be the final published tag, not a placeholder"
    ;;
esac
case "$upload" in
  0|1) ;;
  *) die "UPLOAD_VNEXT_RELEASE_ASSETS must be 0 or 1" ;;
esac
case "$verify_after" in
  0|1) ;;
  *) die "VERIFY_PUBLIC_URLS_AFTER_UPLOAD must be 0 or 1" ;;
esac
case "$allow_localaiservers_upload" in
  0|1) ;;
  *) die "ALLOW_LOCALAISERVERS_RELEASE_UPLOAD must be 0 or 1" ;;
esac

if [ "$upload" = "1" ]; then
  [ -n "$release_repo" ] ||
    die "real upload requires GH_RELEASE_REPO=owner/repo; use your own repo unless you are a LocalAIServers maintainer"
  case "$release_repo" in
    */*) ;;
    *) die "GH_RELEASE_REPO must be owner/repo" ;;
  esac
  if [ "$release_repo" = "joe2gaan/localaiservers" ] &&
     [ "$allow_localaiservers_upload" != "1" ]; then
    die "uploading to joe2gaan/localaiservers is maintainer-only; set ALLOW_LOCALAISERVERS_RELEASE_UPLOAD=1 only for approved maintainer publication"
  fi
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
tmp_uploads=${TMPDIR:-/tmp}/vnext-upload-list.$$
trap 'rm -f "$tmp_uploads"' EXIT HUP INT TERM

"$script_dir/verify_vnext_release_asset_bundle.sh" "$asset_dir" >/dev/null
"$script_dir/list_vnext_release_asset_uploads.sh" "$asset_dir" >"$tmp_uploads"

upload_count=$(wc -l <"$tmp_uploads" | awk '{print $1}')
test "$upload_count" -eq 69 ||
  die "upload list count mismatch: expected 69 got $upload_count"

if [ "$upload" != "1" ]; then
  printf 'dry-run: vNext release asset bundle verified for tag: %s\n' "$release_tag"
  printf 'dry-run: upload file count: %s\n' "$upload_count"
  printf 'dry-run: to upload to your own GitHub Release after creating the tag/release in your repo, run:\n'
  printf '  GH_RELEASE_REPO=owner/repo UPLOAD_VNEXT_RELEASE_ASSETS=1 %s %s %s\n' "$0" "$release_tag" "$asset_dir"
  printf 'dry-run: LocalAIServers publication is maintainer-only and requires GH_RELEASE_REPO=joe2gaan/localaiservers plus ALLOW_LOCALAISERVERS_RELEASE_UPLOAD=1.\n'
  printf 'dry-run: upload list follows:\n'
  cat "$tmp_uploads"
  exit 0
fi

command -v "$gh_bin" >/dev/null 2>&1 || die "gh command not found: $gh_bin"
"$gh_bin" auth status >/dev/null 2>&1 ||
  die "gh is not authenticated"
"$gh_bin" release view "$release_tag" --repo "$release_repo" >/dev/null ||
  die "GitHub Release does not exist for tag: $release_tag"

while IFS= read -r asset_path; do
  test -f "$asset_path" || die "missing upload asset: $asset_path"
  "$gh_bin" release upload "$release_tag" "$asset_path" --repo "$release_repo"
done <"$tmp_uploads"

printf 'vNext release assets uploaded: repo=%s tag=%s files=%s\n' "$release_repo" "$release_tag" "$upload_count"

if [ "$verify_after" = "1" ]; then
  RELEASE_ASSET_BASE="https://github.com/$release_repo/releases/download/$release_tag" \
    "$script_dir/verify_vnext_release_asset_urls.sh" "$release_tag"
fi
