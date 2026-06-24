#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/workspace-containing-Data-preprocessing" >&2
  exit 1
fi

workspace_root="$1"
manifest_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="${manifest_dir}/inputs/raw_preprocessing_inputs.tsv"

if [[ ! -f "$manifest" ]]; then
  echo "Missing manifest: $manifest" >&2
  exit 1
fi

tail -n +2 "$manifest" | while IFS=$'\t' read -r rel_path url size_bytes sha256; do
  dest="${workspace_root}/${rel_path}"
  mkdir -p "$(dirname "$dest")"

  if [[ -f "$dest" ]]; then
    actual_sha="$(sha256sum "$dest" | awk '{print $1}')"
    if [[ "$actual_sha" == "$sha256" ]]; then
      echo "OK existing: $rel_path"
      continue
    fi
    echo "Checksum mismatch for existing file, re-downloading: $rel_path" >&2
    rm -f "$dest"
  fi

  echo "Downloading: $rel_path"
  curl -L --fail --continue-at - --output "$dest" "$url"

  actual_size="$(stat -c '%s' "$dest")"
  if [[ "$actual_size" != "$size_bytes" ]]; then
    echo "Size mismatch for $rel_path: expected $size_bytes, got $actual_size" >&2
    exit 1
  fi

  actual_sha="$(sha256sum "$dest" | awk '{print $1}')"
  if [[ "$actual_sha" != "$sha256" ]]; then
    echo "Checksum mismatch for $rel_path" >&2
    echo "Expected: $sha256" >&2
    echo "Actual:   $actual_sha" >&2
    exit 1
  fi

  gzip -t "$dest"
  echo "OK downloaded: $rel_path"
done
