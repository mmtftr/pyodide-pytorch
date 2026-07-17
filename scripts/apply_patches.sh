#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="${1:?usage: apply_patches.sh PYTORCH_SOURCE_DIR}"
expected_ref="$(python3 "$repo_root/scripts/config.py" get PYTORCH_REF)"

actual_ref="$(git -C "$source_dir" rev-parse HEAD)"
if [[ "$actual_ref" != "$expected_ref" ]]; then
  echo "expected PyTorch $expected_ref, found $actual_ref" >&2
  exit 1
fi

if ! git -C "$source_dir" diff --quiet --ignore-submodules=dirty; then
  echo "refusing to patch a modified PyTorch worktree" >&2
  exit 1
fi

mapfile -t patches < <(find "$repo_root/patches/pytorch" -maxdepth 1 -type f -name '*.patch' -print | LC_ALL=C sort)
if [[ "${#patches[@]}" -eq 0 ]]; then
  echo "no PyTorch patches found" >&2
  exit 1
fi

for patch in "${patches[@]}"; do
  echo "Checking $(basename "$patch")"
  git -C "$source_dir" apply --check "$patch"
done

for patch in "${patches[@]}"; do
  echo "Applying $(basename "$patch")"
  git -C "$source_dir" apply "$patch"
done

git -C "$source_dir" diff --check
git -C "$source_dir" status --short
