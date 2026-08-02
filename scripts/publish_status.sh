#!/usr/bin/env bash
set -euo pipefail

context="${1:?usage: publish_status.sh CONTEXT STATUS [COMMIT]}"
job_status="${2:?usage: publish_status.sh CONTEXT STATUS [COMMIT]}"
commit="${3:-$GITHUB_SHA}"

case "$job_status" in
  pending | queued | in_progress)
    state=pending
    ;;
  success)
    state=success
    ;;
  cancelled)
    state=error
    ;;
  *)
    state=failure
    ;;
esac

description="$context: $job_status"
target_url="$GITHUB_SERVER_URL/$GITHUB_REPOSITORY/actions/runs/$GITHUB_RUN_ID"

gh api --method POST \
  "repos/$GITHUB_REPOSITORY/statuses/$commit" \
  -f state="$state" \
  -f context="$context" \
  -f description="$description" \
  -f target_url="$target_url" \
  >/dev/null
