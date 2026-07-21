#!/bin/bash

# Bumps the version in pyproject.toml (patch/minor/major), commits it, then
# creates a git tag and GitHub release from it.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYPROJECT="$SCRIPT_DIR/../pyproject.toml"

DRY_RUN=false
BUMP_TYPE=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --patch|--minor|--major)
      if [[ -n "$BUMP_TYPE" ]]; then
        echo "Cannot combine multiple bump flags (already got --$BUMP_TYPE, also got $1)."
        exit 1
      fi
      BUMP_TYPE="${1#--}"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 --patch|--minor|--major [--dry-run]"
      exit 1
      ;;
  esac
done

if [[ -z "$BUMP_TYPE" ]]; then
  echo "Usage: $0 --patch|--minor|--major [--dry-run]"
  exit 1
fi

git fetch

branch=$(git rev-parse --abbrev-ref HEAD)

current_version=$(grep -oP '(?<=^version = ")[^"]*' "$PYPROJECT")

# verify current version is valid
if [[ ! $current_version =~ ^[0-9]+\.[0-9]+\.[0-9]+(a[0-9]*)?$ ]]; then
  echo "Version in pyproject.toml is invalid: $current_version"
  exit 1
fi

# pre-releases (aN suffix) are ambiguous to auto-bump. Leave those to a manual edit.
if [[ $current_version =~ a[0-9]*$ ]]; then
  echo "Current version '$current_version' has an alpha suffix, bump it manually in pyproject.toml, this script only auto-bumps final releases."
  exit 1
fi

IFS='.' read -r major minor patch <<< "$current_version"
case "$BUMP_TYPE" in
  major)
    major=$((major + 1)); minor=0; patch=0
    ;;
  minor)
    minor=$((minor + 1)); patch=0
    ;;
  patch)
    patch=$((patch + 1))
    ;;
esac
version="$major.$minor.$patch"

echo "Branch: $branch"
echo "Current version: $current_version"
echo "Bumping ($BUMP_TYPE) to: v$version"
previous_tag=$(git describe --tags --abbrev=0 2>/dev/null || true)
echo "Previous tag: $previous_tag"

# check if the tag already exists locally
if git rev-parse "v$version" >/dev/null 2>&1; then
  echo "Tag v$version already exists locally."
  exit 1
fi

if [[ "$branch" != "main" && "$branch" != "master" ]]; then
  read -r -p "You are not on main/master (current branch: '$branch'). Are you sure you want to tag from this branch? [y/N] " confirm || true
  if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
  fi
fi

if [[ "$DRY_RUN" == true ]]; then
  echo "Dry run: would bump pyproject.toml to $version, commit it, create tag v$version, and push to branch $branch"
  exit 0
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Working tree is not clean. Commit or stash your changes before releasing."
  exit 1
fi

sed -i "s/^version = \".*\"/version = \"$version\"/" "$PYPROJECT"
git add "$PYPROJECT"
git commit -m "bump version to $version"

git push --follow-tags && gh release create "v$version" --target "$branch" --generate-notes --fail-on-no-commits
