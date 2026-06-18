#!/bin/bash

# automatically gets the version from pyproject.toml and sets it as a git tag and pushes it

DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

git fetch

version=$(grep -oP '(?<=^version = ")[^"]*' ../pyproject.toml)
branch=$(git rev-parse --abbrev-ref HEAD)

# verify version is valid
if [[ ! $version =~ ^[0-9]+\.[0-9]+\.[0-9]+(a[0-9]*)?$ ]]; then
  echo "Version is invalid: $version"
  exit 1
fi

echo "Branch: $branch"
echo "Version: v$version"
previous_tag=$(git describe --tags --abbrev=0 2>/dev/null || true)
echo "Previous tag: $previous_tag"

# check if the tag already exists locally
if git rev-parse "v$version" >/dev/null 2>&1; then
  echo "Tag v$version already exists locally."
  exit 1
fi

if [[ "$branch" != "main" && "$branch" != "master" ]]; then
  read -r -p "You are not on main/master (current branch: '$branch'). Are you sure you want to tag from this branch? [y/N] " confirm
  if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
  fi
fi

if [[ "$DRY_RUN" == true ]]; then
  echo "Dry run: would create tag v$version and push to branch $branch"
  exit 0
fi

git push --follow-tags && gh release create "v$version" --target "$branch" --generate-notes --fail-on-no-commits