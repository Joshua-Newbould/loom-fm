#!/usr/bin/env bash
# deploy.sh - publish site/ to the gh-pages branch with cache-busting asset URLs.
# usage: GH_TOKEN=... bash tools/deploy.sh   (without GH_TOKEN, git's own credentials are used)
set -eu
cd "$(dirname "$0")/.."
V=$(git rev-parse --short HEAD)
git branch -D gh-pages >/dev/null 2>&1 || true
git subtree split --prefix site -b gh-pages -q
rm -rf /tmp/loom-fm-ghp; git worktree prune
git worktree add -q /tmp/loom-fm-ghp gh-pages
sed -i -E "s#(href|src)=\"(style\.css|main\.js)\"#\1=\"\2?v=$V\"#g" /tmp/loom-fm-ghp/index.html
git -C /tmp/loom-fm-ghp -c user.name=deploy -c user.email=deploy@local commit -qam "cache-bust $V"
git worktree remove --force /tmp/loom-fm-ghp
if [ -n "${GH_TOKEN:-}" ]; then
  B=$(printf 'x-access-token:%s' "$GH_TOKEN" | base64 -w0)
  git -c http.extraheader="Authorization: Basic $B" push -f origin main gh-pages
else
  git push -f origin main gh-pages
fi
echo "published $V"
