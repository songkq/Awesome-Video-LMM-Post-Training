#!/bin/bash
set -euo pipefail

git fetch origin
git pull --rebase --autostash origin main

python get_readme.py

git add .

git status --porcelain
git diff --staged --name-only

git commit -m "update" || true

git push origin main