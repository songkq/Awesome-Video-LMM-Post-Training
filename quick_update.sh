#!/bin/bash
python get_readme.py
git add .
git commit -m "update local changes" || true
git pull --rebase origin main
git push