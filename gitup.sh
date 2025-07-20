#!/bin/bash
git add .
git commit -m "$1"
git push origin main
echo "Changes pushed to main branch with message: $1"