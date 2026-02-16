#!/bin/bash
set -e

# Initialize git
git init
git branch -M main

# Configure user (local to this repo)
git config user.email "jasonyao@example.com"
git config user.name "Jason Yao"

# Add remote
git remote add origin https://github.com/JasonYao3/Transform-LM.git

# Add files (respecting .gitignore which excludes data/)
git add .
git commit -m "Initial commit of Transformer from Scratch"

# Push
echo "Pushing to origin main..."
git push -u origin main
