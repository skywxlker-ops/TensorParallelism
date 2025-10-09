
#!/bin/bash

# Usage: ./update_github.sh "Your commit message"

# Check if commit message is provided
if [ -z "$1" ]; then
  echo "Please provide a commit message."
  exit 1
fi

# Stage all changes
git add .

# Commit changes
git commit -m "$1"

# Pull remote changes and rebase
git pull --rebase origin main

# Push to GitHub
git push

echo "✅ Changes pushed successfully!"
