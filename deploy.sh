#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

echo "=== Starting Smart Farm Deployment ==="

# 1. Pull latest code changes (if in a git repo)
if [ -d .git ]; then
    echo "Pulling latest code changes from remote repository..."
    git pull origin main
else
    echo "No git repository detected, skipping code pull."
fi

# 2. Build and restart Docker containers
echo "Rebuilding and starting docker containers..."
docker compose down
docker compose build --no-cache
docker compose up -d --remove-orphans

# 3. Clean up old/dangling docker resources to free disk space
echo "Cleaning up unused Docker images..."
docker image prune -f

echo "=== Deployment Successfully Completed ==="
echo "You can check container status with: docker compose ps"
echo "You can view logs with: docker compose logs -f"
