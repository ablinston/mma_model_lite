
#!/bin/bash

# Exit on errors
set +e

# Activate miniconda
source /home/andy/miniconda3/etc/profile.d/conda.sh

# Navigate to the repository
REPO_DIR="/andy/home/Git_repos/mma_model_lite"
cd "$REPO_DIR"

# Get the latest main branch from Github
git checkout main
git pull origin main

# Activate the anaconda python environment and update it with the packages needed
conda activate mma_model
conda env update -f environment.yml

# Delete the old branch and checkout a new version
if git show-ref --verify --quiet refs/heads/raspberry_pi_update; then
    git branch -D raspberry_pi_update
else
    echo "Branch 'raspberry_pi_update' does not exist. Skipping deletion."
fi
git checkout -b raspberry_pi_update

# Run the update code
python3 src/13_predict_future_events.py

# Commit the changes and push
git commit -am "automatic raspberry pi updates"
git push -f origin raspberry_pi_update

# Keep terminal open after script finishes
exec bash
