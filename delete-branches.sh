#!/bin/bash

# Usage
# ./delete/branches.sh branch1 branch2 ...
# or:
# ./delete/branches.sh -f --all (force local branches)

FORCE=false

# Check for force flag
if [ "$1" == "-f" ]; then
    FORCE=true
    shift
fi

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 [-f] branch1 branch2 ..." # Show usage information
    exit 1
fi

for branch in "$@"; do
    echo "Deleting branch: $branch"

    # Delete local branch
    if git show-ref --verify --quiet refs/heads/"$branch"; then
        if $FORCE; then
            git branch -D "$branch" && echo "Force deleted local branch: $branch"
        else
            git branch -d "$branch" && echo "Deleted local branch: $branch"
        fi
    else
        echo "Local branch '$branch' does not exist."
    fi

    # Delete remote branch
    if git ls-remote --exit-code --heads origin "$branch" > /dev/null; then
        git push origin --delete "$branch" && echo "Deleted remote branch: $branch"
    else
        echo "Remote branch '$branch' does not exist."
    fi

    echo "-----------------------------------"
done