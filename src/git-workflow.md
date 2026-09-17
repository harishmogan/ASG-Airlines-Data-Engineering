# Git Workflow Documentation

## Basic Git Commands

- `git clone` – Creates a local copy of a remote repository.
- `git status` – Shows the current state of the working directory.
- `git branch` – Lists the available branches.
- `git switch` – Switches to another branch.
- `git add` – Stages changes for commit.
- `git commit` – Saves the staged changes as a commit.
- `git push` – Uploads local commits to the remote repository.
- `git pull` – Downloads and integrates the latest changes from the remote repository.
- `git merge` – Combines changes from one branch into another.

### Branch Workflow

main -> Create Branch -> Work -> Commit -> Push -> Pull Request -> Review-> Merge

### Merging 

git switch main
git merge feature-branch

### Merge Conflicts

1. Open the file that contains conflict.
2. Review the conflicting changes.
3. Edit the required code.
4. Remove the conflict.
5. Save the file.
6. Stage and commit the solved changes.

### Branch updation

From Latest main -> Update our Branch -> Continue our work