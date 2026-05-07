# GitHub Workflow Guide (for this repository)

This guide helps you start pushing this codebase to GitHub safely while keeping large/generated artifacts out of version control.

## 1) One-time local setup

From the project root:

```powershell
cd C:\Users\irandoust\Desktop\wp\PINN\jax_elast
```

Initialize git (if not already initialized):

```powershell
git init
git branch -M main
```

Set your identity (if needed):

```powershell
git config user.name "Your Name"
git config user.email "you@example.com"
```

## 2) Create the GitHub repository

On GitHub, create a new empty repository (no README/.gitignore/license, since this project already has files).

Copy the remote URL and add it locally:

```powershell
git remote add origin https://github.com/<your-user>/<your-repo>.git
```

If a remote already exists, update it:

```powershell
git remote set-url origin https://github.com/<your-user>/<your-repo>.git
```

## 3) First commit and push

```powershell
git add .
git status
git commit -m "Initial commit: JAX elasticity PINN"
git push -u origin main
```

## 4) Day-to-day workflow

Use this simple cycle:

```powershell
git pull --rebase
git add -A
git commit -m "Describe your change"
git push
```

## 5) Recommended branching pattern

For non-trivial changes, use feature branches:

```powershell
git checkout -b feat/<short-name>
# make changes
git add -A
git commit -m "feat: ..."
git push -u origin feat/<short-name>
```

Open a Pull Request on GitHub and merge after review.

## 6) What is intentionally not committed

The `.gitignore` is configured to exclude:
- training outputs in `results/` and `results_*/`
- large model artifacts (`best_params.pkl`, `final_params.pkl`, `loss_history.npz`)
- logs, caches, IDE metadata, and local virtual environments

This keeps repository history small and avoids pushing machine-specific/generated files.

## 7) Optional: add a minimal GitHub Actions CI later

When you are ready, you can add a workflow under `.github/workflows/` that runs lightweight checks only (imports, lint, smoke test) instead of full training.

Suggested CI steps:
- checkout
- setup Python
- install `requirements.txt`
- run `python src/check_imports.py`
- run `python smoke_test.py`

Keep CI fast; avoid running 150k+ epoch training in GitHub Actions.
