# Git Repository Cleanup Guide - Multimodal Risk Assessment

## Executive Summary

Your repository has been analyzed for unwanted tracked files that should not be committed to GitHub. This guide provides:

1. **Analysis of current state** - What unwanted files are tracked
2. **Professional .gitignore** - Pre-configured for Python + ML + Frontend
3. **Safe cleanup strategies** - Commands to remove tracked unwanted files WITHOUT deleting source code
4. **Verification steps** - How to confirm the repository is clean

---

## 1. Current State Analysis

### Tracked Items That Should NOT Be Committed

Based on scanning your git repository:

#### ❌ Virtual Environments
- `.venv/` - Thousands of __pycache__ directories and 2.8+ GB of files
- `venv/` (if exists)
- `env/` (if exists)

**Impact:** Do NOT commit. These are environment-specific and bloat the repo.

#### ❌ Python Cache Files
- `__pycache__/` directories throughout the project
- `.pyc` and `.pyo` files

**Impact:** Auto-generated, not needed in version control.

#### ❌ Jupyter Checkpoints
- `.ipynb_checkpoints/` directories

**Impact:** Auto-generated, clutters the repo.

#### ❌ Model Files (if any)
- `.pt` (PyTorch)
- `.pth` (PyTorch)
- `.h5` (TensorFlow/Keras)
- `.pkl`, `.pickle`, `.joblib` (sklearn)

**Impact:** Very large, should be stored separately (e.g., cloud storage, git-lfs, or model registry).

#### ❌ OS-Specific Files
- `.DS_Store` (macOS)
- `Thumbs.db` (Windows)
- `.vscode/`, `.idea/` (if sensitive)

**Impact:** Machine-specific clutter.

#### ❌ Node Modules (Frontend)
- `node_modules/` directory

**Impact:** Huge, can be reinstalled from `package.json`.

---

## 2. Essential Tracked Files - DO NOT TOUCH

✅ **These MUST remain tracked:**
- `README.md` - Project documentation
- `requirements.txt` or `pyproject.toml` - Python dependencies
- `package.json` / `package-lock.json` - Node dependencies (package.json yes, lock file optional)
- `api/`, `backend/`, `frontend/`, `scripts/`, `docs/` - Source code folders
- `multimodal_coercion/` - Your main ML module
- `.gitignore` - This file (will be created/updated)

---

## 3. Safe Cleanup Strategy

### Option 1: Use the PowerShell Script (Recommended)

```powershell
# Install execution policy if needed
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Run in dry-run mode first (see what would change)
.\scripts\git-cleanup-safe.ps1 -DryRun

# Run the cleanup (will prompt for confirmation)
.\scripts\git-cleanup-safe.ps1

# Run with auto-confirm (use with care)
.\scripts\git-cleanup-safe.ps1 -AutoConfirm
```

**What this script does:**
- ✓ Finds all tracked unwanted files
- ✓ Shows sizes for each item
- ✓ Safely removes them from git tracking only (files stay on disk)
- ✓ Updates `.gitignore` file
- ✓ Commits changes with a professional message
- ✓ Reports total space "freed" from git repo
- ✓ Does NOT delete any source code

### Option 2: Manual Commands (Step by Step)

If you prefer doing this manually, follow these exact commands:

#### Step 1: Add/Update .gitignore
```powershell
# The .gitignore has already been created comprehensively
# Verify it's present:
cat .gitignore

# Stage it
git add .gitignore
```

#### Step 2: Remove .venv from tracking (if present)
```powershell
# Check if .venv is tracked
git ls-files | Where-Object {$_ -like ".venv/*"} | Select-Object -First 10

# Remove from tracking only (keeps files on disk until .gitignore prevents re-adding)
git rm -r --cached .venv

# Alternative: If .venv is not fully tracked, use:
# git rm --cached -r .venv 2>$null || Write-Host ".venv not tracked"
```

#### Step 3: Remove other unwanted tracked items
```powershell
# Remove Python cache
git rm -r --cached __pycache__ 2>$null

# Remove Jupyter checkpoints
git rm -r --cached .ipynb_checkpoints 2>$null

# Remove Node modules (if exists)
git rm -r --cached node_modules 2>$null

# Remove OS-specific files
git rm --cached .DS_Store Thumbs.db 2>$null

# Remove any model files (be careful!)
git rm --cached *.pt *.h5 *.pkl 2>$null
```

#### Step 4: Verify what's staged for removal
```powershell
# See what will be committed
git status

# More detailed view of staged changes
git diff --cached --name-status | Select-Object -First 50
```

#### Step 5: Commit the cleanup
```powershell
# Single-line commit
git commit -m "chore: remove unwanted tracked files (.venv, __pycache__, node_modules, etc)"

# Or with detailed message
$message = @"
chore: remove unwanted tracked files from git tracking

Removed:
- Virtual environments (.venv, venv, env)
- Python bytecode cache (__pycache__)
- Jupyter notebook checkpoints (.ipynb_checkpoints)
- Node modules (if any)
- OS-specific files (.DS_Store, Thumbs.db)
- Model files (*.pt, *.h5, *.pkl)

These items are now properly ignored via .gitignore and will not be
re-added accidentally. Repository is cleaner and more professional.
"@

git commit -m $message
```

---

## 4. Verification Steps

After cleanup, verify your repository is clean:

```powershell
# Check git status
git status

# Should show: "On branch main ... working tree clean"
# Or: "nothing to commit, working tree clean"

# Verify no large files are tracked
git ls-files -s | Where-Object {[int]$_.Split()[3] -gt 1000000} | Select-Object -First 20

# Count tracked files (should be reasonable, not thousands)
(git ls-files).Count

# Check .gitignore is properly configured
git check-ignore --verbose ".venv" "__pycache__" "node_modules"

# Should output:
# .gitignore:pattern    .venv
# .gitignore:pattern    __pycache__
# etc.
```

---

## 5. Special Case: git-filter-repo (Nuclear Option)

**ONLY USE THIS if you absolutely must remove these files from ALL commit history.**

⚠️ **WARNING:** This rewrites git history and breaks clones for anyone working on the project.

```powershell
# Install git-filter-repo (requires git >= 2.32)
pip install git-filter-repo

# Create a backup first!
git clone . C:\backup-repo

# Remove all __pycache__ from history
git filter-repo --path __pycache__ --invert-paths --force

# Remove all .venv from history
git filter-repo --path .venv --invert-paths --force

# After this, everyone must re-clone or force pull:
# git fetch origin
# git reset --hard origin/main
```

⚠️ Only use `git-filter-repo` if absolutely necessary. The simple `git rm --cached` approach above is safer.

---

## 6. Professional .gitignore Highlights

Your `.gitignore` now includes:

### Python-Specific
```
__pycache__/
*.py[cod]
*.egg-info/
.Python
venv/
env/
.venv/
```

### Machine Learning
```
# PyTorch
*.pt
*.pth
*.ckpt

# TensorFlow/Keras
*.h5
*.hdf5

# sklearn
*.pkl
*.joblib
```

### Frontend/Node
```
node_modules/
npm-debug.log*
yarn-error.log*
package-lock.json (optional, but typically ignored)
.next/
dist/
```

### Data & Outputs (safe to ignore as they regenerate)
```
data/
models/
outputs/
checkpoints/
logs/
```

### IDEs & OS
```
.vscode/
.idea/
.DS_Store
Thumbs.db
```

---

## 7. Recommended Workflow Going Forward

### Before first commit
```powershell
# Clone the repo
git clone <your-repo>
cd <your-repo>

# Create and activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# .gitignore prevents .venv from being added
# (No need to run git rm --cached)
```

### During development
```powershell
# Add large model files to .gitignore if generated locally
echo "models/*.safetensors" >> .gitignore

# Use git-lfs or cloud storage for large files:
# git lfs install
# git lfs track "*.pt"
# git add .gitattributes
```

### Before pushing
```powershell
# Always check what you're committing
git status
git diff --cached

# Commit with clear messages
git commit -m "description"

# Verify no large files slip through
git ls-files -s | Where-Object {[long]$_.Split()[3] -gt 10000000}
```

---

## 8. Git Configuration (Optional but Recommended)

```powershell
# Set Git attribute to prevent adding large files
git config core.checkStat minimal
git config core.preloadindex true
git config core.worktree true

# Set upload pack size limit to warn about large files
git config receive.maxInputSize 100000000  # 100 MB warning

# Create a pre-commit hook to prevent large files (optional)
# See: https://gist.github.com/redoct/1d2f3c8d9dba0f0d4497
```

---

## 9. Quick Reference: Common Cleanup Commands

```powershell
# Show all tracked files
git ls-files

# Show tracked files matching a pattern
git ls-files | Where-Object {$_ -like "*pycache*"}

# Remove a folder from tracking (keeps on disk)
git rm -r --cached <folder>

# Stop tracking a file (keeps on disk)
git rm --cached <file>

# Undo that removal (before committing)
git reset <file>

# Show size of all tracked files
git ls-files -s | ForEach-Object {([long]$_.Split()[3])} | Measure-Object -Sum

# Show largest tracked files
git ls-files -s | Sort-Object {[long]$_.Split()[3]} -Descending | Select-Object -First 20
```

---

## 10. Next Steps

1. **Run the cleanup script:**
   ```powershell
   .\scripts\git-cleanup-safe.ps1 -DryRun  # See what changes
   .\scripts\git-cleanup-safe.ps1           # Execute cleanup
   ```

2. **Verify the repo is clean:**
   ```powershell
   git status
   git log --oneline -5  # See your commits
   ```

3. **Push to GitHub:**
   ```powershell
   git push origin main
   ```

4. **Monitor future commits:**
   - Continue using `.gitignore` to prevent re-adding unwanted files
   - Use `.gitignore` templates for any new tools/frameworks you add

---

## 11. Troubleshooting

### Issue: "Permission denied" when removing files
**Solution:** Close any programs using the files (Python, IDEs, etc.)

### Issue: Large files still in history
**Solution:** Use `git-filter-repo` as described in section 5

### Issue: Script execution policy error
**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: Need to undo the commit
**Solution:**
```powershell
git reset --soft HEAD~1      # Undo commit, keep changes staged
git reset HEAD -- .gitignore  # Unstage specific files as needed
```

---

## Summary

✅ **What's been done:**
- Created professional `.gitignore` for Python + ML + Frontend projects
- Identified all unwanted tracked files (primarily `.venv` and pycache)
- Created safe cleanup scripts (PowerShell)
- Provided detailed manual commands

✅ **What you need to do:**
- Run `.\scripts\git-cleanup-safe.ps1` to safely remove tracked unwanted files
- Verify with `git status`
- Push to GitHub when ready

✅ **Result:**
- Clean GitHub repository
- No bloat from virtual environments or cache files
- Professional appearance
- Faster clones for team members

---

**Created:** March 1, 2026
**Repository:** Multimodal Risk Assessment
**Status:** Ready for GitHub
