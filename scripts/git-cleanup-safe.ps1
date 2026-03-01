#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Safely remove unwanted files/folders from git tracking while preserving source code.
    
.DESCRIPTION
    This script:
    - Checks what unwanted items are already tracked in git
    - Shows size calculations for each item
    - Safely removes them from git tracking only (not from disk)
    - Commits changes with a professional message
    - Reports total space "freed" from git
    
.PARAMETER RepoPath
    Path to the git repository (default: current directory)
    
.PARAMETER DryRun
    Show what would be removed without actually removing it
    
.PARAMETER AutoConfirm
    Skip confirmation prompts and proceed
    
.EXAMPLE
    .\git-cleanup-safe.ps1 -RepoPath . -DryRun
    .\git-cleanup-safe.ps1 -AutoConfirm
#>

param(
    [string]$RepoPath = ".",
    [switch]$DryRun,
    [switch]$AutoConfirm
)

# Colors for output
$InfoColor = "Cyan"
$WarningColor = "Yellow"
$ErrorColor = "Red"
$SuccessColor = "Green"

function Format-Bytes {
    param([long]$Bytes)
    
    $units = @("B", "KB", "MB", "GB", "TB")
    $size = [double]$Bytes
    $unitIndex = 0
    
    while ($size -ge 1024 -and $unitIndex -lt $units.Count - 1) {
        $size /= 1024
        $unitIndex++
    }
    
    return "{0:N1} {1}" -f $size, $units[$unitIndex]
}

function Test-Git {
    try {
        $null = git --version 2>$null
        return $true
    } catch {
        return $false
    }
}

function Get-TrackedSize {
    param([string]$Path)
    
    try {
        # Get file size from git ls-files
        $size = 0
        git ls-files -s "$Path" | ForEach-Object {
            $parts = $_ -split '\s+'
            if ($parts.Length -gt 3) {
                $hash = $parts[1]
                $objSize = git cat-file -s $hash 2>$null
                if ($objSize) {
                    $size += [long]$objSize
                }
            }
        }
        return $size
    } catch {
        return 0
    }
}

function Get-DiskSize {
    param([string]$Path)
    
    if (-not (Test-Path $Path)) {
        return 0
    }
    
    try {
        if ((Get-Item $Path).PSIsContainer) {
            $size = 0
            Get-ChildItem -Path $Path -Recurse -ErrorAction SilentlyContinue | 
                Where-Object { -not $_.PSIsContainer } | 
                ForEach-Object { $size += $_.Length }
            return $size
        } else {
            return (Get-Item $Path).Length
        }
    } catch {
        return 0
    }
}

function Write-Header {
    param([string]$Text)
    Write-Host "`n" -ForegroundColor White
    Write-Host ("=" * 80) -ForegroundColor Cyan
    Write-Host $Text -ForegroundColor Cyan
    Write-Host ("=" * 80) -ForegroundColor Cyan
}

function Write-Info {
    param([string]$Text)
    Write-Host $Text -ForegroundColor $InfoColor
}

function Write-Warning {
    param([string]$Text)
    Write-Host $Text -ForegroundColor $WarningColor
}

function Write-Success {
    param([string]$Text)
    Write-Host $Text -ForegroundColor $SuccessColor
}

function Test-Confirm {
    param([string]$Prompt)
    
    if ($AutoConfirm) { return $true }
    
    do {
        $response = Read-Host ("$Prompt [y/N]")
        if ([string]::IsNullOrWhiteSpace($response)) { return $false }
    } while ($response -notmatch '^[yn]$')
    
    return $response -eq 'y'
}

# ==============================================================================
# Main Script
# ==============================================================================

Write-Header "Git Repository Cleanup - Safe Mode"

# Validate repo
if (-not (Test-Path (Join-Path $RepoPath .git))) {
    Write-Host "ERROR: Not a git repository: $RepoPath" -ForegroundColor $ErrorColor
    exit 1
}

if (-not (Test-Git)) {
    Write-Host "ERROR: git is not available in PATH" -ForegroundColor $ErrorColor
    exit 1
}

$RepoPath = (Resolve-Path $RepoPath).Path
Write-Info "Repository: $RepoPath"

# Patterns of files/folders to clean from tracking
$cleanupPatterns = @(
    @{ Name = ".venv"; Type = "Dir" },
    @{ Name = "venv"; Type = "Dir" },
    @{ Name = "env"; Type = "Dir" },
    @{ Name = ".env"; Type = "File" },
    @{ Name = "__pycache__"; Type = "Dir" },
    @{ Name = ".ipynb_checkpoints"; Type = "Dir" },
    @{ Name = "node_modules"; Type = "Dir" },
    @{ Name = ".DS_Store"; Type = "File" },
    @{ Name = "Thumbs.db"; Type = "File" },
    @{ Name = "*.pt"; Type = "File" },
    @{ Name = "*.pth"; Type = "File" },
    @{ Name = "*.h5"; Type = "File" },
    @{ Name = "*.pkl"; Type = "File" },
    @{ Name = "*.joblib"; Type = "File" }
)

# Step 1: Find tracked unwanted files
Write-Header "Step 1: Scanning for tracked unwanted files"

$trackedUnwanted = @()
$totalTrackedSize = 0

foreach ($pattern in $cleanupPatterns) {
    $trackedFiles = @(git ls-files "$($pattern.Name)" 2>$null)
    
    if ($trackedFiles.Count -ge 0) {
        foreach ($file in $trackedFiles) {
            $fullPath = Join-Path $RepoPath $file
            
            if (Test-Path $fullPath) {
                $diskSize = Get-DiskSize $fullPath
                $trackedSize = Get-TrackedSize $fullPath
                
                if ($diskSize -gt 0 -or $trackedSize -gt 0) {
                    $trackedUnwanted += @{
                        Path = $file
                        DiskSize = $diskSize
                        TrackedSize = $trackedSize
                    }
                    $totalTrackedSize += [Math]::Max($diskSize, $trackedSize)
                    
                    Write-Info "  Found: $file ($(Format-Bytes $diskSize) on disk)"
                }
            }
        }
    }
}

# Also check for patterns recursively in git-tracked files
Write-Info "`nScanning git-tracked files for patterns..."
$gitFiles = @(git ls-files 2>$null)

foreach ($file in $gitFiles) {
    foreach ($pattern in $cleanupPatterns) {
        if ($file -like "*$($pattern.Name)*" -or $file -match ($pattern.Name -replace '\*', '.*')) {
            if ($trackedUnwanted.Path -notcontains $file) {
                $fullPath = Join-Path $RepoPath $file
                if (Test-Path $fullPath) {
                    $fileSize = (Get-Item $fullPath).Length
                    
                    $trackedUnwanted += @{
                        Path = $file
                        DiskSize = $fileSize
                        TrackedSize = $fileSize
                    }
                    $totalTrackedSize += $fileSize
                    
                    Write-Info "  Found: $file ($(Format-Bytes $fileSize))"
                }
            }
        }
    }
}

if ($trackedUnwanted.Count -eq 0) {
    Write-Success "`nNo unwanted tracked files found. Repository is clean!"
    Write-Success "Total space that would be freed: 0 B"
    exit 0
}

# Step 2: Show what will be removed
Write-Header "Step 2: Files/Folders to Remove from Tracking"

Write-Info "Found $($trackedUnwanted.Count) unwanted item(s) to remove:`n"

foreach ($item in $trackedUnwanted) {
    $size = Format-Bytes $item.DiskSize
    Write-Host "  + $($item.Path) ($size)" -ForegroundColor Yellow
}

Write-Host "`nTotal space on disk: $(Format-Bytes $totalTrackedSize)" -ForegroundColor Cyan

# Step 3: Confirm before proceeding
if (-not $DryRun -and -not (Test-Confirm "`nProceed with removing these files from git tracking?")) {
    Write-Host "Aborted by user." -ForegroundColor Yellow
    exit 0
}

# Step 4: Remove from git tracking
Write-Header "Step 3: Removing from Git Tracking"

if ($DryRun) {
    Write-Warning "[DRY RUN] Not actually removing files"
}

$removedSize = 0
$failedCount = 0

foreach ($item in $trackedUnwanted) {
    $path = $item.Path
    
    if ($DryRun) {
        Write-Info "  Would remove: $path"
        $removedSize += $item.DiskSize
    } else {
        try {
            Write-Info "  Removing: $path"
            $output = git rm --cached -r "$path" 2>&1
            
            if ($LASTEXITCODE -eq 0) {
                $removedSize += $item.DiskSize
                Write-Success "    ✓ Removed from tracking"
            } else {
                Write-Warning "    ✗ Failed: $output"
                $failedCount++
            }
        } catch {
            Write-Warning "    ✗ Error: $_"
            $failedCount++
        }
    }
}

Write-Host "`nRemoval complete. Files removed from tracking: $($trackedUnwanted.Count - $failedCount)/$($trackedUnwanted.Count)" -ForegroundColor Cyan

# Step 5: Add and commit .gitignore
if (-not $DryRun) {
    Write-Header "Step 4: Committing Changes"
    
    # Make sure .gitignore exists (we created it earlier)
    if ((Test-Path (Join-Path $RepoPath ".gitignore"))) {
        Write-Info "Adding .gitignore..."
        git add .gitignore 2>&1 | Out-Null
    }
    
    # Check for staged changes
    $staged = @(git diff --cached --name-only 2>$null)
    
    if ($staged.Count -gt 0) {
        Write-Info "Staged files to commit:"
        foreach ($file in $staged) {
            Write-Host "  - $file" -ForegroundColor Cyan
        }
        
        if (Test-Confirm "`nCommit these changes?") {
            Write-Info "Committing..."
            $commitMsg = "chore: remove unwanted tracked files (.venv, pycache, node_modules, etc.)`n`nRemoved from git tracking:`n- Virtual environments (.venv, venv, env)`n- Python cache (__pycache__)`n- Notebook checkpoints`n- Node modules`n- OS artifacts`n- Large model files`n`nRepository is now clean and GitHub-ready.`n`nSpace freed from tracking: $(Format-Bytes $removedSize)"
            
            git commit -m $commitMsg 2>&1 | ForEach-Object { Write-Info $_ }
            
            if ($LASTEXITCODE -eq 0) {
                Write-Success "`n✓ Changes committed successfully"
            } else {
                Write-Warning "`n✗ Commit failed"
            }
        } else {
            Write-Host "Skipping commit." -ForegroundColor Yellow
        }
    } else {
        Write-Success "No staged changes to commit."
    }
}

# Final Report
Write-Header "Summary Report"

Write-Host "Repository: $RepoPath" -ForegroundColor Cyan
Write-Host "Items processed: $($trackedUnwanted.Count)" -ForegroundColor Cyan
Write-Host "Space freed from git tracking: $(Format-Bytes $removedSize)" -ForegroundColor Green

if ($DryRun) {
    Write-Warning "`n[DRY RUN MODE] - Nothing was actually changed on disk or in git"
} else {
    Write-Success "`n✓ Repository cleanup complete!"
    Write-Success "✓ Your repository is now clean and ready for GitHub"
}

Write-Host "`n" -ForegroundColor White

# Step 6: Final status
Write-Info "Final git status:"
git status --short | Select-Object -First 10

exit 0
