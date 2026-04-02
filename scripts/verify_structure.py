#!/usr/bin/env python3
"""Project structure validation script.

Verifies that the Multimodal-risk-assessment project follows the expected
directory structure, contains required __init__.py files, and has no clutter.

Exit codes:
    0: All validations passed (PASS)
    1: One or more validations failed (FAIL)
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple


class ProjectValidator:
    """Validates project structure and configuration."""

    def __init__(self, project_root: Path):
        """Initialize validator with project root path.

        Args:
            project_root: Path to project root directory
        """
        self.project_root = project_root
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_all(self) -> bool:
        """Run all validation checks.

        Returns:
            True if all checks pass, False otherwise
        """
        print("=" * 70)
        print("PROJECT STRUCTURE VALIDATION")
        print("=" * 70)
        print(f"Project root: {self.project_root}\n")

        # Run validation checks
        self.check_required_directories()
        self.check_init_files()
        self.check_no_clutter()
        self.check_pytest_structure()

        # Report results
        return self.report_results()

    def check_required_directories(self):
        """Verify required directory structure exists."""
        print("1. Checking required directories...")

        required_dirs = [
            "multimodal_coercion",
            "multimodal_coercion/core",
            "multimodal_coercion/facial_emotion",
            "multimodal_coercion/speech",
            "multimodal_coercion/fusion",
            "multimodal_coercion/calibration",
            "multimodal_coercion/risk",
            "multimodal_coercion/ui",
            "multimodal_coercion/orchestrator",
            "multimodal_coercion/engine",
            "multimodal_coercion/configs",
            "multimodal_coercion/models",
            "multimodal_coercion/artifacts",
            "backend",
            "api",
            "frontend",
            "docs",
            "tests",
            "outputs",
            "scripts",
        ]

        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if not full_path.is_dir():
                self.errors.append(f"Missing directory: {dir_path}")
            else:
                print(f"   ✓ {dir_path}")

    def check_init_files(self):
        """Verify __init__.py files exist in package directories."""
        print("\n2. Checking __init__.py files...")

        required_init = [
            "multimodal_coercion",
            "multimodal_coercion/core",
            "multimodal_coercion/facial_emotion",
            "multimodal_coercion/speech",
            "multimodal_coercion/fusion",
            "multimodal_coercion/calibration",
            "multimodal_coercion/risk",
            "multimodal_coercion/ui",
            "multimodal_coercion/orchestrator",
            "multimodal_coercion/engine",
            "multimodal_coercion/configs",
            "tests",
        ]

        for dir_path in required_init:
            init_file = self.project_root / dir_path / "__init__.py"
            if not init_file.exists():
                self.errors.append(f"Missing __init__.py: {dir_path}/__init__.py")
            elif init_file.stat().st_size == 0:
                self.warnings.append(f"Empty __init__.py: {dir_path}/__init__.py")
                print(f"   ℹ {dir_path}/__init__.py (empty but present)")
            else:
                print(f"   ✓ {dir_path}/__init__.py")

    def check_no_clutter(self):
        """Verify no clutter/temporary files in root."""
        print("\n3. Checking for clutter files...")

        clutter_files = [
            "CLEANUP_GUIDE.md",
            "EXECUTION_GUIDE.md",
            "GITHUB_READY.md",
            "OPTIMIZATION_QUICK_REFERENCE.md",
            "PERFORMANCE_OPTIMIZATION_REPORT.md",
        ]

        found_clutter = False
        for file_name in clutter_files:
            file_path = self.project_root / file_name
            if file_path.exists():
                self.errors.append(f"Clutter file found: {file_name}")
                found_clutter = True

        if not found_clutter:
            print("   ✓ No clutter files detected")

    def check_pytest_structure(self):
        """Verify pytest test structure."""
        print("\n4. Checking pytest structure...")

        conftest_path = self.project_root / "tests" / "conftest.py"
        if not conftest_path.exists():
            self.errors.append("Missing tests/conftest.py")
        else:
            print("   ✓ tests/conftest.py")

        test_files = [
            "tests/test_facial_emotion.py",
            "tests/test_speech.py",
            "tests/test_fusion.py",
            "tests/test_calibration.py",
        ]

        for test_file in test_files:
            path = self.project_root / test_file
            if not path.exists():
                self.errors.append(f"Missing {test_file}")
            else:
                print(f"   ✓ {test_file}")

    def report_results(self) -> bool:
        """Print validation results and return pass/fail status.

        Returns:
            True if validation passed, False otherwise
        """
        print("\n" + "=" * 70)

        if self.warnings:
            print("WARNINGS:")
            for warning in self.warnings:
                print(f"  ⚠ {warning}")
            print()

        if self.errors:
            print("ERRORS:")
            for error in self.errors:
                print(f"  ✗ {error}")
            print("\nStatus: FAIL")
            print("=" * 70)
            return False
        else:
            print("Status: PASS")
            print("All validations passed successfully!")
            print("=" * 70)
            return True


def main():
    """Main entry point for validation script."""
    # Determine project root (parent of this script if in scripts/, else CWD)
    script_dir = Path(__file__).parent
    if script_dir.name == "scripts":
        project_root = script_dir.parent
    else:
        project_root = Path.cwd()

    # Run validation
    validator = ProjectValidator(project_root)
    if validator.validate_all():
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure


if __name__ == "__main__":
    main()
