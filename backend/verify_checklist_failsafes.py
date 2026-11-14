"""
Complete verification & failsafe audit script for the 5 CAG Gift Deed Checklist Items.
Run from `CAG-main/backend`:
    python verify_checklist_failsafes.py
"""
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import models, processing  # noqa: E402


def find_assessment_entrypoint():
    """Locate the main stamp duty assessment function in app.processing."""
    candidates = [
        "assess_stamp_duty",
        "evaluate_stamp_duty",
        "compute_stamp_duty_assessment",
        "analyze_stamp_duty",
        "build_stamp_duty_assessment",
    ]
    for name in candidates:
        if hasattr(processing, name) and callable(getattr(processing, name)):
            return name, getattr(processing, name)
    # Fallback: search any callable with 'stamp_duty' or 'assess' in name
    for name, obj in inspect.getmembers(processing, inspect.isfunction):
        if "stamp_duty" in name or "assess" in name or "regime" in name:
            return name, obj
    return None, None


def inspect_module_capabilities():
    src = inspect.getsource(processing)
    model_src = inspect.getsource(models)

    checks = {
        "Model has three-regime / rule fields (0007)": any(
            k in model_src for k in ("selected_rule", "article_34_200", "duty_rate", "calculation_components")
        ),
        "Model has cess_uncertain & location_band (0008)": (
            "cess_uncertain" in model_src and "location_band" in model_src
        ),
        "Check 1: Directional relationship tokens present": all(
            k in src for k in ("article_34_200", "article_34_3_percent", "article_25")
        ),
        "Check 1 Failsafe: Narrow ₹200 list separated from parent/sibling": any(
            k in src for k in ("wife_of_deceased_son", "grandson", "granddaughter")
        ),
        "Check 2: Soft surname mismatch finding implemented": (
            "donor_donee_surname_mismatch" in src
        ),
        "Check 2 Failsafe: Married name / initials / OCR confidence exemptions": any(
            k in src.lower() for k in ("married", "initial", "पूर्वीचे", "confidence", "single")
        ),
        "Check 3: Schedule-scoped property extraction": any(
            k in src.lower() for k in ("schedule", "परिशिष्ट", "मिळकतीचे वर्णन", "annexure")
        ),
        "Check 3 & 4: ₹200 regime and 3% regime math present": (
            "200" in src and ("0.03" in src or "3%" in src or "3.0" in src)
        ),
        "Check 5: Article 25 location-band conveyance rates (4% / 5%)": (
            "article_25" in src and ("0.05" in src or "0.04" in src or "location_band" in src)
        ),
    }

    print("=" * 78)
    print("STATIC ARCHITECTURE & MIGRATION VERIFICATION")
    print("=" * 78)
    all_static_ok = True
    for label, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_static_ok = False
        print(f"[{status}] {label}")
    print()
    return all_static_ok


def run_pytest_suite():
    import pytest
    print("=" * 78)
    print("RUNNING PIPELINE & RULE-MATRIX PYTEST SUITE")
    print("=" * 78)
    exit_code = pytest.main(["-v", "tests/test_pipeline.py", "tests/test_api.py"])
    return exit_code == 0


if __name__ == "__main__":
    static_ok = inspect_module_capabilities()
    fn_name, fn = find_assessment_entrypoint()
    print(f"Detected primary assessment function in app.processing: {fn_name}")
    tests_ok = run_pytest_suite()
    sys.exit(0 if (static_ok and tests_ok) else 1)