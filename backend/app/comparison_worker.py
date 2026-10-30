import json
import sys
import traceback
from pathlib import Path

from .document_comparison import _run_engine_direct


def main() -> int:
    if len(sys.argv) != 4:
        print("Usage: comparison_worker ENGINE SOURCE_PDF OUTPUT_DIR", file=sys.stderr)
        return 2
    engine, source_pdf, output_dir = sys.argv[1:]
    try:
        result = _run_engine_direct(engine, Path(source_pdf), Path(output_dir))
        (Path(output_dir) / "_worker-result.json").write_text(
            json.dumps(result, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
