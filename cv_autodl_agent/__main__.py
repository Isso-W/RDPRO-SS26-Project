from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io_utils import read_json, write_json
from .schemas import DatasetManifest, RetrievedModelCandidate
from .workflow import CVAutoDLWorkflow



def main() -> int:
    parser = argparse.ArgumentParser(description="Run the CV Auto-DL codegen workflow")
    parser.add_argument("--manifest", required=True, help="Path to DatasetManifest JSON")
    parser.add_argument("--candidates", required=True, help="Path to RetrievedModelCandidate[] JSON")
    parser.add_argument("--output-dir", required=True, help="Directory where generated artifacts will be written")
    parser.add_argument("--execution-mode", default="simulate", choices=["simulate", "real"])
    parser.add_argument(
        "--notebook-execution-mode",
        choices=["simulate", "real"],
        default=None,
        help="Execution mode written into the exported Colab notebook. Defaults to --execution-mode.",
    )
    args = parser.parse_args()

    manifest = DatasetManifest.from_dict(read_json(args.manifest))
    candidates = [RetrievedModelCandidate.from_dict(item) for item in read_json(args.candidates)]
    workflow = CVAutoDLWorkflow()
    result = workflow.run(
        manifest,
        candidates,
        args.output_dir,
        execution_mode=args.execution_mode,
        notebook_execution_mode=args.notebook_execution_mode,
    )
    write_json(Path(args.output_dir) / "workflow_result.json", result)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
