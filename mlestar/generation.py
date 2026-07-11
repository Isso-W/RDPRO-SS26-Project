"""Generate and statically validate component-scoped MLE-STAR projects.

The LLM boundary is deliberately small and OpenAI-compatible.  A provider may
suggest a small JSON file envelope, but the result is accepted only after the
same source-contract checks used for the deterministic fallback.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Protocol, Sequence

from .contracts import COMPONENT_NAMES, CandidateProject, Component, SearchEvidence, TaskContract
from .dataops import COMPONENT_FUNCTIONS


_MARKER = re.compile(r"^\s*#\s*MLESTAR_COMPONENT:([A-Za-z_]+):(START|END)\s*$")
_SAFE_FILES = frozenset({"pipeline.py", "config.json", "requirements.txt", "README.md"})
_IMPORT_PACKAGES = {
    "sklearn": "scikit-learn",
    "PIL": "pillow",
    "yaml": "pyyaml",
    "cv2": "opencv-python",
}
_NETWORK_MODULES = {"requests", "httpx", "urllib", "socket", "http", "ftplib", "websocket"}
_STDLIB = frozenset(getattr(sys, "stdlib_module_names", ())) | {"__future__", "typing_extensions"}


class GenerationProvider(Protocol):
    """An injected LLM-like source provider; implementations own transport."""

    def generate(self, prompt: str) -> str | Mapping[str, Any]:
        """Return a JSON envelope with ``files``, ``rationale``, and ``assumptions``."""


@dataclass(frozen=True)
class StaticGenerationProvider:
    """Deterministic provider useful for unit tests and offline dry runs."""

    response: str | Mapping[str, Any]

    def generate(self, prompt: str) -> str | Mapping[str, Any]:
        del prompt
        return self.response


@dataclass(frozen=True)
class ConfiguredGenerationProvider:
    """Generate through a standalone OpenAI-compatible provider.

    ``openai`` reads ``OPENAI_API_KEY`` and optional ``MLESTAR_OPENAI_MODEL``.
    ``qwen`` reads ``JIAOZI_DASHSCOPE_API_KEY`` and optional
    ``DASHSCOPE_BASE_URL`` / ``MLESTAR_QWEN_MODEL``.  This avoids depending on
    another Jiaozi module when the MLE-STAR branch is checked out by itself.
    """

    provider: str

    def generate(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency error is environment-specific.
            raise RuntimeError("Install the optional LLM dependency with `pip install openai`.") from exc

        if self.provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            model = os.environ.get("MLESTAR_OPENAI_MODEL", "gpt-4o-mini")
            base_url = os.environ.get("OPENAI_BASE_URL")
        elif self.provider == "qwen":
            api_key = os.environ.get("JIAOZI_DASHSCOPE_API_KEY")
            model = os.environ.get("MLESTAR_QWEN_MODEL", "qwen-plus")
            base_url = os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        else:
            raise ValueError(f"Unsupported standalone MLE-STAR LLM provider {self.provider!r}.")
        if not api_key:
            raise RuntimeError(f"Missing API key for MLE-STAR LLM provider {self.provider!r}.")

        client = OpenAI(api_key=api_key, base_url=base_url)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Return only the requested JSON project envelope. Do not use markdown fences."},
                {"role": "user", "content": prompt},
            ],
        )
        response = completion.choices[0].message.content if completion.choices else None
        if not response or not response.strip():
            raise RuntimeError(f"Configured LLM provider {self.provider!r} returned no project source.")
        return response


@dataclass(frozen=True)
class ValidationResult:
    """The inspectable result of source-contract validation."""

    errors: tuple[str, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class GenerationResult:
    """A generated candidate and the validation result that admitted it."""

    candidate: CandidateProject
    validation: ValidationResult
    used_fallback: bool
    fallback_reason: str = ""
    rationale: str = ""
    assumptions: tuple[str, ...] = ()

    @property
    def project_dir(self) -> Path:
        return Path(self.candidate.project_dir)


def build_generation_prompt(task: TaskContract, evidence: Sequence[SearchEvidence] = ()) -> str:
    """Describe the exact, JSON-only generation contract for an injected provider."""

    evidence_lines = [f"- {item.title}: {item.model_hint or item.summary} ({item.url})" for item in evidence]
    return "\n".join(
        (
            "Generate a Python ML project as JSON only.",
            'Return {"files": {"pipeline.py": "...", "requirements.txt": "..."}, '
            '"rationale": "...", "assumptions": ["..."]}.',
            "pipeline.py must define the five state -> state functions and wrap each one in exactly one marker pair:",
            *(
                f"# MLESTAR_COMPONENT:{component}:START ... # MLESTAR_COMPONENT:{component}:END"
                for component in COMPONENT_NAMES
            ),
            f"Task: {json.dumps(task.to_dict(), sort_keys=True)}",
            "Evidence:",
            *(evidence_lines or ["- No retrieved evidence; use a deterministic baseline."]),
            "Do not use eval, exec, subprocesses, network access, test targets, or absolute output paths.",
        )
    )


def generate_project(
    project_dir: str | Path,
    task: TaskContract,
    *,
    candidate_id: str = "baseline",
    provider: GenerationProvider | None = None,
    evidence: Sequence[SearchEvidence] = (),
) -> GenerationResult:
    """Generate one project, falling back when a provider response is unsafe or invalid."""

    destination = Path(project_dir)
    rationale = ""
    assumptions: tuple[str, ...] = ()
    fallback_reason = "provider was not configured"
    used_fallback = provider is None
    if provider is not None:
        try:
            envelope = _parse_envelope(provider.generate(build_generation_prompt(task, evidence)))
            rationale = _text(envelope.get("rationale"))
            assumptions = _text_tuple(envelope.get("assumptions"))
            _write_provider_files(destination, envelope["files"], task, evidence)
            validation = validate_generated_project(destination, task)
            if validation.is_valid:
                return _result(destination, task, candidate_id, validation, False, rationale=rationale, assumptions=assumptions)
            fallback_reason = "; ".join(validation.errors)
        except Exception as exc:
            fallback_reason = f"{type(exc).__name__}: {exc}"

    write_fallback_project(destination, task, evidence=evidence, fallback_reason=fallback_reason)
    validation = validate_generated_project(destination, task)
    if not validation.is_valid:  # a programming error in the trusted fallback must be loud
        raise RuntimeError(f"Deterministic fallback failed validation: {'; '.join(validation.errors)}")
    return _result(
        destination,
        task,
        candidate_id,
        validation,
        used_fallback=True,
        fallback_reason=fallback_reason,
        rationale=rationale,
        assumptions=assumptions,
    )


def generate_candidate_project(
    project_dir: str | Path,
    task: TaskContract,
    **kwargs: Any,
) -> CandidateProject:
    """Compatibility-sized helper returning just the persisted candidate contract."""

    return generate_project(project_dir, task, **kwargs).candidate


def write_fallback_project(
    project_dir: str | Path,
    task: TaskContract,
    *,
    evidence: Sequence[SearchEvidence] = (),
    fallback_reason: str = "",
) -> Path:
    """Write a deterministic, JSON-state-only DataOps protocol project."""

    destination = Path(project_dir)
    destination.mkdir(parents=True, exist_ok=True)
    config = {
        "task_id": task.task_id,
        "contract": task.to_dict(),
        "generation": {
            "source": "deterministic_fallback",
            "fallback_reason": fallback_reason,
            "evidence": [
                {"title": item.title, "url": item.url, "model_hint": item.model_hint} for item in evidence
            ],
        },
    }
    (destination / "pipeline.py").write_text(_fallback_pipeline(task.modality), encoding="utf-8")
    (destination / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    (destination / "requirements.txt").write_text(
        "skrub>=0.9,<0.10\nscikit-learn\ntimm\ntorch\n",
        encoding="utf-8",
    )
    (destination / "README.md").write_text(
        "# Deterministic MLE-STAR fallback\n\n"
        "This project follows the five-function `mlestar.dataops` pipeline protocol. "
        "The runtime owns model fitting and persistence; this source keeps the DataOps state JSON-compatible.\n",
        encoding="utf-8",
    )
    return destination


def validate_generated_project(project_dir: str | Path, task: TaskContract) -> ValidationResult:
    """Validate marker structure, DataOps protocol, imports, and unsafe AST patterns."""

    del task  # The source protocol is modality-independent; config carries task details.
    project = Path(project_dir)
    pipeline = project / "pipeline.py"
    if not pipeline.is_file():
        return ValidationResult((f"Generated project is missing {pipeline.name}.",))
    try:
        source = pipeline.read_text(encoding="utf-8")
    except OSError as exc:
        return ValidationResult((f"Cannot read {pipeline.name}: {type(exc).__name__}: {exc}",))
    return validate_generated_source(source, requirements_path=project / "requirements.txt")


def validate_generated_source(source: str, *, requirements_path: str | Path | None = None) -> ValidationResult:
    """Validate a pipeline source string without importing or executing it."""

    errors: list[str] = []
    ranges = _marker_ranges(source, errors)
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return ValidationResult(tuple(errors + [f"pipeline.py is not valid Python: {exc.msg} at line {exc.lineno}."]))
    _validate_component_functions(tree, ranges, errors)
    requirements = _requirements(requirements_path, errors)
    visitor = _SafetyVisitor(ranges, requirements, errors)
    visitor.visit(tree)
    return ValidationResult(tuple(errors))


def _marker_ranges(source: str, errors: list[str]) -> dict[str, tuple[int, int]]:
    markers: dict[str, dict[str, list[int]]] = {name: {"START": [], "END": []} for name in COMPONENT_NAMES}
    for line_number, line in enumerate(source.splitlines(), start=1):
        match = _MARKER.match(line)
        if not match:
            continue
        component, boundary = match.groups()
        if component not in markers:
            errors.append(f"Unsupported component marker {component!r} at line {line_number}.")
            continue
        markers[component][boundary].append(line_number)
    ranges: dict[str, tuple[int, int]] = {}
    for component in COMPONENT_NAMES:
        starts, ends = markers[component]["START"], markers[component]["END"]
        if len(starts) != 1 or len(ends) != 1:
            errors.append(f"Component {component} must have exactly one START and one END marker.")
            continue
        if starts[0] >= ends[0]:
            errors.append(f"Component {component} END marker must follow its START marker.")
            continue
        ranges[component] = (starts[0], ends[0])
    return ranges


def _validate_component_functions(tree: ast.Module, ranges: Mapping[str, tuple[int, int]], errors: list[str]) -> None:
    functions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for component, function_name in COMPONENT_FUNCTIONS.items():
        node = functions.get(function_name)
        if node is None:
            errors.append(f"pipeline.py must define callable {function_name}(state).")
            continue
        boundaries = ranges.get(component)
        if boundaries is None:
            continue
        start, end = boundaries
        if not (start < node.lineno and (node.end_lineno or node.lineno) < end):
            errors.append(f"{function_name} must be fully inside the {component} component marker.")
        positional = [*node.args.posonlyargs, *node.args.args]
        if (
            len(positional) != 1
            or positional[0].arg != "state"
            or node.args.vararg is not None
            or node.args.kwonlyargs
            or node.args.kwarg is not None
        ):
            errors.append(f"{function_name} must use the protocol signature {function_name}(state).")


def _requirements(path: str | Path | None, errors: list[str]) -> set[str]:
    if path is None:
        return set()
    requirements = Path(path)
    if not requirements.exists():
        errors.append("Generated project is missing requirements.txt.")
        return set()
    declared: set[str] = set()
    for line in requirements.read_text(encoding="utf-8").splitlines():
        item = line.split("#", 1)[0].strip()
        if not item or item.startswith("-"):
            continue
        name = re.split(r"[<>=!~;\[]", item, maxsplit=1)[0].strip().lower().replace("_", "-")
        if name:
            declared.add(name)
    return declared


class _SafetyVisitor(ast.NodeVisitor):
    def __init__(self, ranges: Mapping[str, tuple[int, int]], requirements: set[str], errors: list[str]) -> None:
        self.ranges = ranges
        self.requirements = requirements
        self.errors = errors

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._import(alias.name.split(".", 1)[0], node.lineno)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if not node.level and node.module:
            self._import(node.module.split(".", 1)[0], node.lineno)

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node.func)
        component = _component_at(node.lineno, self.ranges)
        leaf = name.rsplit(".", 1)[-1]
        if leaf in {"eval", "exec", "compile", "__import__"}:
            self.errors.append(f"Unsafe call {leaf} at line {node.lineno}.")
        if name.startswith("subprocess.") or name in {"os.system", "os.popen"} or name.startswith("os.exec"):
            self.errors.append(f"Shell subprocess call {name} is forbidden at line {node.lineno}.")
        if name.split(".", 1)[0] in _NETWORK_MODULES or leaf in {"urlopen", "urlretrieve"}:
            self.errors.append(f"Network call {name} is forbidden at line {node.lineno}.")
        if leaf in {"fit", "fit_predict", "partial_fit"} and component != "training":
            self.errors.append(f"Model fitting outside component marker at line {node.lineno}.")
        if leaf in {"fit_transform", "transform"} and component not in {"data_preparation", "prediction"}:
            self.errors.append(f"Preprocessing outside component marker at line {node.lineno}.")
        if leaf in {"GridSearchCV", "RandomizedSearchCV", "cross_val_score"} and component != "model":
            self.errors.append(f"Model selection outside component marker at line {node.lineno}.")
        if _is_absolute_write(node, name):
            self.errors.append(f"absolute write outside project directory at line {node.lineno}.")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and _is_test_target(node.value):
            self.errors.append(f"Reading test targets is forbidden at line {node.lineno}.")

    def _import(self, root: str, line: int) -> None:
        if root in {"mlestar", "__future__"} or root in _STDLIB:
            return
        if root in {"subprocess", *(_NETWORK_MODULES)}:
            self.errors.append(f"Unsafe import {root} at line {line}.")
            return
        package = _IMPORT_PACKAGES.get(root, root).lower().replace("_", "-")
        if package not in self.requirements:
            self.errors.append(f"Import {root} at line {line} is missing from requirements.txt.")


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _component_at(line: int, ranges: Mapping[str, tuple[int, int]]) -> str | None:
    for component, (start, end) in ranges.items():
        if start < line < end:
            return component
    return None


def _is_absolute_write(node: ast.Call, name: str) -> bool:
    leaf = name.rsplit(".", 1)[-1]
    path_node: ast.expr | None = None
    if leaf == "open":
        mode = _literal_mode(node)
        if not _is_write_mode(mode):
            return False
        path_node = node.args[0] if node.args else _keyword_value(node, "file")
    elif leaf in {"write_text", "write_bytes", "touch", "mkdir"}:
        path_node = _path_receiver(node.func)
    if path_node is None:
        return False
    value = _literal_path(path_node)
    return value is not None and Path(value).is_absolute()


def _literal_mode(node: ast.Call) -> str:
    mode = node.args[1] if len(node.args) > 1 else _keyword_value(node, "mode")
    return mode.value if isinstance(mode, ast.Constant) and isinstance(mode.value, str) else "r"


def _is_write_mode(mode: str) -> bool:
    return any(flag in mode for flag in "wax+")


def _keyword_value(node: ast.Call, name: str) -> ast.expr | None:
    return next((item.value for item in node.keywords if item.arg == name), None)


def _path_receiver(function: ast.expr) -> ast.expr | None:
    return function.value if isinstance(function, ast.Attribute) else None


def _literal_path(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Call) and _call_name(node.func).rsplit(".", 1)[-1] == "Path" and node.args:
        return _literal_path(node.args[0])
    return None


def _is_test_target(value: str) -> bool:
    lowered = value.lower().replace("-", "_")
    return "test" in lowered and any(token in lowered for token in ("target", "label", "truth", "ground"))


def _parse_envelope(value: str | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("provider did not return a JSON envelope") from exc
    else:
        parsed = dict(value)
    if not isinstance(parsed, Mapping) or not isinstance(parsed.get("files"), Mapping):
        raise ValueError("provider envelope must contain a files mapping")
    return parsed


def _write_provider_files(
    destination: Path,
    files: object,
    task: TaskContract,
    evidence: Sequence[SearchEvidence],
) -> None:
    if not isinstance(files, Mapping):
        raise ValueError("provider files must be a mapping")
    unknown = sorted(str(name) for name in files if str(name) not in _SAFE_FILES)
    if unknown:
        raise ValueError(f"provider attempted unsupported file paths: {', '.join(unknown)}")
    if "pipeline.py" not in files or not isinstance(files["pipeline.py"], str):
        raise ValueError("provider must supply pipeline.py source")
    destination.mkdir(parents=True, exist_ok=True)
    defaults = _fallback_metadata(task, evidence)
    for name in _SAFE_FILES:
        content = files[name] if name in files else defaults[name]
        if not isinstance(content, str):
            raise ValueError(f"provider file {name} must be text")
        (destination / name).write_text(content, encoding="utf-8")


def _fallback_metadata(task: TaskContract, evidence: Sequence[SearchEvidence]) -> dict[str, str]:
    config = {
        "task_id": task.task_id,
        "contract": task.to_dict(),
        "generation": {"source": "provider", "evidence": [item.to_dict() for item in evidence]},
    }
    return {
        "config.json": json.dumps(config, indent=2, sort_keys=True),
        "requirements.txt": "skrub>=0.9,<0.10\n",
        "README.md": "# Generated MLE-STAR project\n",
    }


def _result(
    destination: Path,
    task: TaskContract,
    candidate_id: str,
    validation: ValidationResult,
    used_fallback: bool,
    *,
    fallback_reason: str = "",
    rationale: str = "",
    assumptions: tuple[str, ...] = (),
) -> GenerationResult:
    source = (destination / "pipeline.py").read_bytes()
    candidate = CandidateProject(
        candidate_id=candidate_id,
        project_dir=str(destination.resolve()),
        code_sha256=sha256(source).hexdigest(),
        components=tuple(Component(name) for name in COMPONENT_NAMES),
    )
    return GenerationResult(candidate, validation, used_fallback, fallback_reason, rationale, assumptions)


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _text_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())


def _fallback_pipeline(modality: str) -> str:
    """Return a source-only fallback; its state remains JSON-compatible for DataOps."""

    model = (
        "{'implementation': 'sklearn.dummy.DummyClassifier', 'strategy': 'prior'}"
        if modality == "tabular"
        else "{'implementation': 'timm.create_model', 'architecture': 'resnet18', 'pretrained': False}"
    )
    return f'''# Deterministic fallback. The executor materializes this declared model.

# MLESTAR_COMPONENT:data_loading:START
def load_data(state):
    artifacts = dict(state.get("artifacts", {{}}))
    artifacts["data_loading"] = {{"status": "ready"}}
    return {{**state, "artifacts": artifacts}}
# MLESTAR_COMPONENT:data_loading:END

# MLESTAR_COMPONENT:data_preparation:START
def prepare_data(state):
    artifacts = dict(state.get("artifacts", {{}}))
    artifacts["data_preparation"] = {{"status": "ready"}}
    return {{**state, "artifacts": artifacts}}
# MLESTAR_COMPONENT:data_preparation:END

# MLESTAR_COMPONENT:model:START
def build_model(state):
    artifacts = dict(state.get("artifacts", {{}}))
    artifacts["model"] = {model}
    return {{**state, "artifacts": artifacts}}
# MLESTAR_COMPONENT:model:END

# MLESTAR_COMPONENT:training:START
def train_model(state):
    artifacts = dict(state.get("artifacts", {{}}))
    artifacts["training"] = {{"status": "configured"}}
    return {{**state, "artifacts": artifacts}}
# MLESTAR_COMPONENT:training:END

# MLESTAR_COMPONENT:prediction:START
def predict_or_submit(state):
    artifacts = dict(state.get("artifacts", {{}}))
    artifacts["prediction"] = {{"status": "configured"}}
    return {{**state, "artifacts": artifacts}}
# MLESTAR_COMPONENT:prediction:END
'''


create_fallback_project = write_fallback_project
