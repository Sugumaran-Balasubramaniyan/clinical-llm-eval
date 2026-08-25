"""Benchmark configuration engine for Clinical LLM Evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


def _parse_yaml_scalar(val_str: str) -> Any:
    """Parse a scalar YAML value into appropriate Python types."""
    s = val_str.strip()
    if not s:
        return None

    # Strip inline comment if not enclosed in quotes
    if "#" in s:
        in_quote = False
        quote_char = ""
        clean_chars = []
        for ch in s:
            if ch in ('"', "'"):
                if not in_quote:
                    in_quote = True
                    quote_char = ch
                elif quote_char == ch:
                    in_quote = False
                    quote_char = ""
                clean_chars.append(ch)
            elif ch == "#" and not in_quote:
                break
            else:
                clean_chars.append(ch)
        s = "".join(clean_chars).strip()

    if not s:
        return None

    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    if s.lower() in ("true", "yes", "on"):
        return True
    if s.lower() in ("false", "no", "off"):
        return False
    if s.lower() in ("null", "none", "~"):
        return None

    # Inline flow list: [a, b, c]
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        items = []
        for part in inner.split(","):
            items.append(_parse_yaml_scalar(part.strip()))
        return items

    # Inline flow dict: {a: 1, b: 2}
    if s.startswith("{") and s.endswith("}"):
        inner = s[1:-1].strip()
        if not inner:
            return {}
        res = {}
        for part in inner.split(","):
            if ":" in part:
                k, v = part.split(":", 1)
                res[k.strip()] = _parse_yaml_scalar(v.strip())
        return res

    # Integer conversion
    try:
        if s.isdigit() or (s.startswith(("-", "+")) and s[1:].isdigit()):
            return int(s)
    except ValueError:
        pass

    # Float conversion
    try:
        return float(s)
    except ValueError:
        pass

    return s


def _parse_yaml_fallback(text: str) -> dict[str, Any]:
    """Clean built-in fallback YAML parser supporting nested dicts, lists, and scalar types."""
    clean_lines: list[tuple[int, str]] = []
    for line in text.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        clean_lines.append((indent, line.strip()))

    if not clean_lines:
        return {}

    def parse_block(idx: int, current_indent: int) -> tuple[Any, int]:
        if idx >= len(clean_lines):
            return {}, idx

        first_indent, first_line = clean_lines[idx]
        if first_line.startswith("- "):
            items = []
            while idx < len(clean_lines):
                indent, line = clean_lines[idx]
                if indent < current_indent and not line.startswith("- "):
                    break
                if line.startswith("- "):
                    content = line[2:].strip()
                    if not content:
                        val, idx = parse_block(idx + 1, indent + 2)
                        items.append(val)
                    elif ":" in content and not (content.startswith("{") or content.startswith("[")):
                        k, v = content.split(":", 1)
                        dict_item = {}
                        if v.strip():
                            dict_item[k.strip()] = _parse_yaml_scalar(v.strip())
                            idx += 1
                        else:
                            val, idx = parse_block(idx + 1, indent + 4)
                            dict_item[k.strip()] = val
                        items.append(dict_item)
                    else:
                        items.append(_parse_yaml_scalar(content))
                        idx += 1
                else:
                    break
            return items, idx
        else:
            result: dict[str, Any] = {}
            while idx < len(clean_lines):
                indent, line = clean_lines[idx]
                if indent < current_indent:
                    break
                if line.startswith("- "):
                    break
                if ":" in line:
                    k, v = line.split(":", 1)
                    key = k.strip()
                    val_str = v.strip()
                    if not val_str:
                        if idx + 1 < len(clean_lines):
                            next_indent, next_line = clean_lines[idx + 1]
                            if next_line.startswith("- ") and next_indent >= indent:
                                val, idx = parse_block(idx + 1, next_indent)
                                result[key] = val
                            elif next_indent > indent:
                                val, idx = parse_block(idx + 1, next_indent)
                                result[key] = val
                            else:
                                result[key] = None
                                idx += 1
                        else:
                            result[key] = None
                            idx += 1
                    else:
                        result[key] = _parse_yaml_scalar(val_str)
                        idx += 1
                else:
                    idx += 1
            return result, idx

    parsed, _ = parse_block(0, clean_lines[0][0])
    return parsed if isinstance(parsed, dict) else {}


def _format_yaml_scalar(val: Any) -> str:
    """Format scalar value for YAML serialization."""
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float)):
        return str(val)
    s = str(val)
    if not s:
        return '""'
    if "\n" in s or ":" in s or "#" in s or s.startswith(("-", "[", "{", "@", "*", "&", "!", "%", "|", ">", "'", '"')):
        escaped = s.replace('"', '\\"')
        return f'"{escaped}"'
    return s


def _dump_yaml_fallback(data: dict[str, Any], indent_level: int = 0) -> str:
    """Clean built-in fallback YAML serializer."""
    lines = []
    indent = "  " * indent_level
    for k, v in data.items():
        if isinstance(v, dict):
            if not v:
                lines.append(f"{indent}{k}: {{}}")
            else:
                lines.append(f"{indent}{k}:")
                lines.append(_dump_yaml_fallback(v, indent_level + 1))
        elif isinstance(v, list):
            if not v:
                lines.append(f"{indent}{k}: []")
            else:
                lines.append(f"{indent}{k}:")
                for item in v:
                    if isinstance(item, dict):
                        nested = _dump_yaml_fallback(item, indent_level + 2).lstrip()
                        lines.append(f"{indent}  - {nested}")
                    else:
                        lines.append(f"{indent}  - {_format_yaml_scalar(item)}")
        else:
            lines.append(f"{indent}{k}: {_format_yaml_scalar(v)}")
    return "\n".join(lines)


@dataclass
class BenchmarkConfig:
    """Clinical LLM Benchmark Suite Configuration."""

    name: str = "Clinical Benchmark Suite"
    datasets: list[str] = field(default_factory=lambda: ["sample_medqa"])
    models: list[str] = field(default_factory=lambda: ["mistral"])
    n_samples: int = 50
    concurrency: int = 5
    judge_provider: str = "openai"
    judge_model: Optional[str] = None
    output_dir: str = "reports/output"
    temperature: float = 0.2
    max_tokens: int = 256
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize configuration attributes."""
        if self.datasets is None:
            self.datasets = ["sample_medqa"]
        elif isinstance(self.datasets, str):
            self.datasets = [self.datasets]
        elif not isinstance(self.datasets, list):
            self.datasets = list(self.datasets)

        if self.models is None:
            self.models = ["mistral"]
        elif isinstance(self.models, str):
            self.models = [self.models]
        elif not isinstance(self.models, list):
            self.models = list(self.models)

        self.n_samples = int(self.n_samples)
        self.concurrency = max(1, int(self.concurrency))
        self.temperature = float(self.temperature)
        self.max_tokens = int(self.max_tokens)
        if not isinstance(self.metadata, dict):
            self.metadata = dict(self.metadata) if self.metadata else {}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkConfig:
        """Instantiate BenchmarkConfig from dictionary."""
        valid_fields = {
            "name",
            "datasets",
            "models",
            "n_samples",
            "concurrency",
            "judge_provider",
            "judge_model",
            "output_dir",
            "temperature",
            "max_tokens",
            "metadata",
        }
        kwargs = {}
        extra_meta = {}
        for k, v in data.items():
            if k in valid_fields:
                kwargs[k] = v
            else:
                extra_meta[k] = v

        if extra_meta:
            existing_meta = kwargs.get("metadata") or {}
            if isinstance(existing_meta, dict):
                kwargs["metadata"] = {**extra_meta, **existing_meta}
            else:
                kwargs["metadata"] = extra_meta

        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> BenchmarkConfig:
        """Load and parse YAML configuration file with pyyaml or built-in fallback parser."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        data: Any = None
        if yaml is not None:
            try:
                data = yaml.safe_load(content)
            except Exception:
                data = None

        if data is None or not isinstance(data, dict):
            data = _parse_yaml_fallback(content)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid YAML content in {file_path}: root must be a mapping/dictionary")

        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()

        if yaml is not None:
            try:
                content = yaml.safe_dump(data, default_flow_style=False, sort_keys=False)
            except Exception:
                content = _dump_yaml_fallback(data)
        else:
            content = _dump_yaml_fallback(data)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    @classmethod
    def create_default_config(cls, path: str = "configs/benchmark_default.yaml") -> BenchmarkConfig:
        """Write a template YAML configuration file and return the config instance."""
        config = cls()
        config.to_yaml(path)
        return config
