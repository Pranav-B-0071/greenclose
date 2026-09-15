"""Model artefact registry and the pinned model the forward runner uses.

A registered artefact records everything needed to explain a prediction later: the git SHA, the config
hash, the data snapshot, the seed, the training window and the feature-registry version. `pinned.json`
names the one artefact the nightly forward log is allowed to use, so a prediction can never silently
come from a model that was retrained after the fact.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from greenclose.ops.provenance import utc_now_iso


@dataclass
class ModelCard:
    model_version: str
    head_kind: str
    target: str
    seed: int
    git_sha: str
    config_hash: str
    snapshot_id: str
    feature_registry: str
    model_columns: list[str]
    train_start: str
    train_end: str
    trained_at: str = field(default_factory=utc_now_iso)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True, default=str)


class ModelRegistry:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, model_version: str) -> Path:
        return self.root / model_version

    def save(self, card: ModelCard, model: Any, preprocessor: Any) -> Path:
        d = self.path(card.model_version)
        d.mkdir(parents=True, exist_ok=True)
        (d / "card.json").write_text(card.to_json(), encoding="utf-8")
        with (d / "model.pkl").open("wb") as fh:
            pickle.dump({"model": model, "preprocessor": preprocessor}, fh, protocol=5)
        return d

    def load(self, model_version: str) -> tuple[ModelCard, Any, Any]:
        d = self.path(model_version)
        card = ModelCard(**json.loads((d / "card.json").read_text(encoding="utf-8")))
        with (d / "model.pkl").open("rb") as fh:
            blob = pickle.load(fh)  # noqa: S301 - our own artefact, written by this package
        return card, blob["model"], blob["preprocessor"]

    # ---- the pinned artefact

    @property
    def pin_file(self) -> Path:
        return self.root / "pinned.json"

    def pin(self, model_version: str, *, reason: str = "") -> None:
        self.pin_file.write_text(
            json.dumps(
                {"model_version": model_version, "pinned_at": utc_now_iso(), "reason": reason}, indent=2
            ),
            encoding="utf-8",
        )

    def pinned_version(self) -> str | None:
        if not self.pin_file.exists():
            return None
        return str(json.loads(self.pin_file.read_text(encoding="utf-8"))["model_version"])

    def load_pinned(self) -> tuple[ModelCard, Any, Any]:
        v = self.pinned_version()
        if v is None:
            raise FileNotFoundError(
                f"No pinned model in {self.root}. Run `greenclose pin-model` after the ladder finishes."
            )
        return self.load(v)

    def list_versions(self) -> list[str]:
        return sorted(d.name for d in self.root.iterdir() if d.is_dir() and (d / "card.json").exists())


def make_version(head_kind: str, snapshot_id: str, seed: int, asof: date | str) -> str:
    stamp = asof.isoformat() if isinstance(asof, date) else str(asof)
    return f"{head_kind}__{snapshot_id}__s{seed}__{stamp}"
