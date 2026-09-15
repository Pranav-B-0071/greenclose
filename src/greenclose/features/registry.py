"""Feature registry: every feature declares (name, fn, deps, lookback_days, version, tier).

A feature with lookback L is NaN for the first L rows of a symbol and is never backfilled.
`model_input=False` marks context columns the decision layer needs but the model must not see.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from greenclose.features.panel import PanelContext

FeatureFn = Callable[["PanelContext"], Any]


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    name: str
    fn: FeatureFn
    tier: int  # 1 = stock-level (ranked cross-sectionally), 2 = market context (one value per day)
    lookback_days: int
    deps: tuple[str, ...] = ()
    version: str = "1"
    model_input: bool = True
    ranked: bool = True  # tier-1 only: also produce the cross-sectional rank column `<name>_rk`


class FeatureRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, FeatureSpec] = {}

    def register(self, spec: FeatureSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"feature {spec.name!r} registered twice")
        self._specs[spec.name] = spec

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def get(self, name: str) -> FeatureSpec:
        return self._specs[name]

    def names(self, *, tier: int | None = None, model_input: bool | None = None) -> list[str]:
        out = []
        for s in self.ordered():
            if tier is not None and s.tier != tier:
                continue
            if model_input is not None and s.model_input != model_input:
                continue
            out.append(s.name)
        return out

    def ordered(self) -> list[FeatureSpec]:
        """Dependency order (stable topological sort); raises on a missing or cyclic dependency."""
        done: dict[str, None] = {}
        visiting: set[str] = set()

        def visit(name: str) -> None:
            if name in done:
                return
            if name in visiting:
                raise ValueError(f"cyclic feature dependency at {name!r}")
            if name not in self._specs:
                raise ValueError(f"unknown feature dependency {name!r}")
            visiting.add(name)
            for d in self._specs[name].deps:
                visit(d)
            visiting.discard(name)
            done[name] = None

        for n in self._specs:
            visit(n)
        return [self._specs[n] for n in done]

    def lookback_max(self) -> int:
        return max((s.lookback_days for s in self._specs.values()), default=0)

    def version_hash(self) -> str:
        import hashlib

        payload = "|".join(f"{s.name}:{s.version}:{s.lookback_days}" for s in self.ordered())
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


REGISTRY = FeatureRegistry()


def feature(
    name: str,
    *,
    tier: int,
    lookback_days: int,
    deps: Sequence[str] = (),
    version: str = "1",
    model_input: bool = True,
    ranked: bool = True,
) -> Callable[[FeatureFn], FeatureFn]:
    def deco(fn: FeatureFn) -> FeatureFn:
        REGISTRY.register(
            FeatureSpec(
                name=name,
                fn=fn,
                tier=tier,
                lookback_days=lookback_days,
                deps=tuple(deps),
                version=version,
                model_input=model_input,
                ranked=ranked and tier == 1,
            )
        )
        return fn

    return deco
