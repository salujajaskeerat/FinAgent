"""Load and validate declarative persona policies."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import Field

from finagent.contracts.api import Persona, StrictModel


class PersonaPolicy(StrictModel):
    """Evidence and presentation requirements for one persona."""

    persona: Persona
    label: str
    description: str
    horizon: str
    required_metrics: list[str] = Field(min_length=1)
    event_kinds: list[str] = Field(default_factory=list)
    required_sections: list[str] = Field(min_length=1)


class PersonaPolicyStore:
    """In-memory collection loaded from a versioned YAML resource."""

    def __init__(self, policies: dict[Persona, PersonaPolicy]) -> None:
        self._policies = policies

    @classmethod
    def load(cls, path: Path | None = None) -> PersonaPolicyStore:
        """Load policies from YAML.

        Parameters
        ----------
        path
            Optional configuration path. The packaged policy file is used by default.

        Returns
        -------
        PersonaPolicyStore
            Validated policies indexed by persona.
        """
        config_path = path or Path(__file__).parents[1] / "config" / "personas.yaml"
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("persona configuration must be a mapping")
        policies: dict[Persona, PersonaPolicy] = {}
        for key, value in raw.items():
            persona = Persona(key)
            policies[persona] = PersonaPolicy(persona=persona, **value)
        missing = set(Persona) - set(policies)
        if missing:
            raise ValueError(f"missing persona policies: {sorted(missing)}")
        return cls(policies)

    def get(self, persona: Persona) -> PersonaPolicy:
        """Return a policy for a validated persona."""
        return self._policies[persona]

    def all(self) -> list[PersonaPolicy]:
        """Return all policies in public enum order."""
        return [self._policies[persona] for persona in Persona]
