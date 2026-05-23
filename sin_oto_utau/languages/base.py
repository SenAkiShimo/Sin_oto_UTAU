from dataclasses import dataclass


@dataclass
class AliasInfo:
    language: str
    alias: str
    initial: str | None
    final: str | None
    initial_type: str
    final_type: str
    syllable_type: str


class LanguageProfile:
    name = "generic"

    def analyze_alias(self, alias: str) -> AliasInfo:
        raise NotImplementedError