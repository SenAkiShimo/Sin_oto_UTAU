from sin_oto_utau.languages.base import AliasInfo, LanguageProfile


class GenericProfile(LanguageProfile):
    name = "generic"

    def analyze_alias(self, alias: str) -> AliasInfo:
        raw = alias.strip().lower()

        if " " in raw:
            syllable_type = "multi_part"
        elif "-" in raw:
            syllable_type = "transition"
        else:
            syllable_type = "unknown"

        return AliasInfo(
            language=self.name,
            alias=alias,
            initial=None,
            final=None,
            initial_type="unknown",
            final_type="unknown",
            syllable_type=syllable_type,
        )