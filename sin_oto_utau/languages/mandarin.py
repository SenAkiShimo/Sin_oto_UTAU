from sin_oto_utau.languages.base import AliasInfo, LanguageProfile


INITIAL_TYPES = {
    "zh": "affricate",
    "ch": "aspirated_affricate",
    "sh": "fricative",

    "b": "plosive",
    "p": "aspirated_plosive",
    "m": "nasal",
    "f": "fricative",

    "d": "plosive",
    "t": "aspirated_plosive",
    "n": "nasal",
    "l": "liquid",

    "g": "plosive",
    "k": "aspirated_plosive",
    "h": "fricative",

    "j": "affricate",
    "q": "aspirated_affricate",
    "x": "fricative",

    "r": "fricative",
    "z": "affricate",
    "c": "aspirated_affricate",
    "s": "fricative",

    "y": "glide",
    "w": "glide",
}


FINAL_TYPES = {
    "a": "simple_vowel",
    "o": "simple_vowel",
    "e": "simple_vowel",
    "i": "simple_vowel",
    "u": "simple_vowel",
    "v": "simple_vowel",
    "ü": "simple_vowel",

    "ai": "diphthong",
    "ei": "diphthong",
    "ao": "diphthong",
    "ou": "diphthong",
    "ia": "diphthong",
    "ie": "diphthong",
    "ua": "diphthong",
    "uo": "diphthong",
    "ve": "diphthong",

    "an": "nasal_final",
    "en": "nasal_final",
    "in": "nasal_final",
    "un": "nasal_final",
    "vn": "nasal_final",

    "ang": "nasal_final",
    "eng": "nasal_final",
    "ing": "nasal_final",
    "ong": "nasal_final",

    "ian": "nasal_final",
    "uan": "nasal_final",
    "van": "nasal_final",

    "iang": "nasal_final",
    "uang": "nasal_final",
    "iong": "nasal_final",
}


class MandarinProfile(LanguageProfile):
    name = "mandarin"

    def analyze_alias(self, alias: str) -> AliasInfo:
        raw = alias.strip().lower()

        clean = raw
        clean = clean.replace("-", "")
        clean = clean.replace("_", "")
        clean = clean.replace(" ", "")
        clean = clean.replace("yu", "v")
        clean = clean.replace("ü", "v")

        initial = None
        final = clean

        for candidate in sorted(INITIAL_TYPES.keys(), key=len, reverse=True):
            if clean.startswith(candidate) and len(clean) > len(candidate):
                initial = candidate
                final = clean[len(candidate):]
                break

        if initial is None:
            initial_type = "none"
            syllable_type = "V"
        else:
            initial_type = INITIAL_TYPES.get(initial, "unknown")
            syllable_type = "CV"

        final_type = FINAL_TYPES.get(final, "unknown")

        return AliasInfo(
            language=self.name,
            alias=alias,
            initial=initial,
            final=final,
            initial_type=initial_type,
            final_type=final_type,
            syllable_type=syllable_type,
        )