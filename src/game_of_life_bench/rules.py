from dataclasses import dataclass


@dataclass(frozen=True)
class LifeRule:
    birth: frozenset[int]
    survive: frozenset[int]
    notation: str


def parse_rule(rule_text: str) -> LifeRule:
    text = rule_text.strip().upper()
    parts = text.split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid rule '{rule_text}'. Expected format like B3/S23.")

    birth_part, survive_part = parts
    if not birth_part.startswith("B") or not survive_part.startswith("S"):
        raise ValueError(f"Invalid rule '{rule_text}'. Expected format like B3/S23.")

    birth = frozenset(_parse_counts(birth_part[1:], rule_text))
    survive = frozenset(_parse_counts(survive_part[1:], rule_text))
    return LifeRule(birth=birth, survive=survive, notation=text)


def _parse_counts(counts_text: str, original_rule: str) -> list[int]:
    counts: list[int] = []
    for char in counts_text:
        if not char.isdigit():
            raise ValueError(f"Invalid rule '{original_rule}'. Neighbor counts must be digits.")
        value = int(char)
        if value > 8:
            raise ValueError(f"Invalid rule '{original_rule}'. Neighbor counts must be in 0..8.")
        counts.append(value)
    return counts
