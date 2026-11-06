"""Canonical Danbooru names and consistent text representations."""

import copy
import re
from functools import lru_cache
from collections.abc import Mapping
from typing import TypeVar

from .types import CATEGORIES, Score, Scores, TagInput, WDTagResult
from .tag_names import canonical_tag, parse_tags

_Result = TypeVar("_Result", bound=WDTagResult)

REGEX_RULE_FIELDS = (
    'remove_tags_from_full', 'remove_tags_from_region', 'exclude_from_region',
    'only_from_source', 'allowed_tags', 'full_image_only_tags',
)


def parse_regex(value):
    """One unchanged pattern per line; commas and spaces belong to patterns."""
    values = value.splitlines() if isinstance(value, str) else (value or [])
    if not isinstance(values, (list, tuple)) or any(not isinstance(v, str) for v in values):
        raise ValueError('Regex patterns must be a list of strings')
    return list(dict.fromkeys(v for v in values if v.strip()))


@lru_cache(maxsize=512)
def compile_regex(pattern):
    """Reuse compiled patterns across tag comparisons."""
    return re.compile(pattern)


class TagMatcher:
    """Literal union with explicit full-name regex patterns."""

    def __init__(self, literals=(), patterns=(), field='regex', source=None):
        self.literals = set(parse_tags(literals))
        self.patterns = parse_regex(patterns)
        self.compiled = []
        for index, pattern in enumerate(self.patterns, 1):
            try:
                self.compiled.append(compile_regex(pattern))
            except (re.error, OverflowError) as error:
                raise ValueError(f'{source or "global"}.{field} pattern {index}: {error}') from error

    def __bool__(self):
        return bool(self.literals or self.patterns)

    def __contains__(self, tag):
        tag = canonical_tag(tag)
        return tag in self.literals or any(pattern.fullmatch(tag) for pattern in self.compiled)

    def matches(self, tag):
        tag = canonical_tag(tag)
        matches = [{'kind': 'literal', 'value': tag}] if tag in self.literals else []
        matches.extend({'kind': 'regex', 'pattern': p.pattern}
                       for p in self.compiled if p.fullmatch(tag))
        return matches


def rule_matcher(rules, field, literals=None):
    return TagMatcher(getattr(rules, field, ()) if literals is None else literals,
                      getattr(rules, field + '_regex', ()), field + '_regex',
                      getattr(rules, 'name', None))


def parse_additions(value):
    """Accept tag or tag:score without losing assigned scores during UI import."""
    import math
    if isinstance(value, Mapping):
        pairs=value.items()
    else:
        pairs=[]
        for item in parse_tags(value):
            tag,colon,score=item.rpartition(':')
            if colon:
                try: number=float(score)
                except ValueError as error: raise ValueError('Manual tag score must be a number') from error
                pairs.append((tag,number))
            else: pairs.append((item,1.0))
    result={}
    for tag,score in pairs:
        if isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score) or not 0<=score<=1:
            raise ValueError('Manual tag score must be between 0 and 1')
        name=canonical_tag(tag)
        if not name:raise ValueError('Manual tag name cannot be empty')
        result[name]=max(result.get(name,score),score)
    return result


def normalized_scores(scores: Mapping[str, Score]) -> Scores:
    """Merge equivalent names without losing the maximum score."""
    result: Scores = {}
    for name, score in scores.items():
        tag = canonical_tag(name)
        if tag:
            result[tag] = max(result.get(tag, score), score)
    return result


def source_owners(configs, tags=()):
    owners = {}
    tags = {canonical_tag(tag) for tag in tags}
    for config in configs:
        matcher = rule_matcher(config, 'only_from_source')
        for tag in matcher.literals | {t for t in tags if t in matcher}:
            owners.setdefault(tag, set()).add(config.name)
    return owners


def restrict_allowed_tags(result, rules):
    allowed = rule_matcher(rules, 'allowed_tags')
    if not allowed:
        return result
    result = copy.deepcopy(result)
    for category in CATEGORIES:
        result[category] = {tag: score for tag, score in result[category].items() if tag in allowed}
    accepted = set(result["general"]) | set(result["character"])
    result["provenance"] = {tag: item for tag, item in result.get("provenance", {}).items() if tag in accepted}
    return format_result(result)


def enforce_owners(result, configs):
    """Prevent final manual prefix/suffix additions from bypassing source ownership."""
    owners = source_owners(configs, set(result['general']) | set(result['character']))
    if not owners:
        return result
    result = copy.deepcopy(result)
    for category in CATEGORIES:
        for tag in list(result[category]):
            origin = result.get("provenance", {}).get(tag, {})
            source = origin.get("region_source", origin.get("source", ""))
            name = source.rsplit(":", 1)[0] if ":" in source else ""
            if tag in owners and name not in owners[tag]:
                del result[category][tag]
                result.get("provenance", {}).pop(tag, None)
    return format_result(result, result.get("caption"))


def format_result(result: _Result, order: TagInput = None) -> _Result:
    """Rebuild both strings from the authoritative score dictionaries."""
    names = set(result["general"]) | set(result["character"])
    ordered = [tag for tag in parse_tags(order or []) if tag in names]
    ordered.extend(sorted(names - set(ordered)))
    result["caption"] = ", ".join(ordered)
    result["taglist"] = ", ".join(tag.replace("_", " ") for tag in ordered)
    return result


# Preserve the existing notebook calling convention.
# pylint: disable-next=R0913,R0917
def apply_tag_rules(
    result: _Result, remove: TagInput = (), add: Mapping[str, Score] | None = None,
    before: TagInput = (), after: TagInput = (), character_names: TagInput = (),
    remove_regex: TagInput = (),
) -> _Result:
    """Return a new result; explicit removal wins over manual additions."""
    output = copy.deepcopy(result)
    matcher = TagMatcher(remove, remove_regex, 'remove_tags_regex')
    candidates = set(normalized_scores(output['general'])) | set(normalized_scores(output['character']))
    candidates.update(canonical_tag(tag) for tag in (add or {}))
    candidates.update(parse_tags(before) + parse_tags(after))
    blocked = matcher.literals | {tag for tag in candidates if tag in matcher}
    for category in CATEGORIES:
        output[category] = {
            tag: score
            for tag, score in normalized_scores(output[category]).items()
            if tag not in blocked
        }
    provenance = output.setdefault("provenance", {})
    for tag in blocked:
        provenance.pop(tag, None)
    additions = {canonical_tag(tag): score for tag, score in (add or {}).items()}
    for tag in parse_tags(before) + parse_tags(after):
        additions.setdefault(tag, 1.0)
    known_characters = set(parse_tags(character_names))
    _add_manual_tags(output, additions, blocked, known_characters)
    suffix = [tag for tag in parse_tags(after) if tag not in parse_tags(before)]
    middle = sorted(
        (set(output["general"]) | set(output["character"])) - set(parse_tags(before)) - set(suffix)
    )
    return format_result(output, parse_tags(before) + middle + suffix)


def _add_manual_tags(
    output: WDTagResult, additions: Mapping[str, Score],
    blocked: set[str], known_characters: set[str],
) -> None:
    """Assign only new, unblocked manual tags to their declared category."""
    provenance = output.setdefault("provenance", {})
    for tag, score in additions.items():
        if not tag or tag in blocked or tag in output["general"] or (tag in output["character"]):
            continue
        category = "character" if tag in known_characters else "general"
        output[category][tag] = score
        provenance[tag] = {"source": "manual_rule", "assigned_score": score}
