"""Ontology normalization for knowledge graph entities and relations.

Turns raw extracted entities and predicates into a normalized view:

* **Official names** — a canonical name per entity, plus the surface forms
  that were folded into it (aliases).
* **Facts** — short, quotable statements tying an entity to its relation.
* **Relation normalization** — free-form predicates are mapped onto a
  controlled set of relation types, so a graph can be compared across sites.

Everything here is deterministic and offline: it works on the extracted graph
without an LLM round trip, and is used to power the Report tab.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Sequence


# --------------------------------------------------------------------------
# Relation type normalization
# --------------------------------------------------------------------------

# Ordered most-specific first: the first matching pattern wins.
RELATION_TYPE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("OPERATES", (
        r"operat", r"run", r"manage", r"adminis", r"conduct", r"perform",
    )),
    ("OFFERS", (
        r"offer", r"provide", r"supply", r"sell", r"carry", r"feature",
        r"supplement", r"grant", r"give",
    )),
    ("MEMBER_OF", (
        r"\bmember", r"belong", r"\bpart of\b", r"affiliat", r"\bsubset",
        r"included? in\b", r"contains?",
    )),
    ("BRANDED_BY", (
        r"brand", r"manufactur", r"produce", r"make", r"\bown\b",
    )),
    ("LOCATED_IN", (
        r"locat", r"address", r"headquarter", r"based in", r"situated",
    )),
    ("CONTACT_POINT", (
        r"contact", r"\bcall\b", r"phone", r"\bemail\b", r"helpline", r"inquiry",
    )),
    ("DISCOUNTS", (
        r"discount", r"coupon", r"\bsale\b", r"\bdeal\b", r"promo", r"rebate",
        r"cash ?back", r"deduct", r"\bsave\b", r"할인", r"쿠폰", r"적립", r"혜택",
    )),
    ("PRICED_AT", (
        r"price", r"cost", r"\bpay\b", r"fee", r"charge", r"worth", r"\brate\b",
    )),
    ("SHIPS_TO", (
        r"ship", r"deliver", r"courier",
    )),
    ("INCLUDES", (
        r"include", r"contain", r"consist of", r"comprise",
    )),
    ("RECOMMENDS", (
        r"recommend", r"suggest", r"advise",
    )),
    ("LINKS_TO", (
        r"link", r"connect", r"refer", r"\burl\b", r"redirect", r"\bhref\b",
    )),
    ("REPRESENTS", (
        r"represent", r"spokes", r"proxy", r"on behalf of",
    )),
    ("HAS_ATTRIBUTE", (
        r"has ", r"have ", r"with ", r"attribute", r"property", r"feature",
        r"possess", r"is called", r"named",
    )),
    ("CREATED_BY", (
        r"creat", r"found", r"establish", r"launch", r"develop", r"invent",
    )),
    ("WORKS_FOR", (
        r"work", r"employ", r"hire", r"join", r"\bstaff\b", r"직장", r"소속",
    )),
    ("APPLIES_TO", (
        r"appl", r"valid", r"restrict", r"condition", r"requir", r"eligible",
        r"qualif", r"exclude", r"except",
    )),
    ("PARTNERS_WITH", (
        r"partner", r"collaborat", r"alliance", r"cooperat", r"joint", r"제휴",
    )),
    ("SCHEDULED_FOR", (
        r"schedul", r"plan", r"event", r"until", r"from .* to ", r"기간",
        r"campaign", r"\bopen\b", r"\bstart\b", r"\bend\b",
    )),
    ("CUSTOMER_OF", (
        r"custom", r"buyer", r"subscriber", r"member of", r"구매", r"가입",
    )),
)

_RELATION_TYPE_COMPILED: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (relation_type, re.compile("|".join(patterns), re.IGNORECASE))
    for relation_type, patterns in RELATION_TYPE_RULES
)

# Relation types that carry the most weight when ranking a fact.
_RELATION_PRIORITY: tuple[str, ...] = (
    "OPERATES",
    "OFFERS",
    "BRANDED_BY",
    "MEMBER_OF",
    "DISCOUNTS",
    "CREATED_BY",
    "WORKS_FOR",
    "PARTNERS_WITH",
    "LOCATED_IN",
    "SHIPS_TO",
    "PRICED_AT",
    "APPLIES_TO",
    "CUSTOMER_OF",
    "RECOMMENDS",
    "LINKS_TO",
    "CONTACT_POINT",
    "HAS_ATTRIBUTE",
    "INCLUDES",
    "REPRESENTS",
    "SCHEDULED_FOR",
)


def normalize_relation_type(predicate: str) -> str:
    """Map a free-form predicate onto a controlled relation type."""
    text = (predicate or "").strip()
    if not text:
        return "RELATED_TO"
    for relation_type, pattern in _RELATION_TYPE_COMPILED:
        if pattern.search(text):
            return relation_type
    return "RELATED_TO"


# --------------------------------------------------------------------------
# Entity name normalization
# --------------------------------------------------------------------------

# Corporate/legal suffixes stripped when deriving a canonical short name.
_LEGAL_SUFFIXES: tuple[str, ...] = (
    "주식회사", "(주)", "㈜", "inc.", "inc", "co.,ltd.", "co., ltd.",
    "co.ltd", "ltd.", "ltd", "corp.", "corp", "corporation", "company",
    "llc", "gmbh", "co.",
)

# Noise words that should not make two otherwise-equal names differ.
_ENTITY_NOISE: tuple[str, ...] = (
    "home", "main", "index", "www", "http", "https", "html", "php", "asp",
    "site", "page",
)

_PUNCTUATION = re.compile(r"[\s\-_·.,()\[\]{}'\"!?:;/\\|]+")
_WHITESPACE = re.compile(r"\s+")


def _normalize_text(value: str) -> str:
    """NFKC-normalize and collapse whitespace."""
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKC", value)
    return _WHITESPACE.sub(" ", normalized).strip()


def build_entity_key(label: str) -> str:
    """Build a comparison key for an entity label.

    Two labels share a key when they differ only by case, spacing, punctuation,
    legal suffixes, or trailing noise words.
    """
    text = _normalize_text(label).lower()
    if not text:
        return ""

    for suffix in _LEGAL_SUFFIXES:
        text = re.sub(rf"\s*{re.escape(suffix)}\s*$", "", text)

    text = _PUNCTUATION.sub("", text)

    for noise in _ENTITY_NOISE:
        text = re.sub(rf"{re.escape(noise)}$", "", text)

    return text.strip()


def build_official_name(label: str) -> str:
    """Derive the display-ready official name for an entity label."""
    text = _normalize_text(label)
    if not text:
        return ""

    # Keep the surface form but drop a trailing legal suffix.
    for suffix in ("(주)", "㈜", "주식회사"):
        stripped = text.rstrip()
        if stripped.endswith(suffix):
            return stripped[: -len(suffix)].strip() or stripped

    return text



# --------------------------------------------------------------------------
# Core entity tracking
# --------------------------------------------------------------------------

# The entities a report is usually written about. Each entry lists the
# surface forms that are considered the same concept, plus the canonical
# name to use for it.
CORE_ENTITY_GROUPS: tuple[dict[str, Any], ...] = (
    {
        "id": "ssg7_membership",
        "canonical": "쓱7클럽",
        "label": "쓱7클럽 (SSG7CLUB)",
        "label_en": "SSG7 Membership",
        "surfaceForms": (
            "쓱7클럽",
            "쓱세븐클럽",
            "ssg7club",
            "ssg7클럽",
            "ssg 7 클럽",
            "ssg7",
        ),
        # A surface form only counts as the concept itself when the label is
        # not a longer product name that merely contains the brand string.
        "exactForms": (
            "쓱7클럽",
            "쓱세븐클럽",
            "ssg7club",
            "ssg7클럽",
            "ssg 7 클럽",
            "ssg7",
        ),
    },
)

_CORE_SURFACE_REGEXES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (surface, re.compile(rf"^{re.escape(surface)}$", re.IGNORECASE))
    for group in CORE_ENTITY_GROUPS
    for surface in group["exactForms"]
)

_BRAND_TOKEN_REGEX = re.compile(
    r"쓱\s*7\s*클럽|쓱\s*세븐\s*클럽|쓱세븐|ssg\s*7\s*클럽|ssg\s*7\s*club|ssg7club",
    re.IGNORECASE,
)


def _is_exact_surface(label: str, group: dict[str, Any]) -> bool:
    """True when the label is exactly one of the group's surface forms."""
    normalized = _normalize_text(label)
    for surface in group["exactForms"]:
        if normalized.lower() == surface.lower():
            return True
    return False


def _mentions_brand_token(label: str) -> bool:
    """True when the label contains a brand token anywhere (e.g. a product)."""
    return bool(_BRAND_TOKEN_REGEX.search(label or ""))


def build_core_entity_report(
    entities: Sequence[str],
    relations: Sequence[tuple[str, str, str]],
) -> dict[str, Any]:
    """Track how the core entities of a report are used in the graph.

    The question this answers is "are the key entities properly normalized
    and actually used?", so it separates:

    * the entity itself (exact surface forms),
    * product names that merely contain the brand string,
    * how often the entity appears as a relation subject or object.
    """
    entity_set = list(dict.fromkeys(entities))
    entity_index = {entity for entity in entity_set}

    groups_out: list[dict[str, Any]] = []

    for group in CORE_ENTITY_GROUPS:
        exact_hits = [entity for entity in entity_set if _is_exact_surface(entity, group)]
        # Names that contain the brand but are longer product titles.
        containing = [
            entity
            for entity in entity_set
            if _mentions_brand_token(entity) and entity not in exact_hits
        ]

        # Which surface forms actually appear, and how often.
        form_counts: Counter[str] = Counter()
        for entity in exact_hits:
            normalized = _normalize_text(entity)
            for surface in group["surfaceForms"]:
                if normalized.lower() == surface.lower():
                    form_counts[surface] += 1
                    break
            else:
                form_counts[normalized] += 1

        # Usage: as subject / as object, and the relation types involved.
        as_subject: Counter[str] = Counter()
        as_object: Counter[str] = Counter()
        relation_types: Counter[str] = Counter()
        neighbours: Counter[str] = Counter()
        raw_pairs: list[dict[str, str]] = []

        for subject, predicate, obj in relations:
            relation_type = normalize_relation_type(predicate)
            if subject in exact_hits:
                as_subject[relation_type] += 1
                neighbours[obj] += 1
                raw_pairs.append(
                    {"subject": subject, "relationType": relation_type,
                     "predicate": predicate, "object": obj}
                )
            if obj in exact_hits:
                as_object[relation_type] += 1
                neighbours[subject] += 1
                raw_pairs.append(
                    {"subject": subject, "relationType": relation_type,
                     "predicate": predicate, "object": obj}
                )

        for pair in raw_pairs:
            relation_types[pair["relationType"]] += 1

        total_mentions = sum(as_subject.values()) + sum(as_object.values())
        distinct_forms = len(form_counts)

        # Is the brand consistently represented by one canonical label?
        normalized_keys = {build_entity_key(entity) for entity in exact_hits}
        is_consistent = len(normalized_keys) <= 1

        groups_out.append(
            {
                "id": group["id"],
                "canonical": group["canonical"],
                "label": group["label"],
                "labelEn": group["label_en"],
                "found": bool(exact_hits),
                "exactEntities": sorted(exact_hits),
                "surfaceFormCounts": [
                    {"surface": surface, "count": count}
                    for surface, count in form_counts.most_common()
                ],
                "distinctSurfaceForms": distinct_forms,
                "isConsistent": is_consistent,
                "mentionAsSubject": sum(as_subject.values()),
                "mentionAsObject": sum(as_object.values()),
                "totalMentions": total_mentions,
                "relationTypeCounts": [
                    {"type": relation_type, "count": count}
                    for relation_type, count in relation_types.most_common()
                ],
                "subjectByType": [
                    {"type": relation_type, "count": count}
                    for relation_type, count in as_subject.most_common()
                ],
                "objectByType": [
                    {"type": relation_type, "count": count}
                    for relation_type, count in as_object.most_common()
                ],
                "topNeighbours": [
                    {"label": label, "count": count}
                    for label, count in neighbours.most_common(12)
                ],
                "relations": raw_pairs[:40],
                "containingProductCount": len(containing),
                "containingProducts": sorted(containing)[:15],
            }
        )

    tracked_ids = {
        entity
        for group in CORE_ENTITY_GROUPS
        for entity in group["exactForms"]
    }
    tracked = [
        entity for entity in entity_set
        if _normalize_text(entity).lower() in {form.lower() for form in tracked_ids}
    ]

    return {
        "groups": groups_out,
        "trackedEntityCount": len(tracked),
        "totalEntityCount": len(entity_set),
        "trackedShare": round(len(tracked) / len(entity_set), 4) if entity_set else 0,
        "covered": any(group["found"] for group in groups_out),
    }


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def build_ontology_stats(
    entities: Sequence[str],
    relations: Sequence[tuple[str, str, str]],
) -> dict[str, Any]:
    """Build the normalization statistics shown in the Report tab.

    Returns canonical-name mappings, alias folds, and relation-type counts.
    """
    entity_set = list(dict.fromkeys(entities))

    # Group labels that normalize to the same key.
    groups: dict[str, list[str]] = defaultdict(list)
    for entity in entity_set:
        key = build_entity_key(entity)
        if not key:
            continue
        groups[key].append(entity)

    canonical_entries: list[dict[str, Any]] = []
    alias_folds: list[dict[str, Any]] = []
    merged_count = 0

    for members in groups.values():
        ordered = sorted(members, key=lambda value: (-len(value), value.lower()))
        canonical = build_official_name(ordered[0])
        aliases = [build_official_name(member) for member in ordered[1:]]
        aliases = [alias for alias in aliases if alias and alias != canonical]

        if aliases:
            merged_count += len(aliases)
            alias_folds.append(
                {
                    "canonical": canonical,
                    "aliases": aliases,
                    "count": len(aliases),
                }
            )

        canonical_entries.append(
            {
                "canonical": canonical,
                "sourceLabels": ordered,
                "aliasCount": len(aliases),
                "relationCount": 0,
            }
        )

    # Relation type normalization.
    relation_type_counts: Counter[str] = Counter()
    predicate_to_type: dict[str, str] = {}
    for _, predicate, _ in relations:
        relation_type = normalize_relation_type(predicate)
        relation_type_counts[relation_type] += 1
        predicate_to_type.setdefault(predicate, relation_type)

    # Facts: relations expressed with canonical names.
    canonical_by_key: dict[str, str] = {}
    for entry in canonical_entries:
        canonical_by_key[build_entity_key(entry["canonical"])] = entry["canonical"]

    facts: list[dict[str, Any]] = []
    relation_counts_by_canonical: Counter[str] = Counter()
    for subject, predicate, obj in relations:
        subject_canonical = canonical_by_key.get(build_entity_key(subject), subject)
        object_canonical = canonical_by_key.get(build_entity_key(obj), obj)
        relation_counts_by_canonical[subject_canonical] += 1
        facts.append(
            {
                "subject": subject_canonical,
                "predicate": predicate,
                "relationType": predicate_to_type.get(predicate, "RELATED_TO"),
                "object": object_canonical,
                "subjectRaw": subject,
                "objectRaw": obj,
            }
        )

    fact_counts = Counter(
        (fact["subject"], fact["relationType"], fact["object"]) for fact in facts
    )
    for entry in canonical_entries:
        entry["relationCount"] = relation_counts_by_canonical.get(entry["canonical"], 0)

    # Rank facts: relation priority first, then how often the fact repeats.
    priority_index = {
        relation_type: index for index, relation_type in enumerate(_RELATION_PRIORITY)
    }
    ranked_facts = sorted(
        facts,
        key=lambda fact: (
            priority_index.get(fact["relationType"], len(_RELATION_PRIORITY)),
            -fact_counts[(fact["subject"], fact["relationType"], fact["object"])],
            fact["subject"].lower(),
        ),
    )

    canonical_entries.sort(
        key=lambda entry: (
            -entry["relationCount"],
            -entry["aliasCount"],
            entry["canonical"].lower(),
        )
    )

    return {
        "entityCount": len(entity_set),
        "canonicalCount": len(canonical_entries),
        "aliasCount": merged_count,
        "relationCount": len(relations),
        "relationTypeCount": len(relation_type_counts),
        "coreEntities": build_core_entity_report(entity_set, relations),
        "canonicalEntities": canonical_entries[:60],
        "aliasFolds": sorted(
            alias_folds, key=lambda item: (-item["count"], item["canonical"])
        )[:40],
        "relationTypes": [
            {
                "type": relation_type,
                "count": count,
                "share": round(count / len(relations), 4) if relations else 0,
                "predicates": sorted(
                    {
                        predicate
                        for predicate, mapped in predicate_to_type.items()
                        if mapped == relation_type
                    }
                )[:12],
            }
            for relation_type, count in relation_type_counts.most_common()
        ],
        "facts": ranked_facts[:60],
        "predicateMap": dict(sorted(predicate_to_type.items())),
    }


def build_relation_program(
    entities: Sequence[str],
    relations: Sequence[tuple[str, str, str]],
    *,
    max_nodes: int = 40,
) -> dict[str, Any]:
    """Build the entity-relation graph program data for the Report tab.

    Returns nodes, edges, and a ready-to-use adjacency listing so the graph
    can be re-laid-out or traversed without the main canvas.
    """
    entity_set = set(entities)
    for subject, _, obj in relations:
        entity_set.add(subject)
        entity_set.add(obj)

    degree: Counter[str] = Counter()
    adjacency: dict[str, list[str]] = defaultdict(list)
    typed_edges: list[dict[str, Any]] = []

    for subject, predicate, obj in relations:
        degree[subject] += 1
        degree[obj] += 1
        adjacency[subject].append(obj)
        adjacency[obj].append(subject)
        typed_edges.append(
            {
                "source": subject,
                "predicate": predicate,
                "target": obj,
                "relationType": normalize_relation_type(predicate),
            }
        )

    ranked_entities = sorted(entity_set, key=lambda value: (-degree[value], value.lower()))
    selected = ranked_entities[:max_nodes]
    selected_set = set(selected)

    nodes = [
        {
            "id": entity,
            "label": build_official_name(entity),
            "degree": degree[entity],
            "officialName": build_official_name(entity),
            "surfaceForms": sorted(
                {
                    build_official_name(label)
                    for label in entity_set
                    if build_entity_key(label) == build_entity_key(entity)
                }
            )[:5],
        }
        for entity in selected
    ]

    edges = [
        edge
        for edge in typed_edges
        if edge["source"] in selected_set and edge["target"] in selected_set
    ]

    return {
        "nodes": nodes,
        "edges": edges,
        "truncated": max(0, len(ranked_entities) - max_nodes),
        "totalNodes": len(entity_set),
        "totalEdges": len(typed_edges),
        "adjacency": {
            entity: sorted(set(adjacency.get(entity, ())), key=lambda value: value.lower())
            for entity in selected
        },
    }
