"""High fidelity visualization utilities for kngraph knowledge graphs."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable
import colorsys
import webbrowser

from kngraph.models import Graph


# Heuristic fallback clusters applied at render time when the stored graph
# has no entity_clusters / edge_clusters (e.g. SEMHASH-only deduplication).
# Each rule is (cluster_id, match_function).
def _heuristic_entity_cluster(label: str) -> str | None:
    text = (label or "").strip()
    if not text:
        return None
    lowered = text.lower()

    if re.fullmatch(r"[\d\s,./~\-–—()%$원개월년일화수목금토:\-+]+", text) or re.search(
        r"\d\s*(원|%|개월|년|일|회|건|만원|천원)", text
    ):
        return "날짜·금액"
    if re.search(r"\d{1,4}[/.\-]\d{1,2}([/.\-]\d{1,2})?|\(\s*[월화수목금토일]\s*\)|~", text) and re.search(r"\d", text):
        return "날짜·금액"
    if re.search(r"쿠폰|할인|적립|캐시백|혜택|할인쿠폰|쿠폰팩", text):
        return "쿠폰·혜택"
    if re.search(r"쓱7클럽|ssg7club|7club|멤버십|membership", lowered):
        return "쓱7클럽"
    if re.search(r"ssg|에스에스지|신세계|이마트|g마켓|옥션|쿠팡", lowered):
        return "SSG·계열사"
    if re.search(r"티빙|tving|넷플릭스|유튜브|ott", lowered):
        return "제휴·구독"
    if re.search(r"cj|풀무원|매일유업|남양|오뚜기|lg생활건강|신세계푸드|햇반", text):
        return "입점 브랜드·상품"
    if re.search(r"event\s*\d+|이벤트|응모|당첨|경품|혜택\d+", lowered):
        return "이벤트"
    if re.search(r"카드|결제|money|머니|페이|포인트|마일리지", lowered):
        return "결제·포인트"
    if re.search(r"배송|새벽배송|장보기|장바구니|주문|반품|교환", text):
        return "주문·배송"
    if re.search(r"고객센터|1577|1644|문의|상담|cs|약관|정책|개인정보|동의|제공", text):
        return "고객지원·약관"
    if re.search(r"인스타|페이스북|유튜브|블로그|sns|팔로우", lowered):
        return "SNS·채널"
    if re.search(r"앱|어플|푸시|알림|알람|다운로드|설치", text):
        return "앱·알림"
    if re.search(r"http|www\.|\.com|\.kr|\.co", lowered):
        return "링크·채널"
    return None


def _heuristic_edge_cluster(predicate: str) -> str | None:
    text = (predicate or "").strip().lower()
    if not text:
        return None
    if re.search(r"쿠폰|할인|적립|cashback|discount|coupon|offers_coupon|gives_cashback", text):
        return "혜택 제공"
    if re.search(r"part of|member|belongs|includes|contains|is part", text):
        return "소속·포함"
    if re.search(r"provides|offers|operates|serves|service", text):
        return "제공·운영"
    if re.search(r"requires|required|applies|condition|eligible|has_expiry|has_value", text):
        return "조건·자격"
    if re.search(r"links|url|hosted|registered|address|contact|social|has_social", text):
        return "연락·링크"
    if re.search(r"alias|representative|corresponds|associated", text):
        return "별칭·연관"
    return None


def _apply_heuristic_clusters(
    entity_clusters: dict[str, Any] | None,
    edge_clusters: dict[str, Any] | None,
    entities: list[str],
    relations: list[tuple[str, str, str]],
) -> tuple[dict[str, set[str]], dict[str, set[str]], bool]:
    """Fill missing clusters with heuristic rules. Returns (entity, edge, used_fallback)."""
    entity_clusters = dict(entity_clusters or {})
    edge_clusters = dict(edge_clusters or {})
    used_fallback = False

    if not entity_clusters and entities:
        grouped: dict[str, set[str]] = defaultdict(set)
        for entity in entities:
            cluster_id = _heuristic_entity_cluster(entity)
            if cluster_id:
                grouped[cluster_id].add(entity)
        # Keep only meaningful groups (2+ members) so singletons stay unclustered.
        entity_clusters = {
            cluster_id: members for cluster_id, members in grouped.items() if len(members) >= 2
        }
        used_fallback = bool(entity_clusters)

    if not edge_clusters and relations:
        predicates = {predicate for _, predicate, _ in relations}
        grouped_edges: dict[str, set[str]] = defaultdict(set)
        for predicate in predicates:
            cluster_id = _heuristic_edge_cluster(predicate)
            if cluster_id:
                grouped_edges[cluster_id].add(predicate)
        edge_clusters = {
            cluster_id: members for cluster_id, members in grouped_edges.items() if len(members) >= 2
        }
        used_fallback = used_fallback or bool(edge_clusters)

    return entity_clusters, edge_clusters, used_fallback


def _string_to_color(label: str) -> str:
    """Generate a deterministic pastel-like color for a given label."""
    digest = hashlib.sha1(label.encode("utf-8")).hexdigest()
    hue = int(digest[:2], 16) / 255.0
    saturation = 0.55 + (int(digest[2:4], 16) / 255.0) * 0.3
    lightness = 0.45 + (int(digest[4:6], 16) / 255.0) * 0.25
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def _sorted_ignore_case(items: Iterable[str]) -> list[str]:
    return sorted(items, key=lambda value: value.lower())


def _normalize_source_documents(
    source_documents: list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    if not source_documents:
        return []

    from urllib.parse import urlparse

    normalized_documents: list[dict[str, str]] = []
    for doc_index, source_document in enumerate(source_documents):
        page_url = str(source_document.get("url", "")).strip()
        page_text = str(source_document.get("text", "")).strip()
        crawl_source = str(source_document.get("crawl_source", "")).strip()
        if not page_url or not page_text:
            continue
        try:
            domain = urlparse(page_url).netloc or ""
        except Exception:
            domain = ""
        normalized_document = {
            "url": page_url,
            "text": page_text,
            "docIndex": doc_index,
            "domain": domain,
            "textLength": len(page_text),
        }
        if crawl_source:
            normalized_document["crawl_source"] = crawl_source
        normalized_documents.append(normalized_document)
    return normalized_documents


def _collapse_whitespace(value: str) -> str:
    return " ".join(value.split())


def _build_snippet(text: str, start: int, end: int, radius: int = 120, max_length: int = 320) -> str:
    snippet_start = max(0, start - radius)
    snippet_end = min(len(text), end + radius)
    snippet = _collapse_whitespace(text[snippet_start:snippet_end])
    if len(snippet) > max_length:
        snippet = snippet[:max_length].rstrip() + " …"
    prefix = "… " if snippet_start > 0 else ""
    suffix = " …" if snippet_end < len(text) else ""
    return f"{prefix}{snippet}{suffix}"


def _find_entity_mentions(
    label: str,
    documents: list[dict[str, str]],
    max_mentions: int = 5,
) -> dict[str, Any]:
    needle = (label or "").strip()
    mentions: list[dict[str, Any]] = []
    total_count = 0
    seen_docs: set[str] = set()
    if not needle:
        return {"mentions": [], "mentionCount": 0, "documentCount": 0, "firstSeen": None}

    lowered_needle = needle.lower()
    for doc in documents:
        text = str(doc.get("text", ""))
        lowered_text = text.lower()
        search_from = 0
        while True:
            found = lowered_text.find(lowered_needle, search_from)
            if found < 0:
                break
            total_count += 1
            seen_docs.add(str(doc.get("url", "")))
            if len(mentions) < max_mentions:
                end = found + len(needle)
                mentions.append(
                    {
                        "url": doc.get("url", ""),
                        "docIndex": doc.get("docIndex", 0),
                        "domain": doc.get("domain", ""),
                        "crawlSource": doc.get("crawl_source", ""),
                        "charStart": found,
                        "charEnd": end,
                        "textLength": doc.get("textLength", len(text)),
                        "positionPercent": round(found / len(text) * 100, 1) if text else 0,
                        "snippet": _build_snippet(text, found, end),
                    }
                )
            search_from = found + max(1, len(lowered_needle))
            if search_from >= len(lowered_text):
                break

    mentions.sort(key=lambda item: (item["docIndex"], item["charStart"]))
    first_seen = mentions[0] if mentions else None
    return {
        "mentions": mentions[:max_mentions],
        "mentionCount": total_count,
        "documentCount": len(seen_docs),
        "firstSeen": first_seen,
    }


def _find_edge_evidence(
    subject: str,
    obj: str,
    documents: list[dict[str, str]],
    max_docs: int = 3,
) -> dict[str, Any]:
    lowered_subject = (subject or "").strip().lower()
    lowered_object = (obj or "").strip().lower()
    supporting: list[dict[str, Any]] = []
    doc_count = 0
    if not lowered_subject or not lowered_object:
        return {"supportingDocs": [], "documentCount": 0, "firstSeen": None}

    for doc in documents:
        text = str(doc.get("text", ""))
        lowered_text = text.lower()
        subject_pos = lowered_text.find(lowered_subject)
        object_pos = lowered_text.find(lowered_object)
        if subject_pos < 0 or object_pos < 0:
            continue
        doc_count += 1
        if len(supporting) < max_docs:
            window_start = min(subject_pos, object_pos)
            window_end = max(
                subject_pos + len(subject.strip()),
                object_pos + len(obj.strip()),
            )
            supporting.append(
                {
                    "url": doc.get("url", ""),
                    "docIndex": doc.get("docIndex", 0),
                    "domain": doc.get("domain", ""),
                    "crawlSource": doc.get("crawl_source", ""),
                    "subjectPos": subject_pos,
                    "objectPos": object_pos,
                    "distance": abs(object_pos - subject_pos),
                    "textLength": doc.get("textLength", len(text)),
                    "positionPercent": round(window_start / len(text) * 100, 1) if text else 0,
                    "snippet": _build_snippet(text, window_start, window_end, radius=100, max_length=360),
                }
            )

    supporting.sort(key=lambda item: (item["distance"], item["docIndex"]))
    return {
        "supportingDocs": supporting,
        "documentCount": doc_count,
        "firstSeen": supporting[0] if supporting else None,
    }


def _build_view_model(
    graph: Graph,
    source_documents: list[dict[str, str]] | None = None,
    qa_runtime_config: dict[str, Any] | None = None,
    build_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Collect all entities from both the entities set and relations
    all_entities = set(graph.entities)
    for subject, _, obj in graph.relations:
        all_entities.add(subject)
        all_entities.add(obj)
    entities = _sorted_ignore_case(all_entities)

    relations = sorted(
        graph.relations,
        key=lambda triple: (triple[1].lower(), triple[0].lower(), triple[2].lower()),
    )

    normalized_docs = _normalize_source_documents(source_documents)

    entity_clusters, edge_clusters, used_heuristic_clusters = _apply_heuristic_clusters(
        graph.entity_clusters, graph.edge_clusters, entities, relations
    )

    entity_member_to_cluster: dict[str, str] = {}
    cluster_view: list[dict[str, Any]] = []

    for representative, members in entity_clusters.items():
        full_members = set(members)
        full_members.add(representative)
        ordered_members = _sorted_ignore_case(full_members)
        color = _string_to_color(f"entity::{representative}")
        cluster_view.append(
            {
                "id": representative,
                "label": representative,
                "members": ordered_members,
                "size": len(ordered_members),
                "color": color,
            }
        )
        for member in ordered_members:
            entity_member_to_cluster[member] = representative

    node_color_lookup: dict[str, str] = {}
    if cluster_view:
        for cluster in cluster_view:
            for member in cluster["members"]:
                node_color_lookup[member] = cluster["color"]
    else:
        for entity in entities:
            node_color_lookup[entity] = _string_to_color(f"entity::{entity}")

    edge_member_to_cluster: dict[str, str] = {}
    edge_color_lookup: dict[str, str] = {}
    edge_cluster_view: list[dict[str, Any]] = []

    for representative, members in edge_clusters.items():
        full_members = set(members)
        full_members.add(representative)
        ordered_members = _sorted_ignore_case(full_members)
        color = _string_to_color(f"edge::{representative}")
        edge_cluster_view.append(
            {
                "id": representative,
                "label": representative,
                "members": ordered_members,
                "size": len(ordered_members),
                "color": color,
            }
        )
        for member in ordered_members:
            edge_member_to_cluster[member] = representative
            edge_color_lookup[member] = color

    degree = Counter()
    indegree = Counter()
    outdegree = Counter()
    predicate_counts = Counter()

    adjacency: dict[str, set[str]] = defaultdict(set)
    node_neighbors: dict[str, set[str]] = defaultdict(set)
    node_edges: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"incoming": [], "outgoing": []}
    )

    edges_view: list[dict[str, Any]] = []

    for index, (subject, predicate, obj) in enumerate(relations):
        predicate_counts[predicate] += 1
        degree[subject] += 1
        degree[obj] += 1
        outdegree[subject] += 1
        indegree[obj] += 1
        adjacency[subject].add(obj)
        adjacency[obj].add(subject)
        node_neighbors[subject].add(obj)
        node_neighbors[obj].add(subject)

        edge_id = f"e{index}"
        color = edge_color_lookup.get(predicate)
        if not color:
            color = _string_to_color(f"predicate::{predicate}")
            edge_color_lookup[predicate] = color

        edges_view.append(
            {
                "id": edge_id,
                "source": subject,
                "target": obj,
                "predicate": predicate,
                "cluster": edge_member_to_cluster.get(predicate),
                "color": color,
                "tooltip": f"{subject} —{predicate}→ {obj}",
                "provenance": _find_edge_evidence(subject, obj, normalized_docs),
            }
        )

        node_edges[subject]["outgoing"].append(edge_id)
        node_edges[obj]["incoming"].append(edge_id)

    isolated_entities = [entity for entity in entities if degree[entity] == 0]

    def connected_components() -> list[dict[str, Any]]:
        visited: set[str] = set()
        components: list[dict[str, Any]] = []
        for node in entities:
            if node in visited:
                continue
            queue: deque[str] = deque([node])
            visited.add(node)
            members: list[str] = []
            while queue:
                current = queue.popleft()
                members.append(current)
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(
                {
                    "size": len(members),
                    "members": _sorted_ignore_case(members),
                }
            )
        components.sort(key=lambda comp: (-comp["size"], comp["members"][0]))
        return components

    components = connected_components()

    nodes_view: list[dict[str, Any]] = []
    for entity in entities:
        cluster_id = entity_member_to_cluster.get(entity)
        radius = 18 + min(degree[entity], 8) * 2
        provenance = _find_entity_mentions(entity, normalized_docs)
        nodes_view.append(
            {
                "id": entity,
                "label": entity,
                "cluster": cluster_id,
                "color": node_color_lookup.get(entity, "#64748b"),
                "degree": degree[entity],
                "indegree": indegree[entity],
                "outdegree": outdegree[entity],
                "isRepresentative": cluster_id == entity if cluster_id else False,
                "radius": radius,
                "neighbors": _sorted_ignore_case(node_neighbors.get(entity, set())),
                "edgeIds": node_edges.get(entity, {"incoming": [], "outgoing": []}),
                "provenance": provenance,
            }
        )

    top_entities = sorted(
        (
            {
                "label": node["label"],
                "degree": node["degree"],
                "indegree": node["indegree"],
                "outdegree": node["outdegree"],
                "cluster": node["cluster"],
            }
            for node in nodes_view
        ),
        key=lambda item: (-item["degree"], item["label"].lower()),
    )[:10]

    top_relations = sorted(
        (
            {
                "predicate": predicate,
                "count": count,
                "cluster": edge_member_to_cluster.get(predicate),
                "color": edge_color_lookup.get(predicate, "#64748b"),
            }
            for predicate, count in predicate_counts.items()
        ),
        key=lambda item: (-item["count"], item["predicate"].lower()),
    )[:10]

    stats = {
        "entities": len(entities),
        "relations": len(edges_view),
        "relationTypes": len(predicate_counts),
        "entityClusters": len(cluster_view),
        "edgeClusters": len(edge_cluster_view),
        "isolatedEntities": len(isolated_entities),
        "components": len(components),
        "averageDegree": round(
            sum(degree[entity] for entity in entities) / len(entities), 2
        )
        if entities
        else 0,
        "density": round(len(edges_view) / (len(entities) * (len(entities) - 1)), 3)
        if len(entities) > 1
        else 0,
    }

    relation_records = [
        {
            "source": subject,
            "predicate": predicate,
            "target": obj,
            "edgeId": edge["id"],
            "color": edge["color"],
            "provenance": edge.get("provenance"),
        }
        for edge, (subject, predicate, obj) in zip(edges_view, relations)
    ]

    # UI limits: calculate sensible defaults for client-side lists
    total_entities = len(entities)
    # Root entity suggestions: scale with entity count, clamp to [30,100], but never exceed actual entities
    root_suggestion_limit = min(total_entities, max(30, min(100, int(total_entities * 0.2))))
    # Entity cluster suggestions: base on entity count, clamp to [30,100], but not more than available clusters
    cluster_suggestion_limit = min(len(cluster_view), max(30, min(100, int(total_entities * 0.05))))
    # Relation suggestion limit: based on visible edges, clamp to [20,50]
    relation_suggestion_limit = min(len(edges_view), max(20, min(50, int(len(edges_view) * 0.1))))

    return {
        "nodes": nodes_view,
        "edges": edges_view,
        "clusters": cluster_view,
        "edgeClusters": edge_cluster_view,
        "topEntities": top_entities,
        "topRelations": top_relations,
        "stats": stats,
        "isolatedEntities": isolated_entities,
        "components": components,
        "relations": relation_records,
        "sourceDocuments": normalized_docs,
        "clusterProvenance": "heuristic" if used_heuristic_clusters else "stored",
        "qaRuntime": qa_runtime_config or {"enabled": False, "endpoint": None, "model": None},
        "buildInfo": build_info or {"series": None, "build": None, "version": None, "builtAt": None},
        "ui": {
            "rootEntitySuggestionLimit": root_suggestion_limit,
            "clusterSuggestionLimit": cluster_suggestion_limit,
            "relationSuggestionLimit": relation_suggestion_limit,
        },
    }


HTML_TEMPLATE = (Path(__file__).parent / "template.html").read_text(encoding="utf-8")


def _build_html_output_comment(output_metadata: dict[str, Any] | None) -> str:
    if not output_metadata:
        return ""

    lines = [
        "Generated File Metadata",
        f"Meaning: {output_metadata.get('meaning', '')}",
        f"File Type: {output_metadata.get('fileType', '')}",
        f"Created At: {output_metadata.get('createdAt', '')}",
        f"Source URL: {output_metadata.get('sourceUrl', '')}",
        f"License: {output_metadata.get('license', '')}",
        f"Build Version: {output_metadata.get('buildVersion', '')}",
        f"Generator: {output_metadata.get('generatedBy', '')}",
    ]

    attributes = output_metadata.get("attributes") or []
    if attributes:
        lines.append(f"Attributes: {', '.join(str(attribute) for attribute in attributes)}")

    comment_body = "\n".join(lines)
    return f"<!--\n{comment_body}\n-->\n"


def visualize(
    graph: Graph,
    output_path: str | None = None,
    *,
    open_in_browser: bool = False,
    source_documents: list[dict[str, str]] | None = None,
    qa_runtime_config: dict[str, Any] | None = None,
    build_info: dict[str, Any] | None = None,
    output_metadata: dict[str, Any] | None = None,
) -> Path:
    """Render an interactive dashboard for a graph.

    Args:
        graph: Graph instance to visualize.
        output_path: Optional path where the HTML document should be stored.
        open_in_browser: When True, open the generated file in the default browser.

    Returns:
        Path to the generated HTML file.
    """

    if not graph or not graph.entities:
        raise ValueError("Cannot visualize an empty graph")

    view_model = _build_view_model(
        graph,
        source_documents=source_documents,
        qa_runtime_config=qa_runtime_config,
        build_info=build_info,
    )
    html = HTML_TEMPLATE.replace(
        "<!--DATA-->",
        json.dumps(view_model, ensure_ascii=False, indent=2),
    )
    html = _build_html_output_comment(output_metadata) + html

    # Make sidebar visible for standalone mode by removing display: none
    # display none must be set to prevent flicker when loading in main app
    html = html.replace(
        "display: none; /* Hidden by default - controlled by main app */",
        "display: block; /* Visible in standalone mode */",
    )

    destination = Path(output_path or "graph-visualization.html").resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html, encoding="utf-8")

    if open_in_browser:
        webbrowser.open(destination.as_uri())

    return destination
