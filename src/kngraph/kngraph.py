from typing import Any, Dict, List, Optional, Union, cast
from typing_extensions import deprecated

from kngraph.steps._1_get_entities import get_entities
from kngraph.steps._2_get_relations import get_relations
from kngraph.steps._3_deduplicate import run_deduplication, DeduplicateMethod
from kngraph.utils.chunk_text import chunk_text
from kngraph.utils.ollama_client import is_ollama_model
from kngraph.utils.visualize_kg import visualize as visualize_kg
from kngraph.models import Graph
import dspy
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Configure dspy logging to only show errors
import logging

logger = logging.getLogger(__name__)

dspy_logger = logging.getLogger("dspy")
dspy_logger.setLevel(logging.CRITICAL)


def _should_print_local_progress() -> bool:
    return os.getenv('KG_OLLAMA_PROGRESS', '1').lower() not in {'0', 'false', 'no', 'off'}


def _print_local_progress(message: str) -> None:
    if _should_print_local_progress():
        print(f'[kngraph] {message}', flush=True)


class KNGraph:
    def __init__(
        self,
        model: str = "openai/gpt-4o",
        max_tokens: int = 16000,  # minimum for gpt-5 family models
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        retrieval_model: Optional[str] = None,
        disable_cache: bool = False,
    ):
        """Initialize KNGraph with optional model configuration

        Args:
            model: Name of model to use (e.g. 'gpt-4')
            temperature: Temperature for model sampling
            api_key: API key for model access
            api_base: Specify the base URL endpoint for making API calls to a language model service
        """
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_key = api_key
        self.api_base = api_base
        self.retrieval_model: Optional[SentenceTransformer] = None
        self.lm: dspy.LM | None = None
        self.disable_cache = disable_cache

        self.init_model(
            model=model,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=api_key,
            api_base=api_base,
            retrieval_model=retrieval_model,
        )

    def _uses_gpt5_family(self, model: str | None = None) -> bool:
        resolved_model = model or self.model
        return "gpt-5" in resolved_model

    def _resolved_temperature(self, temperature: float | None = None) -> float | None:
        candidate = self.temperature if temperature is None else temperature
        if self._uses_gpt5_family():
            return None
        return candidate

    def _uses_direct_ollama(self, model: str | None = None, api_base: str | None = None) -> bool:
        resolved_model = model or self.model
        resolved_api_base = self.api_base if api_base is None else api_base
        return is_ollama_model(resolved_model, resolved_api_base)

    def validate_max_tokens(self, max_tokens: int):
        if "gpt-5" in self.model and max_tokens < 16000:
            raise ValueError("Max tokens must be 16000 for gpt-5 family models")

    def init_model(
        self,
        model: str | None = None,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        retrieval_model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
    ):
        """Initialize or reinitialize the model with new parameters

        Args:
            model: Name of model to use (e.g. 'gpt-4')
            temperature: Temperature for model sampling
            api_key: API key for model access
            api_base: API base for model access
            retrieval_model: Name of retrieval model to use
            reasoning_effort: Reasoning effort for model
            max_tokens: Maximum tokens for model
            temperature: Temperature for model sampling
        """

        # Update instance variables if new values provided
        if model is not None:
            self.model = model
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if api_key is not None:
            self.api_key = api_key
        if api_base is not None:
            self.api_base = api_base
        if temperature is not None:
            self.temperature = temperature
        if reasoning_effort is not None:
            self.reasoning_effort = reasoning_effort
        if retrieval_model is not None:
            self.retrieval_model = SentenceTransformer(retrieval_model)

        self.validate_max_tokens(self.max_tokens)

        if self._uses_direct_ollama():
            self.lm = None
            return

        lm_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "api_base": self.api_base,
            "cache": not self.disable_cache,
            "model_type": "responses" if self.model.startswith("openai/") else "chat",
        }

        if self.api_key:
            lm_kwargs["api_key"] = self.api_key

        if self.reasoning_effort:
            lm_kwargs["reasoning"] = {"effort": self.reasoning_effort}

        resolved_temperature = self._resolved_temperature()
        if resolved_temperature is not None:
            lm_kwargs["temperature"] = resolved_temperature

        # Initialize dspy LM with current settings
        self.lm = dspy.LM(**lm_kwargs)

    @staticmethod
    def from_file(file_path: str) -> Graph:
        with open(file_path, "r") as f:
            raw_data = json.load(f)
            graph = Graph(
                **{
                    key: value
                    for key, value in raw_data.items()
                    if not str(key).startswith("_")
                }
            )
        return graph

    @staticmethod
    def from_dict(graph_dict: dict) -> Graph:
        return Graph(**graph_dict)

    def generate(
        self,
        input_data: Union[str, List[Dict]],
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        context: str = "",
        chunk_size: Optional[int] = None,
        reasoning_effort: str | None = None,
        deduplication_method: DeduplicateMethod | None = DeduplicateMethod.SEMHASH,
        temperature: float | None = None,
        output_folder: Optional[str] = None,
        no_dspy: bool = False,
    ) -> Graph:
        """Generate a knowledge graph from input text or messages.

        Args:
            input_data: Text string or list of message dicts
            model: Name of OpenAI model to use
            api_key (str): OpenAI API key for making model calls
            chunk_size: Max size of text chunks in characters to process
            context: Description of data context
            output_folder: Path to save partial progress

        Returns:
            Graph: Generated knowledge graph
        """

        # Process input data
        is_conversation = isinstance(input_data, list)
        if is_conversation:
            # Extract text from messages
            text_content = []
            for message in input_data:
                if (
                    not isinstance(message, dict)
                    or "role" not in message
                    or "content" not in message
                ):
                    raise ValueError(
                        "Messages must be dicts with 'role' and 'content' keys"
                    )
                if message["role"] in ["user", "assistant"]:
                    text_content.append(f"{message['role']}: {message['content']}")

            # Join with newlines to preserve message boundaries
            processed_input = "\n".join(text_content)
        else:
            processed_input = input_data

        # Reinitialize dspy with new parameters if any are provided
        if any([model, temperature, api_key, api_base, reasoning_effort]):
            self.init_model(
                model=model or self.model,
                temperature=temperature if temperature is not None else self.temperature,
                api_key=api_key or self.api_key,
                api_base=api_base or self.api_base,
                reasoning_effort=reasoning_effort or self.reasoning_effort,
            )

        use_direct_ollama = no_dspy or self._uses_direct_ollama()
        active_lm = self.lm
        if not use_direct_ollama and active_lm is None:
            raise RuntimeError("Language model is not initialized")

        request_temperature = self._resolved_temperature(temperature)

        def _normalize_entities(values: Any) -> set[str]:
            return {str(value) for value in values}

        def _normalize_relations(values: Any) -> set[tuple[str, str, str]]:
            return {
                (str(source), str(relation), str(target))
                for source, relation, target in values
            }

        def _process(content: str, lm: dspy.LM | None, chunk_index: int | None = None, chunk_count: int | None = None) -> tuple[set[str], set[tuple[str, str, str]]]:
            chunk_label = ''
            if chunk_index is not None and chunk_count is not None:
                chunk_label = f'청크 {chunk_index}/{chunk_count} '
            if use_direct_ollama:
                _print_local_progress(f'{chunk_label}엔티티 추출 시작 (길이 {len(content)})')
                entities = get_entities(
                    content,
                    is_conversation,
                    model=self.model,
                    api_key=self.api_key,
                    api_base=self.api_base,
                    temperature=request_temperature,
                    use_ollama=True,
                    progress_label=f'{chunk_label}엔티티 추출',
                )
                _print_local_progress(f'{chunk_label}엔티티 추출 완료 ({len(entities)}개)')
                if not entities:
                    _print_local_progress(f'{chunk_label}엔티티가 없어 관계 추출 건너뜀')
                    return set(), set()
                _print_local_progress(f'{chunk_label}관계 추출 시작')
                relations = get_relations(
                    content,
                    entities,
                    is_conversation=is_conversation,
                    model=self.model,
                    api_key=self.api_key,
                    api_base=self.api_base,
                    temperature=request_temperature,
                    use_ollama=True,
                    progress_label=f'{chunk_label}관계 추출',
                )
                _print_local_progress(f'{chunk_label}관계 추출 완료 ({len(relations)}개)')
                return _normalize_entities(entities), _normalize_relations(relations)

            if lm is None:
                raise RuntimeError("Language model is not initialized")

            with dspy.context(lm=lm):
                entities = get_entities(
                    content,
                    is_conversation,
                    use_litellm_prompt=no_dspy,
                    model=self.model,
                    api_key=self.api_key,
                    api_base=self.api_base,
                    temperature=request_temperature,
                )
                relations = get_relations(
                    content,
                    entities,
                    is_conversation=is_conversation,
                    use_litellm_prompt=no_dspy,
                    model=self.model,
                    api_key=self.api_key,
                    api_base=self.api_base,
                    temperature=request_temperature,
                )
                return _normalize_entities(entities), _normalize_relations(relations)

        entities: set[str] = set()
        relations: set[tuple[str, str, str]] = set()

        if not chunk_size:
            try:
                entities, relations = _process(processed_input, active_lm)
            except Exception as e:
                if "context length" in str(e).lower():
                    logger.warning(
                        f"Context length error: {e}. Chunking text with chunk size 16384."
                    )
                    chunk_size = 16384
                else:
                    raise e

        if chunk_size:
            chunks = chunk_text(processed_input, chunk_size)
            entities = set()
            relations = set()

            if use_direct_ollama:
                _print_local_progress(f'로컬 Ollama 청크 처리 시작 ({len(chunks)}개 청크)')
                for index, chunk in enumerate(chunks, start=1):
                    chunk_entities, chunk_relations = _process(
                        chunk,
                        active_lm,
                        chunk_index=index,
                        chunk_count=len(chunks),
                    )
                    entities.update(chunk_entities)
                    relations.update(chunk_relations)
                _print_local_progress('로컬 Ollama 청크 처리 완료')
            else:
                with ThreadPoolExecutor() as executor:
                    future_to_chunk = {
                        executor.submit(_process, chunk, active_lm): chunk for chunk in chunks
                    }

                    for future in as_completed(future_to_chunk):
                        chunk_entities, chunk_relations = future.result()
                        entities.update(chunk_entities)
                        relations.update(chunk_relations)

        graph = Graph(
            entities=entities,
            relations=relations,
            edges={relation[1] for relation in relations},
        )

        if deduplication_method:
            graph = self.deduplicate(
                graph, method=deduplication_method, context=context
            )

        if output_folder:
            self.export_graph(graph, os.path.join(output_folder, "graph.json"))
        return graph

    @deprecated("Use KNGraph.deduplicate() method instead")
    def cluster(
        self,
        graph: Graph,
        **kwargs,
    ) -> Graph:
        return self.deduplicate(graph, **kwargs)

    def deduplicate(
        self,
        graph: Graph,
        method: DeduplicateMethod = DeduplicateMethod.FULL,
        semhash_similarity_threshold: float = 0.95,  # recommended to keep at 0.95
        model: str | None = None,
        temperature: float | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        context: str = "",  # TODO: implement context
    ) -> Graph:
        # Reinitialize dspy with new parameters if any are provided
        if any([model, temperature, api_key, api_base]):
            self.init_model(
                model=model or self.model,
                temperature=temperature if temperature is not None else self.temperature,
                api_key=api_key or self.api_key,
                api_base=api_base or self.api_base,
            )

        active_lm = self.lm
        if method != DeduplicateMethod.SEMHASH and active_lm is None:
            raise RuntimeError("Language model is not initialized")

        return run_deduplication(
            lm=active_lm,
            graph=graph,
            method=method,
            retrieval_model=self.retrieval_model,
            semhash_similarity_threshold=semhash_similarity_threshold,
        )

    def aggregate(self, graphs: list[Graph]) -> Graph:
        # Initialize empty sets for combined graph
        all_entities = set()
        all_relations = set()
        all_edges = set()
        all_entity_metadata: dict[str, set[str]] = {}

        # Combine all graphs
        for graph in graphs:
            all_entities.update(graph.entities)
            all_relations.update(graph.relations)
            all_edges.update(graph.edges)
            if graph.entity_metadata:
                for entity, metadata_set in graph.entity_metadata.items():
                    if entity in all_entity_metadata:
                        all_entity_metadata[entity].update(metadata_set)
                    else:
                        all_entity_metadata[entity] = metadata_set.copy()

        # Create and return aggregated graph
        return Graph(
            entities=all_entities,
            relations=all_relations,
            edges=all_edges,
            entity_metadata=all_entity_metadata if all_entity_metadata else None,
        )

    @staticmethod
    def visualize(
        graph: Graph,
        output_path: str,
        open_in_browser: bool = False,
        source_documents: Optional[List[Dict[str, str]]] = None,
        qa_runtime_config: Optional[Dict[str, Union[str, bool, None]]] = None,
        build_info: Optional[Dict[str, Union[str, int, None]]] = None,
        output_metadata: Optional[Dict[str, Union[str, int, list[str], None]]] = None,
    ):
        visualize_kg(
            graph,
            output_path,
            open_in_browser=open_in_browser,
            source_documents=source_documents,
            qa_runtime_config=qa_runtime_config,
            build_info=build_info,
            output_metadata=output_metadata,
        )

    # ====== Retrieval Methods ======

    def _parse_embedding_model(
        self, model: Optional[SentenceTransformer] = None
    ) -> SentenceTransformer:
        if model is None:
            model = self.retrieval_model
        if model is None:
            raise ValueError("No retrieval model provided")
        return model

    @staticmethod
    def to_nx(graph: Graph) -> nx.DiGraph:
        G = nx.DiGraph()
        for entity in graph.entities:
            G.add_node(entity)

        for relation in graph.relations:
            source, rel, target = relation
            G.add_edge(source, target, relation=rel)
        return G

    def generate_embeddings(
        self,
        graph: Union[Graph, nx.DiGraph],
        model: Optional[SentenceTransformer] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        model = self._parse_embedding_model(model)
        if isinstance(graph, Graph):
            graph = self.to_nx(graph)

        node_embeddings = {
            node: np.asarray(model.encode(node), dtype=float) for node in graph.nodes
        }
        relation_embeddings = {
            rel: np.asarray(model.encode(rel), dtype=float)
            # TODO: this is triggering index out of range error
            for rel in set(edge[2]["relation"] for edge in graph.edges(data=True))
        }
        return node_embeddings, relation_embeddings

    def retrieve(
        self,
        query: str,
        node_embeddings: dict[str, np.ndarray],
        graph: nx.DiGraph,
        model: Optional[SentenceTransformer] = None,
        k: int = 8,
        verbose: bool = False,
    ) -> tuple[list[tuple[str, float]], set[str], str]:
        model = self._parse_embedding_model(model)
        top_nodes = self.retrieve_relevant_nodes(query, node_embeddings, model, k)
        context = set()
        for node, _ in top_nodes:
            node_context = self.retrieve_context(node, graph)
            if verbose:
                print(f"Context for node {node}: {node_context}")
            context.update(node_context)
        context_text = " ".join(context)
        if verbose:
            print(f"Combined context: '{context_text}'\n---")
        return top_nodes, context, context_text

    @staticmethod
    def retrieve_relevant_nodes(
        query: str,
        node_embeddings: dict[str, np.ndarray],
        model: SentenceTransformer,
        k: int = 8,
    ) -> list[tuple[str, float]]:
        query_embedding = np.asarray(model.encode(query), dtype=float).reshape(1, -1)
        similarities = []
        for node, embed in node_embeddings.items():
            target_embedding = np.array(embed).reshape(1, -1)
            similarity = cosine_similarity(query_embedding, target_embedding)[0][0]
            similarities.append((node, similarity))
        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
        return similarities[:k]

    @staticmethod
    def retrieve_context(node: str, graph: nx.DiGraph, depth: int = 2) -> list[str]:
        context = set()

        def explore_neighbors(current_node, current_depth):
            if current_depth > depth:
                return
            # Outgoing edges
            for neighbor in graph.neighbors(current_node):
                rel = graph[current_node][neighbor]["relation"]
                context.add(f"{current_node} {rel} {neighbor}.")
                explore_neighbors(neighbor, current_depth + 1)
            # Incoming edges
            for neighbor in graph.predecessors(current_node):
                rel = graph[neighbor][current_node]["relation"]
                context.add(f"{neighbor} {rel} {current_node}.")
                explore_neighbors(neighbor, current_depth + 1)

        explore_neighbors(node, 1)
        return list(context)

    @staticmethod
    def export_graph(graph: Graph, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        graph_dict = {
            "entities": list(graph.entities),
            "relations": list(graph.relations),
            "edges": list(graph.edges),
            "entity_clusters": {k: list(v) for k, v in graph.entity_clusters.items()}
            if graph.entity_clusters
            else None,
            "edge_clusters": {k: list(v) for k, v in graph.edge_clusters.items()}
            if graph.edge_clusters
            else None,
            "entity_metadata": graph.entity_metadata,
        }

        with open(output_path, "w") as f:
            json.dump(graph_dict, f, indent=2)

    # ====== Token Usage ======
    def reset_token_usage(self):
        if self.lm is None:
            raise RuntimeError("Language model is not initialized")
        self.lm.history = []

    def extract_token_usage_from_history(self) -> Dict[str, int]:
        """Extract token usage from dspy LM history."""

        if self.lm is None:
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0

        for entry in cast(list[dict[str, Any]], self.lm.history):
            if isinstance(entry, dict):
                # Check for usage information in various possible locations
                usage = entry.get("usage") or entry.get("response", {}).get("usage")

                if usage:
                    total_prompt_tokens += usage.get("prompt_tokens", 0)
                    total_completion_tokens += usage.get("completion_tokens", 0)
                    total_tokens += usage.get("total_tokens", 0)

        return {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
        }
