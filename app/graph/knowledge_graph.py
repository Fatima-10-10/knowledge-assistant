import networkx as nx
from app.graph.extraction import extract_entities_and_relations


class KnowledgeGraph:
    """
    Wraps a NetworkX graph. Nodes = entities (plans, policies, roles).
    Edges = relationships between them, each tagged with which chunk
    it came from (for traceability, same principle as our citations).
    """

    def __init__(self):
        self.graph = nx.DiGraph()  # directed: relationships have a direction

    def build_from_chunks(self, chunks, llm=None, max_chunks: int = None):
        """
        Extract entities/relationships from each chunk and add them to
        the graph. max_chunks limits how many we process (LLM calls
        add up -- useful for testing on a subset first).
        """
        chunks_to_process = chunks[:max_chunks] if max_chunks else chunks

        for chunk in chunks_to_process:
            extracted = extract_entities_and_relations(chunk.content, llm=llm)

            for entity in extracted.get("entities", []):
                self.graph.add_node(entity["name"], type=entity.get("type", "UNKNOWN"))

            for rel in extracted.get("relationships", []):
                self.graph.add_edge(
                    rel["source"], rel["target"],
                    relation=rel["relation"],
                    source_page=chunk.page
                )

    def get_related_entities(self, entity_name: str, max_hops: int = 1) -> list[dict]:
        """
        Find entities connected to a given entity, up to N hops away,
        following relationships in EITHER direction (not just outgoing).
        """
        if entity_name not in self.graph:
            return []

        undirected = self.graph.to_undirected()
        related = []
        for target in nx.single_source_shortest_path_length(undirected, entity_name, cutoff=max_hops):
            if target != entity_name:
                related.append(target)
        return related

    def get_relationships_for(self, entity_name: str) -> list[dict]:
        """Get all direct relationships (edges) involving this entity."""
        results = []
        if entity_name in self.graph:
            for _, target, data in self.graph.out_edges(entity_name, data=True):
                results.append({"source": entity_name, "relation": data["relation"], "target": target})
            for source, _, data in self.graph.in_edges(entity_name, data=True):
                results.append({"source": source, "relation": data["relation"], "target": entity_name})
        return results

    def summary(self) -> dict:
        return {
            "num_entities": self.graph.number_of_nodes(),
            "num_relationships": self.graph.number_of_edges(),
            "entities": list(self.graph.nodes())
        }
