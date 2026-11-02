"""Lineage — 数据血缘追踪。

建立完整的数据关系链：
  Video → Dataset → Version → Recipe → Workspace → Training → ONNX

每个环节记录上下游引用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class LineageNode:
    """血缘关系中的一个节点。"""
    id: str
    type: str            # video / dataset / version / recipe / workspace / training / onnx
    name: str
    created_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LineageEdge:
    """节点间的关系。"""
    source_id: str
    target_id: str
    relation: str        # derived_from / built_from / trained_on / exported_to


class LineageStore:
    """血缘关系存储。"""

    def __init__(self, asset_dir: Path):
        self.lineage_file = asset_dir / "lineage.yaml"
        self._nodes: Dict[str, LineageNode] = {}
        self._edges: List[LineageEdge] = []
        self._batch_mode: bool = False
        self._load()

    def _load(self) -> None:
        if self.lineage_file.exists():
            with open(self.lineage_file, encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data:
                    for n in data.get("nodes", []):
                        node = LineageNode(**n)
                        self._nodes[node.id] = node
                    for e in data.get("edges", []):
                        self._edges.append(LineageEdge(**e))

    def _save(self) -> None:
        data = {
            "nodes": [{"id": n.id, "type": n.type, "name": n.name, "created_at": n.created_at, "metadata": n.metadata}
                      for n in self._nodes.values()],
            "edges": [{"source_id": e.source_id, "target_id": e.target_id, "relation": e.relation}
                      for e in self._edges],
        }
        with open(self.lineage_file, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def add_node(self, node: LineageNode) -> None:
        self._nodes[node.id] = node
        if not self._batch_mode:
            self._save()

    def add_edge(self, edge: LineageEdge) -> None:
        self._edges.append(edge)
        if not self._batch_mode:
            self._save()

    def __enter__(self):
        """Enter batch mode: add_node/add_edge defer _save() until exit."""
        self._batch_mode = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit batch mode: save once. On exception, skip save to avoid partial state."""
        self._batch_mode = False
        if exc_type is None:
            self._save()
        return False

    def get_graph(self) -> Dict[str, Any]:
        """Return the whole lineage graph (all nodes and edges)."""
        return {"nodes": list(self._nodes.values()), "edges": list(self._edges)}

    def get_lineage(self, node_id: str) -> Dict[str, List[Dict]]:
        """获取指定节点的完整血缘链。

        返回上游（来源）和下游（产物）。
        """
        upstream = []
        downstream = []

        for edge in self._edges:
            if edge.target_id == node_id:
                source = self._nodes.get(edge.source_id)
                if source:
                    upstream.append({
                        "node": {"id": source.id, "type": source.type, "name": source.name},
                        "relation": edge.relation,
                    })
            if edge.source_id == node_id:
                target = self._nodes.get(edge.target_id)
                if target:
                    downstream.append({
                        "node": {"id": target.id, "type": target.type, "name": target.name},
                        "relation": edge.relation,
                    })

        return {"upstream": upstream, "downstream": downstream}

    def get_full_chain(self) -> List[Dict]:
        """获取完整的血缘链。"""
        # 找到所有没有上游的节点（根节点，Video）
        roots = []
        all_targets = {e.target_id for e in self._edges}
        for nid, node in self._nodes.items():
            if nid not in all_targets:
                roots.append(node)

        chains = []
        for root in roots:
            chain = self._build_chain(root.id)
            if chain:
                chains.append(chain)
        return chains

    def register_chain(
        self,
        video_id: str = "",
        image_id: str = "",
        dataset_name: str = "",
        version: str = "",
        recipe: str = "",
        workspace: str = "",
        experiment: str = "",
        checkpoint: str = "",
        onnx_path: str = "",
    ) -> None:
        """Register a full 8-layer lineage chain.

        Video → Image → Dataset → Version → Recipe → Workspace → Experiment → Checkpoint → ONNX
        """
        with self:  # batch mode: defer _save() until all nodes/edges added
            prev_id = ""
            for layer_id, layer_type, layer_name in [
                (video_id, "video", video_id),
                (image_id, "image", image_id),
                (dataset_name, "dataset", dataset_name),
                (version, "version", version),
                (recipe, "recipe", recipe),
                (workspace, "workspace", workspace),
                (experiment, "experiment", experiment),
                (checkpoint, "checkpoint", checkpoint),
                (onnx_path, "onnx", onnx_path),
            ]:
                if not layer_id:
                    continue
                self.add_node(LineageNode(id=layer_id, type=layer_type, name=layer_name))
                if prev_id:
                    self.add_edge(LineageEdge(source_id=prev_id, target_id=layer_id, relation="produced"))
                prev_id = layer_id

    def query_by_type(self, type_name: str) -> List[LineageNode]:
        """Find all nodes of a given type."""
        return [n for n in self._nodes.values() if n.type == type_name]

    def trace_to_root(self, node_id: str) -> List[Dict]:
        """Trace from a node all the way to the root (earliest ancestor)."""
        chain = []
        visited = set()
        current = node_id
        while current and current not in visited:
            visited.add(current)
            node = self._nodes.get(current)
            if node:
                chain.append({"id": node.id, "type": node.type, "name": node.name})
            # Find parent (the node that produced this one)
            parent_id = None
            for edge in self._edges:
                if edge.target_id == current:
                    parent_id = edge.source_id
                    break
            if parent_id is None:
                break
            current = parent_id
        return list(reversed(chain))

    def _build_chain(self, node_id: str, visited: Optional[set] = None) -> Optional[Dict]:
        if visited is None:
            visited = set()
        if node_id in visited:
            return None
        visited.add(node_id)

        node = self._nodes.get(node_id)
        if node is None:
            return None

        children = []
        for edge in self._edges:
            if edge.source_id == node_id:
                child = self._build_chain(edge.target_id, visited)
                if child:
                    children.append({"relation": edge.relation, "node": child})

        return {
            "id": node.id,
            "type": node.type,
            "name": node.name,
            "children": children,
        }
