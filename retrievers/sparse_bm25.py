import json
import math
import re
import hashlib
from collections import Counter, defaultdict

from qdrant_client.http import models
from llama_index.core.schema import NodeWithScore


class SparseBM25Encoder:
    def __init__(self, k1=1.5, b=0.75, sparse_dim=1048576):
        self.k1 = k1
        self.b = b
        self.sparse_dim = sparse_dim
        self.doc_count = 0
        self.avgdl = 0.0
        self.df = {}

    def _tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        return re.findall(r"\w+", text.lower(), flags=re.UNICODE)

    def _hash_token(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.sparse_dim

    def _idf(self, df: int) -> float:
        return math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)

    def fit(self, texts: list[str]) -> None:
        df = defaultdict(int)
        doc_lens = []

        for text in texts:
            tokens = self._tokenize(text)
            doc_lens.append(len(tokens))
            unique_indices = {self._hash_token(t) for t in tokens}
            for idx in unique_indices:
                df[idx] += 1

        self.doc_count = len(doc_lens)
        self.avgdl = sum(doc_lens) / max(1, self.doc_count)
        self.df = dict(df)

    def encode_document(self, text: str) -> models.SparseVector:
        tokens = self._tokenize(text)
        if not tokens or self.doc_count == 0:
            return models.SparseVector(indices=[], values=[])

        tf = Counter(self._hash_token(t) for t in tokens)
        dl = len(tokens)

        indices = []
        values = []
        for idx, count in tf.items():
            df = self.df.get(idx, 0)
            if df == 0:
                continue
            idf = self._idf(df)
            denom = count + self.k1 * (1 - self.b + self.b * dl / max(1.0, self.avgdl))
            score = idf * (count * (self.k1 + 1)) / denom
            if score > 0:
                indices.append(idx)
                values.append(score)

        return models.SparseVector(indices=indices, values=values)

    def encode_query(self, text: str) -> models.SparseVector:
        tokens = self._tokenize(text)
        if not tokens or self.doc_count == 0:
            return models.SparseVector(indices=[], values=[])

        tf = Counter(self._hash_token(t) for t in tokens)
        indices = []
        values = []
        for idx, count in tf.items():
            df = self.df.get(idx, 0)
            if df == 0:
                continue
            idf = self._idf(df)
            score = idf * count
            if score > 0:
                indices.append(idx)
                values.append(score)

        return models.SparseVector(indices=indices, values=values)

    def to_dict(self) -> dict:
        return {
            "k1": self.k1,
            "b": self.b,
            "sparse_dim": self.sparse_dim,
            "doc_count": self.doc_count,
            "avgdl": self.avgdl,
            "df": {str(k): v for k, v in self.df.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "SparseBM25Encoder":
        inst = cls(
            k1=payload.get("k1", 1.5),
            b=payload.get("b", 0.75),
            sparse_dim=payload.get("sparse_dim", 1048576),
        )
        inst.doc_count = payload.get("doc_count", 0)
        inst.avgdl = payload.get("avgdl", 0.0)
        inst.df = {int(k): int(v) for k, v in payload.get("df", {}).items()}
        return inst

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "SparseBM25Encoder":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls.from_dict(payload)


class QdrantSparseRetriever:
    def __init__(self, client, collection_name, docstore, encoder, vector_name="bm25", top_k=20):
        self.client = client
        self.collection_name = collection_name
        self.docstore = docstore
        self.encoder = encoder
        self.vector_name = vector_name
        self.top_k = top_k

    def retrieve(self, query: str, selected_files: list[str] | None = None):
        sparse_query = self.encoder.encode_query(query)
        if not sparse_query.indices:
            return []

        query_vector = models.NamedSparseVector(name=self.vector_name, vector=sparse_query)
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=self.top_k,
            with_payload=False,
            with_vectors=False,
        )

        nodes = []
        for point in results:
            node_id = str(point.id)
            node = self.docstore.docs.get(node_id) or self.docstore.docs.get(point.id)
            if not node:
                continue
            if selected_files:
                file_name = node.metadata.get("file_name", "")
                if file_name not in selected_files:
                    continue
            nodes.append(NodeWithScore(node=node, score=point.score))

        return nodes