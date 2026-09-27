import re

import chromadb

from sentence_transformers import SentenceTransformer

from rank_bm25 import BM25Okapi
from src.config import (
    CHROMA_DIR,
    EMBEDDING_MODEL,
    VECTOR_TOP_K,
    BM25_TOP_K,
    FINAL_TOP_K,
    RRF_K,

)


# =========================================================
# RETRIEVER
# =========================================================

class HybridRetriever:

    def __init__(self):

        # -------------------------------------------------
        # Embedding model
        # -------------------------------------------------

        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        # -------------------------------------------------
        # Chroma
        # -------------------------------------------------

        client = chromadb.PersistentClient(
                path=CHROMA_DIR,
                settings=chromadb.config.Settings(
                anonymized_telemetry=False
            )
        )
        self.collection = client.get_collection(
            name="sama_finance"
        )

        # -------------------------------------------------
        # Load all documents for BM25
        # -------------------------------------------------

        data = self.collection.get(
            include=["documents", "metadatas"]
        )

        self.documents = data["documents"]
        self.metadatas = data["metadatas"]
        self.ids = data["ids"]

        # -------------------------------------------------
        # BM25
        # -------------------------------------------------

        tokenized_documents = [
            self.tokenize(doc)
            for doc in self.documents
        ]

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

    # =====================================================
    # TOKENIZATION
    # =====================================================

    @staticmethod
    def tokenize(text):

        text = text.lower()

        # Keep Arabic and English words/numbers
        tokens = re.findall(
            r"[\w\u0600-\u06FF]+",
            text
        )

        return tokens

    # =====================================================
    # VECTOR SEARCH
    # =====================================================

    def vector_search(self, query):

        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True
        )[0]

        results = self.collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],
            n_results=VECTOR_TOP_K,
        )

        ids = results["ids"][0]

        return ids

    # =====================================================
    # BM25 SEARCH
    # =====================================================

    def bm25_search(self, query):

        tokens = self.tokenize(query)

        scores = self.bm25.get_scores(tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        top_indices = ranked_indices[:BM25_TOP_K]

        return [
            self.ids[i]
            for i in top_indices
        ]

    # =====================================================
    # RRF
    # =====================================================

    def reciprocal_rank_fusion(
        self,
        vector_ids,
        bm25_ids
    ):
        scores = {}

        for rank, doc_id in enumerate(vector_ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + (
                1 / (RRF_K + rank)
            )

        for rank, doc_id in enumerate(bm25_ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + (
                2 / (RRF_K + rank)
            )

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Keep only the highest-scoring chunk from each page
        best_by_page = {}

        for doc_id, score in ranked:
            index = self.ids.index(doc_id)
            page = self.metadatas[index]["page"]

            if (
                page not in best_by_page
                or score > best_by_page[page][1]
            ):
                best_by_page[page] = (doc_id, score)

        # Rank the best chunk from each page
        page_diversified = sorted(
            best_by_page.values(),
            key=lambda x: x[1],
            reverse=True
        )

        return page_diversified[:FINAL_TOP_K]
    # =====================================================
    # FINAL RETRIEVAL
    # =====================================================

    def expand_with_neighbors(self, doc_ids):

        expanded_ids = []

        chunk_map = {}

        for i, metadata in enumerate(self.metadatas):

            page = int(metadata["page"])
            chunk_id = int(metadata["chunk_id"])

            chunk_map[(page, chunk_id)] = self.ids[i]

        for doc_id in doc_ids:

            index = self.ids.index(doc_id)

            page = int(
                self.metadatas[index]["page"]
            )

            chunk_id = int(
                self.metadatas[index]["chunk_id"]
            )

            # Put the original chunk first
            neighbor_chunk_ids = [
                chunk_id,
                chunk_id - 1,
                chunk_id + 1,
            ]

            for neighbor_chunk_id in neighbor_chunk_ids:

                neighbor_id = chunk_map.get(
                    (page, neighbor_chunk_id)
                )

                if neighbor_id is not None:
                    expanded_ids.append(
                        neighbor_id
                    )

        # Remove duplicates while preserving order
        return list(
            dict.fromkeys(expanded_ids)
        )

    def bm25_is_confident(
        self,
        query,
        bm25_ids,
        min_token_overlap=0.30
    ):
        query_tokens = set(
            self.tokenize(query)
        )

        if not query_tokens:
            return False

        id_to_index = {
            doc_id: i
            for i, doc_id in enumerate(self.ids)
        }

        for doc_id in bm25_ids[:5]:

            index = id_to_index[doc_id]

            document_tokens = set(
                self.tokenize(
                    self.documents[index]
                )
            )

            overlap = (
                query_tokens
                & document_tokens
            )

            overlap_ratio = (
                len(overlap)
                / len(query_tokens)
            )

            if overlap_ratio >= min_token_overlap:
                return True

        return False



    def retrieve(self, query):

        # ---------------------------------------------
        # 1. BM25 search
        # ---------------------------------------------

        bm25_ids = self.bm25_search(query)

        # ---------------------------------------------
        # 2. BM25 confidence check
        # ---------------------------------------------

        if self.bm25_is_confident(
            query,
            bm25_ids
        ):

            selected_ids = bm25_ids[:FINAL_TOP_K]

            retrieval_method = "BM25"

        else:

            # -----------------------------------------
            # 3. Fallback to RRF
            # -----------------------------------------

            vector_ids = self.vector_search(query)

            fused = self.reciprocal_rank_fusion(
                vector_ids,
                bm25_ids
            )

            selected_ids = [
                doc_id
                for doc_id, score in fused
            ][:FINAL_TOP_K]

            retrieval_method = "RRF"

        # ---------------------------------------------
        # 4. Build final results
        # ---------------------------------------------

        id_to_index = {
            doc_id: i
            for i, doc_id in enumerate(self.ids)
        }

        results = []

        for doc_id in selected_ids:

            index = id_to_index[doc_id]

            results.append(
                {
                    "id": doc_id,
                    "text": self.documents[index],
                    "metadata": self.metadatas[index],
                    "rrf_score": 0.0,
                    "retrieval_method": retrieval_method,
                }
            )

        return results