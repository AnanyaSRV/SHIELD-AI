from pathlib import Path

import networkx as nx
import pandas as pd


class GraphInvestigationAgent:
    """Memory-safe, on-demand shared-card proxy investigation.

    The CSV is never loaded into a full DataFrame. Each investigation scans
    only allowlisted columns in chunks, finds the selected transaction, then
    scans for other rows sharing its card1 value. card1 is a proxy feature,
    not a verified account identifier. Offline fraud labels are never loaded.
    """

    def __init__(self, dataset_path=None, max_rows=None, max_group_size=100,
                 cache_full_dataset=False, chunk_size=50_000):
        project_root = Path(__file__).resolve().parents[1]
        self.dataset_path = Path(
            dataset_path or project_root / "data" / "train_transaction.csv"
        )
        if max_rows is not None and max_rows < 1:
            raise ValueError("max_rows must be a positive integer or None.")
        if max_group_size < 2:
            raise ValueError("max_group_size must be at least 2.")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive.")

        self.max_rows = max_rows
        self.max_group_size = int(max_group_size)
        self.chunk_size = int(chunk_size)
        # Retained for backwards compatibility; no full-dataset cache is built.
        self.cache_full_dataset = False
        self.graph = nx.Graph()
        self.card1_map = {}
        self.transaction_index = {}
        self._full_dataset = None
        self.available_columns = []
        self.graph_columns = []
        self.df = pd.DataFrame()
        self.available = False
        self.initialization_error = None

        if not self.dataset_path.is_file():
            self.initialization_error = f"Graph dataset not found: {self.dataset_path}"
            return
        try:
            self.available_columns = pd.read_csv(self.dataset_path, nrows=0).columns.tolist()
            required = {"TransactionID", "card1"}
            missing = required - set(self.available_columns)
            if missing:
                self.initialization_error = (
                    f"Graph dataset is missing required columns: {sorted(missing)}"
                )
                return
            # Explicit allowlist: never read isFraud or other target labels.
            self.graph_columns = [
                col for col in ("TransactionID", "card1", "TransactionAmt", "TransactionDT")
                if col in self.available_columns
            ]
            self.available = True
        except Exception as exc:
            self.initialization_error = (
                f"Could not initialize graph dataset: {type(exc).__name__}: {exc}"
            )

    @staticmethod
    def _normalize_transaction_id(value):
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        if isinstance(value, str):
            value = value.strip()
            try:
                number = float(value)
                if number.is_integer():
                    return int(number)
            except ValueError:
                pass
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value

    @staticmethod
    def _is_missing(value):
        try:
            return bool(pd.isna(value))
        except (TypeError, ValueError):
            return value is None

    def _iter_chunks(self):
        rows_seen = 0
        for chunk in pd.read_csv(
            self.dataset_path,
            usecols=self.graph_columns,
            chunksize=self.chunk_size,
            low_memory=False,
        ):
            if self.max_rows is not None:
                remaining = self.max_rows - rows_seen
                if remaining <= 0:
                    break
                if len(chunk) > remaining:
                    chunk = chunk.iloc[:remaining]
            rows_seen += len(chunk)
            yield chunk
            if self.max_rows is not None and rows_seen >= self.max_rows:
                break

    def _lookup_transaction_record(self, transaction_id):
        if not self.available:
            return None
        for chunk in self._iter_chunks():
            ids = chunk["TransactionID"].map(self._normalize_transaction_id)
            matches = chunk.loc[ids == transaction_id]
            if not matches.empty:
                return matches.iloc[0].to_dict()
        return None

    def investigate_transaction(self, transaction_id):
        transaction_id = self._normalize_transaction_id(transaction_id)
        base = {
            "transaction_found": False,
            "transaction_id": transaction_id,
            "in_graph_sample": False,
            "proxy_neighbor_count": 0,
            "proxy_neighbors": [],
            "proxy_relationships": [],
            "relationship_type": "shared card1 proxy; not verified account identity",
            "fraud_ring_detected": False,
            "hub_detected": False,
            "finding_validated": False,
        }
        if not self.available:
            base["status"] = (
                "Graph investigation unavailable. "
                + (self.initialization_error or "Graph dataset is unavailable.")
                + " Risk prediction and SHAP explanations can still run."
            )
            return base
        if transaction_id is None:
            base["status"] = "Graph lookup not performed: no valid TransactionID was supplied."
            return base

        try:
            record = self._lookup_transaction_record(transaction_id)
            if record is None:
                base["status"] = "Transaction ID was not found in the graph dataset."
                return base

            base["transaction_found"] = True
            base["in_graph_sample"] = True  # relationships are computed on demand
            for col in ("TransactionAmt", "TransactionDT"):
                value = record.get(col)
                if value is not None and not self._is_missing(value):
                    base[col] = float(value)

            card_value = record.get("card1")
            if self._is_missing(card_value):
                base["status"] = (
                    "Transaction found, but card1 is missing; no shared-card proxy links can be computed."
                )
                return base

            # Keep only up to max_group_size + 1 records. If the group exceeds
            # the cap, suppress links rather than creating a misleading partial graph.
            peers = []
            group_too_large = False
            for chunk in self._iter_chunks():
                same_card = chunk.loc[chunk["card1"].eq(card_value)]
                for _, row in same_card.iterrows():
                    peer_id = self._normalize_transaction_id(row.get("TransactionID"))
                    if peer_id is None:
                        continue
                    if not any(item["transaction_id"] == peer_id for item in peers):
                        peers.append({
                            "transaction_id": peer_id,
                            "card1": row.get("card1"),
                            "TransactionAmt": row.get("TransactionAmt"),
                            "TransactionDT": row.get("TransactionDT"),
                        })
                    if len(peers) > self.max_group_size:
                        group_too_large = True
                        break
                if group_too_large:
                    break

            if group_too_large:
                base["status"] = (
                    "Transaction found. Its shared-card proxy group exceeds the configured "
                    f"safety limit of {self.max_group_size} transactions; links were suppressed. "
                    "This is not evidence of a fraud ring."
                )
                base["proxy_group_limit_exceeded"] = True
                return base

            peer_ids = [p["transaction_id"] for p in peers if p["transaction_id"] != transaction_id]
            base["proxy_neighbor_count"] = len(peer_ids)
            base["proxy_neighbors"] = peer_ids[:20]
            for peer in peers:
                if peer["transaction_id"] == transaction_id or len(base["proxy_relationships"]) >= 20:
                    continue
                relationship = {
                    "transaction_id": peer["transaction_id"],
                    "relationship": "shared_card1_proxy",
                    "shared_card1": peer["card1"],
                }
                for col in ("TransactionAmt", "TransactionDT"):
                    value = peer.get(col)
                    if value is not None and not self._is_missing(value):
                        relationship[col] = float(value)
                base["proxy_relationships"].append(relationship)
            base["status"] = (
                "Exploratory shared-card proxy links only. "
                "Account ownership and fraud rings are not confirmed."
            )
            # Populate a tiny, per-investigation graph for backwards compatibility.
            self.graph = nx.Graph()
            self.graph.add_node(transaction_id)
            for peer_id in peer_ids:
                self.graph.add_edge(
                    transaction_id, peer_id,
                    relationship="shared_card1_proxy", shared_card1=card_value,
                )
            return base
        except Exception as exc:
            base["status"] = (
                f"Graph investigation failed: {type(exc).__name__}: {exc}. "
                "Risk prediction and SHAP explanations may still be available."
            )
            return base

    def investigate_account(self, account_id):
        """Backward-compatible alias; the value is treated as TransactionID."""
        return self.investigate_transaction(account_id)


if __name__ == "__main__":
    agent = GraphInvestigationAgent()
    print("Dataset:", agent.dataset_path)
    print("Graph available:", agent.available)
    if agent.initialization_error:
        print("Graph status:", agent.initialization_error)
    print("Dataset loaded into memory: no")
    if agent.available:
        sample = None
        for chunk in agent._iter_chunks():
            if not chunk.empty:
                sample = chunk["TransactionID"].iloc[0]
                break
        if sample is not None:
            print("Sample investigation:")
            print(agent.investigate_transaction(sample))
