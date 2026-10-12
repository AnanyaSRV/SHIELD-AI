from pathlib import Path

import networkx as nx
import pandas as pd


class GraphInvestigationAgent:
    """
    Exploratory graph investigation for IEEE-CIS transactions.

    Transactions are connected when they share a non-null card1 value.
    card1 is a proxy feature, not a verified account identifier.

    The offline isFraud label is never loaded into graph records or
    exposed as investigation evidence.

    If the source CSV is not deployed, graph investigation is marked
    unavailable instead of preventing the rest of SHIELD-AI from running.
    """

    def __init__(
        self,
        dataset_path=None,
        max_rows=None,
        max_group_size=100,
        cache_full_dataset=False,
    ):
        project_root = Path(__file__).resolve().parents[1]
        self.dataset_path = Path(
            dataset_path or project_root / "data" / "train_transaction.csv"
        )

        if max_rows is not None and max_rows < 1:
            raise ValueError("max_rows must be a positive integer or None.")
        if max_group_size < 2:
            raise ValueError("max_group_size must be at least 2.")

        self.max_rows = max_rows
        self.max_group_size = max_group_size
        self.cache_full_dataset = cache_full_dataset
        self.graph = nx.Graph()
        self.card1_map = {}
        self.transaction_index = {}
        self._full_dataset = None
        self.available_columns = []
        self.graph_columns = []
        self.df = pd.DataFrame()
        self.available = False
        self.initialization_error = None

        # A missing large training CSV is a deployment/configuration issue,
        # not a reason to break the risk-scoring and SHAP workflow.
        if not self.dataset_path.is_file():
            self.initialization_error = (
                f"Graph dataset not found: {self.dataset_path}"
            )
            return

        try:
            columns = pd.read_csv(self.dataset_path, nrows=0).columns.tolist()
            required = {"TransactionID", "card1"}
            missing = required - set(columns)
            if missing:
                self.initialization_error = (
                    f"Graph dataset is missing required columns: {sorted(missing)}"
                )
                return

            self.available_columns = columns

            # Explicit allowlist: never load isFraud into graph data.
            self.graph_columns = [
                column
                for column in (
                    "TransactionID",
                    "card1",
                    "TransactionAmt",
                    "TransactionDT",
                )
                if column in columns
            ]

            self.df = pd.read_csv(
                self.dataset_path,
                usecols=self.graph_columns,
                nrows=self.max_rows,
                low_memory=False,
            )
            self._build_graph()
            self.available = True
        except Exception as exc:
            self.initialization_error = (
                f"Could not initialize graph dataset: {type(exc).__name__}: {exc}"
            )
            self.df = pd.DataFrame()
            self.graph = nx.Graph()
            self.card1_map = {}
            self.transaction_index = {}
            self.available = False

    def _build_graph(self):
        """Build exploratory shared-card proxy edges."""
        for row in self.df.itertuples(index=False, name="Transaction"):
            record = row._asdict()
            transaction_id = record.get("TransactionID")
            if pd.isna(transaction_id):
                continue

            transaction_id = self._normalize_transaction_id(transaction_id)
            self.graph.add_node(transaction_id, transaction_id=transaction_id)
            self.transaction_index[transaction_id] = record

        valid = self.df.dropna(subset=["TransactionID", "card1"])
        for card_value, group in valid.groupby("card1", sort=False):
            transaction_ids = [
                self._normalize_transaction_id(value)
                for value in group["TransactionID"].tolist()
            ]
            self.card1_map[card_value] = transaction_ids

            # Prevent quadratic edge growth for large shared-card groups.
            if len(transaction_ids) < 2 or len(transaction_ids) > self.max_group_size:
                continue

            for i in range(len(transaction_ids)):
                for j in range(i + 1, len(transaction_ids)):
                    left = transaction_ids[i]
                    right = transaction_ids[j]
                    self.graph.add_edge(
                        left,
                        right,
                        relationship="shared_card1_proxy",
                        shared_card1=card_value,
                    )

    @staticmethod
    def _normalize_transaction_id(transaction_id):
        """Normalize integer-like transaction IDs."""
        if transaction_id is None:
            return None
        try:
            if pd.isna(transaction_id):
                return None
        except (TypeError, ValueError):
            pass

        if isinstance(transaction_id, str):
            transaction_id = transaction_id.strip()
            try:
                numeric = float(transaction_id)
                if numeric.is_integer():
                    return int(numeric)
            except ValueError:
                pass
            return transaction_id

        if isinstance(transaction_id, float) and transaction_id.is_integer():
            return int(transaction_id)
        return transaction_id

    def _lookup_transaction_record(self, transaction_id):
        """Search the graph sample, then the source CSV using allowlisted columns."""
        if not self.available:
            return None
        if transaction_id in self.transaction_index:
            return self.transaction_index[transaction_id]

        for chunk in pd.read_csv(
            self.dataset_path,
            usecols=self.graph_columns,
            chunksize=100_000,
            low_memory=False,
        ):
            normalized_ids = chunk["TransactionID"].map(
                self._normalize_transaction_id
            )
            matches = chunk.loc[normalized_ids == transaction_id]
            if not matches.empty:
                return matches.iloc[0].to_dict()
        return None

    def investigate_transaction(self, transaction_id):
        """
        Return transaction details and exploratory shared-card links.

        This does not validate account ownership or confirm fraud rings.
        """
        transaction_id = self._normalize_transaction_id(transaction_id)

        if not self.available:
            return {
                "transaction_found": False,
                "transaction_id": transaction_id,
                "in_graph_sample": False,
                "proxy_neighbor_count": 0,
                "proxy_neighbors": [],
                "proxy_relationships": [],
                "relationship_type": (
                    "shared card1 proxy; not verified account identity"
                ),
                "fraud_ring_detected": False,
                "hub_detected": False,
                "finding_validated": False,
                "status": (
                    "Graph investigation unavailable. "
                    + (self.initialization_error or "Graph dataset is unavailable.")
                    + " Risk prediction and SHAP explanations can still run."
                ),
            }

        if transaction_id is None:
            return {
                "transaction_found": False,
                "status": "Graph lookup not performed: no valid TransactionID was supplied.",
                "relationship_type": "shared card1 proxy",
            }

        record = self._lookup_transaction_record(transaction_id)
        if record is None:
            return {
                "transaction_found": False,
                "transaction_id": transaction_id,
                "status": "Transaction ID was not found in the graph dataset.",
                "relationship_type": "shared card1 proxy",
            }

        in_graph = transaction_id in self.graph
        result = {
            "transaction_found": True,
            "transaction_id": transaction_id,
            "in_graph_sample": in_graph,
            "proxy_neighbor_count": 0,
            "proxy_neighbors": [],
            "proxy_relationships": [],
            "relationship_type": (
                "shared card1 proxy; not verified account identity"
            ),
            "fraud_ring_detected": False,
            "hub_detected": False,
            "finding_validated": False,
            "status": (
                "Exploratory shared-card proxy links only. "
                "Account ownership and fraud rings are not confirmed."
            ),
        }

        for column in ("TransactionAmt", "TransactionDT"):
            value = record.get(column)
            if value is not None and not pd.isna(value):
                result[column] = float(value)

        # Never return isFraud or offline_dataset_label.
        if not in_graph:
            result["status"] += (
                " Transaction exists in the dataset but is outside "
                "the currently built graph sample; no neighbors were computed."
            )
            return result

        neighbors = list(self.graph.neighbors(transaction_id))
        result["proxy_neighbor_count"] = len(neighbors)
        result["proxy_neighbors"] = neighbors[:20]

        for neighbor_id in neighbors[:20]:
            edge = self.graph.get_edge_data(transaction_id, neighbor_id) or {}
            neighbor_record = self.transaction_index.get(neighbor_id, {})
            relationship = {
                "transaction_id": neighbor_id,
                "relationship": edge.get(
                    "relationship", "shared_card1_proxy"
                ),
                "shared_card1": edge.get("shared_card1"),
            }
            for column in ("TransactionAmt", "TransactionDT"):
                value = neighbor_record.get(column)
                if value is not None and not pd.isna(value):
                    relationship[column] = float(value)
            result["proxy_relationships"].append(relationship)

        return result

    def investigate_account(self, account_id):
        """
        Backward-compatible alias.

        The supplied value is interpreted as a TransactionID, not
        a verified account identifier.
        """
        return self.investigate_transaction(account_id)


if __name__ == "__main__":
    agent = GraphInvestigationAgent()
    print("Dataset:", agent.dataset_path)
    print("Graph available:", agent.available)
    if agent.initialization_error:
        print("Graph status:", agent.initialization_error)
    print("Rows loaded:", len(agent.df))
    print("Graph nodes:", agent.graph.number_of_nodes())
    print("Graph edges:", agent.graph.number_of_edges())
    if agent.available and not agent.df.empty:
        sample_id = agent.df["TransactionID"].iloc[0]
        print("\nSample investigation:")
        print(agent.investigate_transaction(sample_id))
