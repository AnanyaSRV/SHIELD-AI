
from pathlib import Path

import networkx as nx
import pandas as pd


class GraphInvestigationAgent:
    """
    Exploratory graph investigation for IEEE-CIS transactions.

    Nodes represent transactions. Edges connect transactions sharing
    a non-null card1 value. card1 is a proxy identifier, NOT a verified
    account identifier. Findings must not be treated as confirmed
    fraud rings or account relationships.
    """

    def __init__(self, dataset_path=None, max_rows=50000):
        project_root = Path(__file__).resolve().parents[1]

        self.dataset_path = Path(
            dataset_path or project_root / "data" / "train_transaction.csv"
        )
        self.max_rows = max_rows
        self.df = pd.read_csv(self.dataset_path, nrows=max_rows)

        required = {"TransactionID", "card1"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(
                f"Missing required graph columns: {sorted(missing)}"
            )

        self.graph = nx.Graph()
        self._build_graph()

    def _build_graph(self):
        # Add transactions as nodes. Missing card values are not linked.
        for row in self.df.itertuples(index=False):
            tx_id = getattr(row, "TransactionID")
            self.graph.add_node(tx_id)

        valid = self.df.dropna(subset=["card1"])

        # Connect transactions sharing card1. Group size is capped to
        # avoid creating a huge clique from common/shared proxy values.
        for card_value, group in valid.groupby("card1", sort=False):
            tx_ids = group["TransactionID"].tolist()

            # A shared card value is only a proxy relationship.
            # Avoid quadratic edge growth for very large groups.
            if len(tx_ids) < 2 or len(tx_ids) > 100:
                continue

            for i in range(len(tx_ids)):
                for j in range(i + 1, len(tx_ids)):
                    left, right = tx_ids[i], tx_ids[j]
                    self.graph.add_edge(
                        left,
                        right,
                        relationship="shared_card1_proxy",
                        card1=str(card_value),
                    )

        self.card1_map = {}
        for card_value, group in valid.groupby("card1", sort=False):
            self.card1_map[card_value] = group["TransactionID"].tolist()

    def investigate_transaction(self, transaction_id):
        if transaction_id not in self.graph:
            return {
                "transaction_found": False,
                "status": "Transaction not found in graph sample.",
                "relationship_type": "shared card1 proxy",
            }

        neighbors = list(self.graph.neighbors(transaction_id))
        node = self.df.loc[
            self.df["TransactionID"] == transaction_id
        ]

        result = {
            "transaction_found": True,
            "transaction_id": transaction_id,
            "proxy_neighbor_count": len(neighbors),
            "proxy_neighbors": neighbors[:20],
            "relationship_type": "shared card1 proxy; not verified account identity",
            "fraud_ring_detected": False,
            "status": (
                "Exploratory proxy links only. This implementation does not "
                "confirm fraud rings or account ownership."
            ),
        }

        if not node.empty and "isFraud" in node.columns:
            # This field is shown only for offline dataset analysis.
            result["offline_dataset_label"] = int(node.iloc[0]["isFraud"])

        return result

    def investigate_account(self, account_id):
        # Compatibility with the previous method name. The input is a
        # transaction ID in this implementation, not an account ID.
        return self.investigate_transaction(account_id)


if __name__ == "__main__":
    agent = GraphInvestigationAgent()
    print("Graph sample rows:", len(agent.df))
    print("Graph nodes:", agent.graph.number_of_nodes())
    print("Graph edges:", agent.graph.number_of_edges())

    if not agent.df.empty:
        sample_id = agent.df["TransactionID"].iloc[0]
        print(agent.investigate_transaction(sample_id))
