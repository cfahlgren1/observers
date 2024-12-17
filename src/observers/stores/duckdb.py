import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, List, Optional

import duckdb

from observers.stores.base import Store

if TYPE_CHECKING:
    from observers.base import Record

DEFAULT_DB_NAME = "store.db"


@dataclass
class DuckDBStore(Store):
    """
    DuckDB store
    """

    path: str = field(
        default_factory=lambda: os.path.join(os.getcwd(), DEFAULT_DB_NAME)
    )
    _tables: List[str] = field(default_factory=list)
    _conn: Optional[duckdb.DuckDBPyConnection] = None

    def __post_init__(self):
        """Initialize database connection and table"""
        if self._conn is None:
            self._conn = duckdb.connect(self.path)
            self._tables = self._get_tables()

    @classmethod
    def connect(cls, path: Optional[str] = None) -> "DuckDBStore":
        """Create a new store instance with optional custom path"""
        if not path:
            path = os.path.join(os.getcwd(), DEFAULT_DB_NAME)
        return cls(path=path)

    def _init_table(self, record: "Record") -> str:
        table_name = f"{record.table_name}_{uuid.uuid4().hex[:8]}"
        duckdb_schema = record.duckdb_schema.replace(record.table_name, table_name)
        self._conn.execute(duckdb_schema)
        self._tables.append(table_name)
        return table_name

    def _get_tables(self) -> List[str]:
        """Get all tables in the database"""
        return [table[0] for table in self._conn.execute("SHOW TABLES").fetchall()]

    def add(self, record: "Record"):
        """Add a new record to the database"""
        table_name = next(
            (table for table in self._tables if record.table_name in table), None
        )
        if not table_name:
            table_name = self._init_table(record)

        record_dict = asdict(record)
        record_dict["synced_at"] = None

        for json_field in record.json_fields:
            if record_dict[json_field]:
                record_dict[json_field] = json.dumps(record_dict[json_field])

        placeholders = ", ".join(
            ["$" + str(i + 1) for i in range(len(record.table_columns))]
        )

        # Sort record_dict based on table_columns order
        if hasattr(record, "table_columns"):
            sorted_dict = {k: record_dict[k] for k in record.table_columns}
            record_dict = sorted_dict

        self._conn.execute(
            f"INSERT INTO {table_name} VALUES ({placeholders})",
            [
                record_dict[k] if k in record_dict else None
                for k in record.table_columns
            ],
        )

    def get_unsynced(self, table_name: str) -> List[tuple]:
        """Retrieve unsynced records"""
        return self._conn.execute(
            f"SELECT * FROM {table_name} WHERE synced_at IS NULL"
        ).fetchall()

    def mark_as_synced(self, record_ids: List[str], table_name: str) -> None:
        """Mark specified records as synced"""
        self._conn.execute(
            f"UPDATE {table_name} SET synced_at = CURRENT_TIMESTAMP WHERE id = ANY($1)",
            [record_ids],
        )

    def close(self) -> None:
        """Close the database connection"""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
