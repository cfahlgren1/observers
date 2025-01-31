import asyncio
import glob
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import duckdb

from observers.stores.sql_base import SQLStore

if TYPE_CHECKING:
    from observers.base import Record

DEFAULT_DB_NAME = "store.db"


@dataclass
class DuckDBStore(SQLStore):
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
            self._get_current_schema_version()
            self._apply_pending_migrations()

    @classmethod
    def connect(cls, path: Optional[str] = None) -> "DuckDBStore":
        """Create a new store instance with optional custom path"""
        if not path:
            path = os.path.join(os.getcwd(), DEFAULT_DB_NAME)
        return cls(path=path)

    def _init_table(self, record: "Record") -> str:
        self._conn.execute(record.duckdb_schema)
        self._tables.append(record.table_name)

    def _get_tables(self) -> List[str]:
        """Get all tables in the database"""
        return [table[0] for table in self._conn.execute("SHOW TABLES").fetchall()]

    def add(self, record: "Record"):
        """Add a new record to the database"""
        if record.table_name not in self._tables:
            self._init_table(record)

        record_dict = asdict(record)

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
            f"INSERT INTO {record.table_name} VALUES ({placeholders})",
            [
                record_dict[k] if k in record_dict else None
                for k in record.table_columns
            ],
        )

    async def add_async(self, record: "Record"):
        """Add a new record to the database asynchronously"""
        await asyncio.to_thread(self.add, record)

    def close(self) -> None:
        """Close the database connection"""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def _migrate_schema(self, migration_script: str):
        """Apply a schema migration"""
        self._conn.execute(migration_script)

    def _get_current_schema_version(self) -> int:
        """Get the current schema version, creating the table if it doesn't exist"""
        table_exists = self._conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'schema_version'"
        ).fetchone()[0]

        # create the schema_version table if it doesn't exist
        if not table_exists:
            self._conn.execute(
                """
                CREATE TABLE schema_version (
                    version INTEGER PRIMARY KEY,
                    migration_name VARCHAR,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            self._conn.execute(
                "INSERT INTO schema_version (version, migration_name) VALUES (0, 'initial')"
            )

        # retrieve the current schema version
        result = self._conn.execute(
            "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
        ).fetchone()
        return result[0] if result else 0

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _get_migrations_path(self) -> Path:
        """Get the path to migrations directory"""
        return Path(__file__).parent / "migrations"

    def _get_available_migrations(self) -> List[tuple[int, str]]:
        """Get all available migration files sorted by version"""
        migrations_path = self._get_migrations_path()
        migration_files = glob.glob(str(migrations_path / "*.sql"))

        # extract version and path using regex
        migrations = []
        for file_path in migration_files:
            # Match migration files in format: any_prefix_NUMBER_any_suffix.sql
            # e.g., "001_create_users.sql" or "v1_init.sql" - extracts "1" as version
            if match := re.match(r".*?(\d+)_.+\.sql$", file_path):
                version = int(match.group(1))
                migrations.append((version, file_path))

        return sorted(migrations)

    def _apply_pending_migrations(self):
        """Apply any pending migrations"""
        current_version = self._get_current_schema_version()
        available_migrations = self._get_available_migrations()

        for version, migration_path in available_migrations:
            if version > current_version:
                with open(migration_path, "r") as f:
                    migration_script = f.read()

                migration_name = Path(
                    migration_path
                ).stem  # Gets filename without extension

                self._conn.execute("BEGIN TRANSACTION")
                try:
                    self._migrate_schema(migration_script)
                    self._conn.execute(
                        "INSERT INTO schema_version (version, migration_name) VALUES (?, ?)",
                        [version, migration_name],
                    )
                    self._conn.execute("COMMIT")
                except Exception as e:
                    self._conn.execute("ROLLBACK")
                    raise Exception(f"Migration {version} failed: {str(e)}")

    def _check_table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database"""
        result = self._conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table_name],
        ).fetchone()[0]
        return bool(result)

    def _create_version_table(self):
        """Create the schema version table"""
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                migration_name VARCHAR,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

    def _execute(self, query: str, params: Optional[List] = None):
        """Execute a SQL query"""
        return self._conn.execute(query, params if params else [])
