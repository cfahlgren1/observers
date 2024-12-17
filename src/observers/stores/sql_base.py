from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from observers.stores.base import Store


@dataclass
class SQLStore(Store):
    """Base class for SQL-based stores with migration capabilities"""

    @abstractmethod
    def _check_table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database"""
        pass

    @abstractmethod
    def _create_version_table(self):
        """Create the schema version table"""
        pass

    @abstractmethod
    def _execute(self, query: str, params: Optional[List] = None):
        """Execute a SQL query"""
        pass

    @abstractmethod
    def _migrate_schema(self, migration_script: str):
        """Execute a migration script"""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the database connection"""
        pass

    @abstractmethod
    def _get_current_schema_version(self) -> int:
        """Get the current schema version"""
        pass

    @abstractmethod
    def _apply_pending_migrations(self):
        """Apply any pending migrations"""
        pass
