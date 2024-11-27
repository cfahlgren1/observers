from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argilla import Argilla


@dataclass
class Record(ABC):
    """
    Base class for storing model response information
    """

    @property
    @abstractmethod
    def json_fields(self):
        """Return the DuckDB JSON fields for the record"""
        pass

    @property
    @abstractmethod
    def duckdb_schema(self):
        """Return the DuckDB schema for the record"""
        pass

    @property
    @abstractmethod
    def table_name(self):
        """Return the DuckDB table name for the record"""
        pass

    @abstractmethod
    def argilla_settings(self, client: "Argilla"):
        """Return the Argilla settings for the record"""
        pass
