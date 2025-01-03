import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

if TYPE_CHECKING:
    from argilla import Argilla


@dataclass
class Message:
    role: Literal["system", "user", "assistant", "function"]
    content: str


@dataclass
class Record(ABC):
    """
    Base class for storing model response information
    """

    client_name: str = field(init=False)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tags: List[str] = None
    properties: Dict[str, Any] = None
    error: Optional[str] = None
    raw_response: Optional[Dict] = None

    @property
    @abstractmethod
    def json_fields(self):
        """Return the DuckDB JSON fields for the record"""
        pass

    @property
    @abstractmethod
    def image_fields(self):
        """Return the DuckDB image fields for the record"""
        pass

    @property
    @abstractmethod
    def table_columns(self):
        """Return the DuckDB table columns for the record"""
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
