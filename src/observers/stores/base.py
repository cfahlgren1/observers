from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from observers.base import Record


@dataclass
class Store(ABC):
    """
    Base class for storing records
    """

    @abstractmethod
    def add(self, record: "Record"):
        """Add a new record to the store"""
        pass

    @abstractmethod
    async def add_async(self, record: "Record"):
        """Add a new record to the store asynchronously"""
        pass

    @abstractmethod
    def connect(self):
        """Connect to the store"""
        pass

    @abstractmethod
    def _init_table(self, record: "Record"):
        """Initialize the table"""
        pass
