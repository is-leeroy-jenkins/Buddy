"""
Queue management for Streamlit notifications.
"""

from typing import Dict, Generator, Iterable, List, Literal, Optional, Union, Callable
import copy

import streamlit as st

from .notification_dataclass import StatusElementNotification


def default_sort_func(x: StatusElementNotification) -> int:
    """Sort notifications by priority (highest first)."""
    return -x.priority


class NotificationQueue:
    """
    A queue for managing Streamlit notifications.
    """

    def __init__(
        self,
        queue_name: str,
        sort_func: Optional[Callable[[StatusElementNotification], int]] = None,
    ) -> None:
        """Initialize the queue."""
        self._queue_name: str = queue_name

        if sort_func is None:
            sort_func = default_sort_func
        self._sort_func: Callable[[StatusElementNotification], int] = sort_func

        self.ensure_queue()

    @property
    def queue(self) -> List[StatusElementNotification]:
        """Get the current queue."""
        self.ensure_queue()
        return st.session_state[self._queue_name]

    @property
    def queue_name(self) -> str:
        """Get the name of the queue."""
        return self._queue_name

    @property
    def sort_func(self) -> Callable[[StatusElementNotification], int]:
        """Get the sorting function for the queue."""
        return self._sort_func

    def ensure_queue(self) -> None:
        """Ensure the queue exists in session state."""
        if self._queue_name not in st.session_state:
            st.session_state[self._queue_name] = []

    def _sort(self) -> None:
        """Sort the queue by priority."""
        self.queue.sort(key=self._sort_func)

    def has_items(self) -> bool:
        """Check if the queue has items."""
        return len(self) > 0

    def is_empty(self) -> bool:
        """Check if the queue is empty."""
        return len(self) == 0

    def append(self, item: StatusElementNotification) -> None:
        """Add an item to the queue."""
        self.queue.append(item)
        self._sort()

    def extend(self, items: Iterable[StatusElementNotification]) -> None:
        """Add multiple items to the queue."""
        self.queue.extend(items)
        self._sort()

    def remove(self, item: Union[StatusElementNotification, int]) -> None:
        """Remove an item from the queue."""

        if isinstance(item, int):
            self.queue.pop(item)
            return

        if item in self.queue:
            self.queue.remove(item)
            return

        raise ValueError(f"Item: {item} not found in queue.")

    def contains(self, item: StatusElementNotification) -> bool:
        """Check if an item is in the queue."""
        return item in self.queue

    def _get_all_less_than(self, priority: int) -> List[StatusElementNotification]:
        """Get all items in the queue with priority less than the specified value."""
        return [item for item in self.queue if item.priority < priority]

    def _get_all_less_than_equal_to(
        self, priority: int
    ) -> List[StatusElementNotification]:
        """Get all items in the queue with priority less than or equal to the specified value."""
        return [item for item in self.queue if item.priority <= priority]

    def _get_all_greater_than(self, priority: int) -> List[StatusElementNotification]:
        """Get all items in the queue with priority greater than the specified value."""
        return [item for item in self.queue if item.priority > priority]

    def _get_all_greater_than_equal_to(
        self, priority: int
    ) -> List[StatusElementNotification]:
        """Get all items in the queue with priority greater than or equal to the specified value."""
        return [item for item in self.queue if item.priority >= priority]

    def _get_all_equal_to(self, priority: int) -> List[StatusElementNotification]:
        """Get all items in the queue with priority equal to the specified value."""
        return [item for item in self.queue if item.priority == priority]

    def get_all(
        self,
        priority: Optional[int] = None,
        priority_type: Literal["le", "lt", "ge", "gt", "eq"] = "eq",
    ) -> List[StatusElementNotification]:
        """Get all items in the queue."""
        if priority is not None:
            if priority_type == "le":
                return self._get_all_less_than_equal_to(priority)
            elif priority_type == "lt":
                return self._get_all_less_than(priority)
            elif priority_type == "ge":
                return self._get_all_greater_than_equal_to(priority)
            elif priority_type == "gt":
                return self._get_all_greater_than(priority)
            elif priority_type == "eq":
                return self._get_all_equal_to(priority)
        return self.queue.copy()

    def clear(self) -> None:
        """Clear the queue."""
        self.queue.clear()

    def pop(self, index: int = 0) -> StatusElementNotification:
        """Pop an item from the queue."""
        return self.queue.pop(index)

    def get(self, index: int = 0) -> StatusElementNotification:
        """Get an item from the queue without removing it."""
        return self.queue[index]

    def size(self) -> int:
        """Get the size of the queue."""
        return len(self)

    def __len__(self) -> int:
        """Get the size of the queue."""
        return len(self.queue)

    def __repr__(self) -> str:
        """String representation of the queue."""
        return f"NotificationQueue(name={self._queue_name!r}, items={len(self.queue)})"

    def __str__(self) -> str:
        """String representation of the queue."""
        return f"NotificationQueue({self._queue_name}, {len(self.queue)} items)"

    def __bool__(self) -> bool:
        """Boolean representation of the queue."""
        return len(self) > 0

    def __contains__(self, item: StatusElementNotification) -> bool:
        """Check if an item is in the queue."""
        return self.contains(item)

    def __getitem__(self, index: int) -> StatusElementNotification:
        """Get an item by index."""
        return self.queue[index]

    def __setitem__(self, index: int, value: StatusElementNotification) -> None:
        """Set an item by index."""
        self.queue[index] = value
        self._sort()

    def __delitem__(self, index: int) -> None:
        """Delete an item by index."""
        del self.queue[index]

    def __hash__(self) -> int:
        """Hash of the queue based on its name."""
        return hash(self._queue_name)

    def __eq__(self, other: object) -> bool:
        """Check if this queue is equal to another."""
        if not isinstance(other, NotificationQueue):
            return False
        return (
            self._queue_name == other._queue_name and self.get_all() == other.get_all()
        )

    def __ne__(self, other: object) -> bool:
        """Check if this queue is not equal to another."""
        return not self.__eq__(other)

    def __lt__(self, other: object) -> bool:
        """Check if this queue is less than another."""
        if not isinstance(other, NotificationQueue):
            return NotImplemented
        return self.size() < other.size()

    def __iter__(self) -> Generator[StatusElementNotification, None, None]:
        """Iterate over the notifications in the queue."""
        yield from self.queue.copy()

    def __reversed__(self) -> Generator[StatusElementNotification, None, None]:
        """Iterate over the notifications in reverse order."""
        yield from reversed(self.queue.copy())

    def __copy__(self):
        """Create a shallow copy of the queue."""
        new_queue = NotificationQueue(f"{self._queue_name}_copy")
        new_queue._queue_name = self._queue_name  # Keep original name for copy
        new_queue.extend(self.get_all())
        return new_queue

    def __deepcopy__(self, memo: Dict[int, object]):
        """Create a deep copy of the queue."""
        new_queue = NotificationQueue(f"{self._queue_name}_copy")
        new_queue._queue_name = self._queue_name  # Keep original name for copy
        new_queue.extend(copy.deepcopy(self.get_all(), memo))
        return new_queue
