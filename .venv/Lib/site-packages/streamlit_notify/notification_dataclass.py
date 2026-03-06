"""
Streamlit notification objects.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class StatusElementNotification:
    """
    Represents a notification for a Streamlit widget.
    """

    base_widget: Callable[..., Any]
    args: OrderedDict[str, Any]
    priority: int = 0
    data: Any = None

    def notify(self) -> None:
        """Display the notification using the widget."""
        self.base_widget(**self.args)

    @property
    def name(self) -> str:
        """Get the name of the widget function."""
        return self.base_widget.__name__
