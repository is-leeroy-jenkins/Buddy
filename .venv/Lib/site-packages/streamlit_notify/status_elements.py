"""
Widgets with notification queueing for Streamlit.
"""

import inspect
from typing import Any, Callable, Literal, Optional

from .notification_queue import NotificationQueue
from .notification_dataclass import StatusElementNotification


class RerunnableStatusElement:
    """
    A wrapper for Streamlit widgets to enable notification queueing.
    """

    def __init__(self, base_widget: Callable[..., Any]) -> None:
        """Initialize the wrapper."""
        self._base_widget = base_widget
        self._session_state_key = (
            f"ST_NOTIFY_{self._base_widget.__name__.upper()}_QUEUE"
        )
        self._queue = NotificationQueue(self._session_state_key)

    @property
    def session_state_key(self) -> str:
        """Get the session state key for the notification queue."""
        return self._session_state_key

    @property
    def base_widget(self) -> Callable[..., Any]:
        """Get the base widget function."""
        return self._base_widget

    @property
    def name(self) -> str:
        """Get the name of the base widget."""
        return self._base_widget.__name__

    @property
    def notifications(self) -> NotificationQueue:
        """Get the notification queue."""
        return self._queue

    def setup_queue(self) -> None:
        """Ensure the notification queue is set up in session state."""
        self._queue.ensure_queue()

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        """Add a notification to the queue."""
        notification = self.create_notification(*args, **kwargs)
        self._queue.append(notification)

    def create_notification(
        self, *args: Any, **kwargs: Any
    ) -> StatusElementNotification:
        """Create a notification without adding it to the queue."""
        priority = kwargs.pop("priority", 0)
        data = kwargs.pop("data", None)
        signature = inspect.signature(self._base_widget)
        bound_args = signature.bind_partial(*args, **kwargs)

        return StatusElementNotification(
            base_widget=self._base_widget,
            args=bound_args.arguments,
            priority=priority,
            data=data,
        )

    def notify(
        self,
        remove: bool = True,
        priority: Optional[int] = None,
        priority_type: Literal["le", "lt", "ge", "gt", "eq"] = "eq",
    ) -> None:
        """
        Display all queued notifications. Will display in order of priority and remove
        from queue if specified.
        """
        for notification in self.notifications.get_all(
            priority=priority, priority_type=priority_type
        ):
            notification.notify()
            if remove:
                self.notifications.remove(notification)

    def __repr__(self) -> str:
        """String representation of the wrapper."""
        return f"RerunnableStatusElement({self._base_widget.__name__})"

    def __str__(self) -> str:
        """String representation of the wrapper."""
        return f"RerunnableStatusElement({self._base_widget.__name__})"
