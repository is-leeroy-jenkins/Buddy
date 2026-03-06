"""
Initialization module for the st_notify package.
"""

__version__ = "0.3.1"

from typing import Any
import streamlit as st

from .status_elements import RerunnableStatusElement  # type: ignore
from .functional import (
    create_notification,  # type: ignore
    notify,  # type: ignore
    get_notifications,  # type: ignore
    clear_notifications,  # type: ignore
    get_notification_queue,  # type: ignore
    has_notifications,  # type: ignore
    remove_notifications,  # type: ignore
    add_notifications,  # type: ignore
)
from .functional import (
    toast_stn,  # type: ignore
    balloons_stn,  # type: ignore
    snow_stn,  # type: ignore
    success_stn,  # type: ignore
    info_stn,  # type: ignore
    error_stn,  # type: ignore
    warning_stn,  # type: ignore
    exception_stn,  # type: ignore
)
from .notification_queue import NotificationQueue  # type: ignore
from .notification_dataclass import StatusElementNotification  # type: ignore
from .status_element_types import (
    STATUS_ELEMENTS,
    NotificationType,  # type: ignore
    toast,  # type: ignore
    balloons,  # type: ignore
    snow,  # type: ignore
    success,  # type: ignore
    info,  # type: ignore
    error,  # type: ignore
    warning,  # type: ignore
    exception,  # type: ignore
)

from .utils import get_status_element  # type: ignore


def init_session_state() -> None:
    """
    Initialize session state for all notification elements.
    This ensures that the notification queues are set up in the session state.
    """
    for _, element in STATUS_ELEMENTS.items():
        element.setup_queue()


init_session_state()


def __getattr__(name: str) -> Any:
    """
    Delegate attribute access to Streamlit if not found in this module.

    Parameters:
        name (str): Name of the attribute to get.

    Returns:
        Any: The requested attribute from Streamlit.

    Raises:
        AttributeError: If the attribute is not found in Streamlit.
    """
    try:
        return getattr(st, name)
    except AttributeError as err:
        raise AttributeError(str(err).replace("streamlit", "st_notify")) from err
