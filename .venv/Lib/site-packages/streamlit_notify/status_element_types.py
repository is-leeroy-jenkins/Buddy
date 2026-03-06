# Define notification widgets
import streamlit as st
from .status_elements import RerunnableStatusElement
from enum import Enum

toast = RerunnableStatusElement(st.toast)
balloons = RerunnableStatusElement(st.balloons)
snow = RerunnableStatusElement(st.snow)
success = RerunnableStatusElement(st.success)
info = RerunnableStatusElement(st.info)
error = RerunnableStatusElement(st.error)
warning = RerunnableStatusElement(st.warning)
exception = RerunnableStatusElement(st.exception)


class NotificationType(Enum):
    """
    Enum class for notification types.
    """

    TOAST = "toast"
    BALLOONS = "balloons"
    SNOW = "snow"
    SUCCESS = "success"
    INFO = "info"
    ERROR = "error"
    WARNING = "warning"
    EXCEPTION = "exception"


STATUS_ELEMENTS = {
    NotificationType.TOAST.value: toast,
    NotificationType.BALLOONS.value: balloons,
    NotificationType.SNOW.value: snow,
    NotificationType.SUCCESS.value: success,
    NotificationType.INFO.value: info,
    NotificationType.ERROR.value: error,
    NotificationType.WARNING.value: warning,
    NotificationType.EXCEPTION.value: exception,
}
