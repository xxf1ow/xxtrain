from __future__ import annotations

import logging
from threading import Event, Thread

from xxtrain.platform.training_service import TrainingService

_INTERVAL_SECONDS = 5
_LOGGER = logging.getLogger(__name__)


class TrainingCoordinator:
    """Run durable training reconciliation until application shutdown.

    Starting creates one thread and performs the first pass immediately. Closing prevents another pass and waits for
    an in-flight reconciliation call to return.
    """

    def __init__(self, service: TrainingService) -> None:
        self._service = service
        self._stop = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        """Start the single reconciliation thread once."""
        if self._thread is not None:
            return
        self._thread = Thread(target=self._run, name='xxtrain-training-coordinator')
        self._thread.start()

    def close(self) -> None:
        """Request shutdown and wait for the reconciliation thread to exit."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._service.reconcile_pending()
            except Exception:
                _LOGGER.exception('Training coordination pass failed')
            if self._stop.wait(_INTERVAL_SECONDS):
                break
