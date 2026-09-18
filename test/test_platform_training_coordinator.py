import threading
import unittest

from xxtrain.platform.training_coordinator import TrainingCoordinator


class ControlledService:
    def __init__(self) -> None:
        self.calls = 0
        self.entered = threading.Event()
        self.release = threading.Event()

    def reconcile_pending(self) -> None:
        self.calls += 1
        self.entered.set()
        self.release.wait()


class AdvancingStop:
    def __init__(self) -> None:
        self.stopped = threading.Event()
        self.waited = threading.Event()
        self.wait_calls = 0

    def is_set(self) -> bool:
        return self.stopped.is_set()

    def set(self) -> None:
        self.stopped.set()

    def wait(self, timeout: float) -> bool:
        self.wait_calls += 1
        self.waited.set()
        if self.wait_calls == 1:
            return False
        return self.stopped.wait(1)


class TrainingCoordinatorTest(unittest.TestCase):
    def test_start_reconciles_immediately_and_close_joins_inflight_pass(self) -> None:
        service = ControlledService()
        coordinator = TrainingCoordinator(service)
        coordinator.start()
        self.addCleanup(service.release.set)
        self.addCleanup(coordinator.close)
        self.assertTrue(service.entered.wait(1))

        closed = threading.Event()
        closer = threading.Thread(target=lambda: (coordinator.close(), closed.set()))
        closer.start()
        self.addCleanup(closer.join, 1)
        self.assertFalse(closed.wait(0.05))

        service.release.set()
        self.assertTrue(closed.wait(1))
        closer.join(1)
        self.assertEqual(1, service.calls)

    def test_failure_is_logged_and_a_later_pass_still_executes(self) -> None:
        class FailingOnceService:
            def __init__(self) -> None:
                self.calls = 0
                self.second = threading.Event()

            def reconcile_pending(self) -> None:
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError('controlled failure')
                self.second.set()

        service = FailingOnceService()
        coordinator = TrainingCoordinator(service)
        stop = AdvancingStop()
        coordinator._stop = stop
        with self.assertLogs('xxtrain.platform.training_coordinator', level='ERROR') as logs:
            coordinator.start()
            self.addCleanup(coordinator.close)
            self.assertTrue(service.second.wait(1))
            coordinator.close()

        self.assertEqual(2, service.calls)
        self.assertEqual(1, len(logs.output))
        self.assertIn('Training coordination pass failed', logs.output[0])


if __name__ == '__main__':
    unittest.main()
