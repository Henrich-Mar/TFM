import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from game_interface import GameServerCluster


class _FakeSession:
    def __init__(self):
        self.closed = False
        self.close_calls = 0

    async def close(self):
        self.close_calls += 1
        self.closed = True


class TestSessionRecycleLifecycle(unittest.TestCase):
    def test_recycle_keeps_old_session_alive_until_shutdown(self):
        original_grace = os.environ.get("TM_RECYCLE_SESSION_GRACE_SEC")
        os.environ["TM_RECYCLE_SESSION_GRACE_SEC"] = "120"

        async def _run() -> None:
            cluster = GameServerCluster(["localhost:8080"])
            old_session = _FakeSession()
            new_session = _FakeSession()

            cluster.session = old_session
            cluster._build_session = lambda timeout_total=None: new_session

            await cluster.recycle_session()

            self.assertIs(cluster.session, new_session)
            self.assertFalse(old_session.closed)
            self.assertEqual(old_session.close_calls, 0)
            self.assertEqual(len(cluster._retired_sessions), 1)

            await cluster.close()

            self.assertIsNone(cluster.session)
            self.assertTrue(old_session.closed)
            self.assertTrue(new_session.closed)
            self.assertEqual(len(cluster._retired_sessions), 0)

        try:
            asyncio.run(_run())
        finally:
            if original_grace is None:
                os.environ.pop("TM_RECYCLE_SESSION_GRACE_SEC", None)
            else:
                os.environ["TM_RECYCLE_SESSION_GRACE_SEC"] = original_grace


if __name__ == "__main__":
    unittest.main()