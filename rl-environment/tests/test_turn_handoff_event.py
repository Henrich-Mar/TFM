import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from game_interface import GameInstance, GameServer


def _instance() -> GameInstance:
    return GameInstance("g1", GameServer(host="localhost", port=8080), session=None)


def test_wait_returns_early_when_another_seat_submits():
    async def scenario():
        game = _instance()
        seq = game.input_seq
        started = time.perf_counter()
        waiter = asyncio.create_task(game.wait_for_input_after(seq, 5.0))
        await asyncio.sleep(0.01)
        game.notify_input_submitted()
        await waiter
        return time.perf_counter() - started

    assert asyncio.run(scenario()) < 1.0


def test_input_during_state_read_skips_the_wait():
    async def scenario():
        game = _instance()
        seq = game.input_seq
        game.notify_input_submitted()  # lands while the seat's GET is in flight
        started = time.perf_counter()
        await game.wait_for_input_after(seq, 5.0)
        return time.perf_counter() - started

    assert asyncio.run(scenario()) < 0.1


def test_wait_times_out_without_input():
    async def scenario():
        game = _instance()
        started = time.perf_counter()
        await game.wait_for_input_after(game.input_seq, 0.05)
        return time.perf_counter() - started

    assert 0.04 <= asyncio.run(scenario()) < 1.0
