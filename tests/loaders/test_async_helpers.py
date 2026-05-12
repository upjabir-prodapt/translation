import pytest
from src.loaders.utils.async_helpers import AsyncBridge
from src.loaders.utils.async_helpers import sync_to_async


class TestAsyncHelpers:
    @pytest.mark.asyncio
    async def test_sync_to_async(self):
        def sync_func(x):
            return x + 1

        res = await sync_to_async(sync_func, 1)
        assert res == 2

    def test_async_bridge_run_async_sync_context(self):
        async def coro():
            return "ok"

        res = AsyncBridge.run_async(coro())
        assert res == "ok"

    def test_async_bridge_ensure_sync_result_sync_context(self):
        async def coro():
            return "ok"

        res = AsyncBridge.ensure_sync_result(coro())
        assert res == "ok"

    @pytest.mark.asyncio
    async def test_async_bridge_in_loop_error(self):
        async def coro():
            return "ok"

        c = coro()
        # In an active loop, ensure_sync_result should raise RuntimeError
        try:
            with pytest.raises(RuntimeError, match="Cannot get sync result"):
                AsyncBridge.ensure_sync_result(c)
        finally:
            c.close()
