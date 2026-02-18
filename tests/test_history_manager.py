import asyncio

from fury import HistoryManager, StaticHistoryManager


class FakeCompletion:
    def __init__(self, content: str) -> None:
        message = type("Message", (), {"content": content})
        choice = type("Choice", (), {"message": message})
        self.choices = [choice]


class FakeChatCompletions:
    def __init__(self) -> None:
        self.last_request = None

    async def create(self, **kwargs):
        self.last_request = kwargs
        return FakeCompletion("Summary content")


class FakeChat:
    def __init__(self) -> None:
        self.completions = FakeChatCompletions()


class FakeClient:
    def __init__(self) -> None:
        self.chat = FakeChat()


def test_history_manager_auto_compaction():
    client = FakeClient()
    manager = HistoryManager(
        client=client,
        summary_model="fake-model",
        auto_compact=True,
        context_window=60,
        reserve_tokens=10,
        keep_recent_tokens=10,
        summary_prefix="Summary:",
    )

    async def run():
        for idx in range(6):
            await manager.add({"role": "user", "content": f"hello {'x' * 50} {idx}"})
            await manager.add(
                {"role": "assistant", "content": f"reply {'y' * 50} {idx}"}
            )
        return manager.history

    history = asyncio.run(run())

    assert history[0]["role"] == "system"
    assert history[0]["content"].startswith("Summary:")
    assert history[-1]["content"].endswith("5")
    assert client.chat.completions.last_request is not None
    assert len(history) < 12


def test_static_history_manager_fits_initial_history():
    manager = StaticHistoryManager(
        target_context_length=10,
        history=[
            {"role": "user", "content": "a" * 16},  # ~4 tokens
            {"role": "assistant", "content": "b" * 16},  # ~4 tokens
            {"role": "user", "content": "c" * 16},  # ~4 tokens
        ],
    )

    assert len(manager.history) == 2
    assert manager.history[0]["content"] == "b" * 16
    assert manager.history[1]["content"] == "c" * 16


def test_static_history_manager_add_drops_older_messages():
    manager = StaticHistoryManager(
        target_context_length=8,
        history=[
            {"role": "user", "content": "a" * 16},
            {"role": "assistant", "content": "b" * 16},
        ],
    )

    asyncio.run(manager.add({"role": "user", "content": "c" * 16}))

    assert len(manager.history) == 2
    assert manager.history[0]["content"] == "b" * 16
    assert manager.history[1]["content"] == "c" * 16
