import asyncio

from fury import HistoryManager


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
