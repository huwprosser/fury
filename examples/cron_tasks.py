import asyncio
from datetime import datetime
from fury import Agent, create_tool
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger


def check_time():
    now = datetime.now().isoformat(timespec="seconds")
    return {"current_time": now}


background_tasks = [
    {
        "name": "check_time",
        "prompt": "What time is it right now? Use the check_time tool. If it's in the morning, say 'good morning' and if it's in the afternoon, say 'good afternoon'.",
        "tools": [
            create_tool(
                "check_time",
                "Return the current local time as an ISO-8601 string.",
                check_time,
                "Checking the time...",
                {"type": "object", "properties": {}, "required": []},
                {"type": "object", "properties": {}, "required": []},
            )
        ],
        "schedule": "*/10 * * * * *",
        "reasoning": False,
    }
]


async def run_job(task: dict):
    print(f"\nRunning task: {task['name']}")
    agent = Agent(
        model="unsloth/GLM-4.6V-Flash-GGUF:Q8_0",
        system_prompt="You are a helpful assistant. Use the correct tools to answer the user's question.",
        tools=task["tools"],
    )
    buffer = []
    async for chunk, _, _ in agent.chat(
        [{"role": "user", "content": task["prompt"]}],
        reasoning=task["reasoning"],
    ):
        if chunk:
            buffer.append(chunk)
    response = "".join(buffer).strip()
    if response:
        print(f"\n[{task['name']}] {response}")


def cron_job(task: dict):
    asyncio.run(run_job(task))


def build_cron_trigger(schedule: str) -> CronTrigger:
    fields = schedule.split()
    if len(fields) == 5:
        return CronTrigger.from_crontab(schedule)
    if len(fields) == 6:
        second, minute, hour, day, month, day_of_week = fields
        return CronTrigger(
            second=second,
            minute=minute,
            hour=hour,
            day=day,
            month=month,
            day_of_week=day_of_week,
        )
    raise ValueError(f"Unsupported cron format: {schedule!r}")


def main():
    scheduler = BlockingScheduler()
    for task in background_tasks:
        scheduler.add_job(
            cron_job,
            trigger=build_cron_trigger(task["schedule"]),
            args=[task],
        )
    scheduler.start()


main()
