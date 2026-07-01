from pydantic import BaseModel


class ThreadPoolStatusSchema(BaseModel):
    max_workers: int
    active_workers: int
    available_workers: int
    queued_tasks: int
