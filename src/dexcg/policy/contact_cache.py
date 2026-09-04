"""Slow-rate contact plan cache."""

from collections.abc import Callable

from dexcg.common.typing import ContactPlan


class ContactCache:
    def __init__(self, update_interval: int) -> None:
        if update_interval < 1:
            raise ValueError("update_interval must be at least one")
        self.update_interval = update_interval
        self._plan: ContactPlan | None = None
        self._languages: tuple[str, ...] | None = None

    def get(
        self,
        action_step: int,
        languages: list[str],
        planner: Callable[[], ContactPlan],
    ) -> ContactPlan:
        language_key = tuple(languages)
        refresh = (
            self._plan is None
            or self._languages != language_key
            or action_step % self.update_interval == 0
        )
        if refresh:
            self._plan = planner()
            self._languages = language_key
        return self._plan

    def reset(self) -> None:
        self._plan = None
        self._languages = None
