
import queue
from typing import Any


class spoiling_queue(queue.Queue):
	'''
	Special type of queue, in which elements "spoil" - which means if you
	attempt to put element into full queue, the "oldest" element will become
	stale and will be removed to make space for new one.
	'''

	def put(
		self,
		item: Any,
		block: bool = True,
		timeout: float | None = None
	) -> None:
		if self.full():
			self.get_nowait()
		super().put(item, block, timeout)

	def avg(self) -> float:
		'''
		Computes average value of elements in this queue.

		If queue contains elements that are not real numbers, behavior
		is unspecified.
		'''
		if self.empty():
			return 0
		return sum(self.queue) / len(self.queue)
