import collections.abc as col_abc
import typing as T


def sign(a) -> T.Literal[-1, 0, 1]:
	return (a > 0) - (a < 0)


offsets_t = T.TypedDict("offsets_t", {
	"x_": slice,
	"_x": slice,
})


def offsets(v: int) -> offsets_t:
	match sign(v):
		case -1:
			return {
				"x_": slice(None, v),
				"_x": slice(-v, None),
			}
		case 0:
			return {
				"x_": slice(None, None),
				"_x": slice(None, None),
			}
		case 1:
			return {
				"x_": slice(v, None),
				"_x": slice(None, -v),
			}
		case _:
			raise NotImplementedError()


def ints() -> col_abc.Generator[int, None, None]:
	i = 0
	while True:
		yield i
		i += 1
