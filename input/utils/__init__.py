import collections.abc as col_abc
import typing as T

__all__ = [
	"sign",
	"offsets",
	"ints"
]

def sign(a) -> T.Literal[-1, 0, 1]:
	'''
	Returns *signum* function for ``v`` ( *sgn(v)* )
	'''

	return (a > 0) - (a < 0)


offsets_t = T.TypedDict("offsets_t", {
	"x_": slice,
	"_x": slice,
})
'''
Offset types - used when setting part of array with another array:

"x_" <- where to set to,

"_x" <- what to set to.
'''


def offsets(v: int) -> offsets_t:
	'''
	Based on signum of ``v``, returns slice indexes used to set part of array \
		as part of another
	'''

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
	'''
	Returns lazy iterator over subsequent ``int`` values (possibly infinite).
	'''
	i = 0
	while True:
		yield i
		i += 1
