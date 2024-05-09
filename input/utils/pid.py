import typing as T

__all__ = [
	"pid"
]

class pid:
	'''
	A simple PID controller implementation, with arbitrary tick
	'''

	def __init__( self, *, set_point: float, p: float, i: float, d: float) -> None:
		
		self.p: float = p
		'''
		Proportional term
		'''

		self.i: float = i
		'''
		Integral term
		'''

		self.d: float = d
		'''
		Derivative term
		'''

		self.set_point: float = set_point
		'''
		Desired target value
		'''

		self.outs: list[float] = [ 0. ]
		'''
		List of all output values
		'''

		self.errs: list[float] = [ 0. ]
		'''
		List of all error values
		'''

		self.err_acc: float = 0.
		'''
		Accumulated error - integral of all errors
		'''

		self.cur_i: int = -1
		'''
		Current tick that PID is calculating
		'''

	@T.overload
	def update(self, val: float, *, get_err: T.Literal[True]) -> T.Tuple[float, float]:
		'''
		Calculates new correction output.

		:param val: Input value that controller should correct
		:param get_err: Whether return error as well
		:returns: new correction output and current error of input value
		'''
		...

	@T.overload
	def update(self, val: float, *, get_err: T.Literal[False] = False) -> float:
		'''
		Calculates new correction output.

		:param val: Input value that controller should correct
		:param get_err: Whether return error as well
		:returns: new correction output
		'''
		...

	def update(self, val: float, *, get_err: bool = False) -> float | T.Tuple[float, float]:
		'''
		Calculates new correction output.

		:param val: Input value that controller should correct
		:param get_err: Whether return error as well
		:returns: new correction output (and current error of input value, if ``get_err == True``)
		'''
		
		err = self.set_point - val
		self.errs.append(err)
		self.err_acc += err

		out = (
			self.errs[-1] * self.p
			+ self.err_acc * self.i
			+ (self.errs[-1] - self.errs[-2]) * self.d
		)
		self.outs.append(out)

		self.cur_i += 1

		if get_err:
			return out, err
		else:
			return out

	def change_param( 
		self, *, set_point: float | None = None,
		p: float | None = None, i: float | None = None, d: float | None = None
	) -> None:
		'''
		Changes controllers parameters. Arguments that are not passed (or None) \
			will not alter current values.

		:param set_point: New desired target value
		:param p: New proportional term
		:param i: New integral term
		:param d: New derivative term
		'''

		if set_point is not None:
			self.set_point = set_point
		if p is not None:
			self.p = p
		if i is not None:
			self.i = i
		if d is not None:
			self.d = d
