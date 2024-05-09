import typing as T

class pid:

	def __init__( self, *, set_point: float, p: float, i: float, d: float) -> None:
		
		self.p: float = p
		self.i: float = i
		self.d: float = d
		self.set_point: float = set_point

		self.outs: list[float] = [ 0. ]
		self.errs: list[float] = [ 0. ]

		self.err_acc: float = 0.
		self.cur_i: int = 0

	@T.overload
	def update(self, val: float | None = None, *, get_err: T.Literal[True]) -> T.Tuple[float, float]:
		...

	@T.overload
	def update(self, val: float | None = None, *, get_err: T.Literal[False] = False) -> float:
		...

	def update(self, val: float, *,get_err: bool = False) -> float | T.Tuple[float, float]:
		
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
		if set_point is not None:
			self.set_point = set_point
		if p is not None:
			self.p = p
		if i is not None:
			self.i = i
		if d is not None:
			self.d = d
