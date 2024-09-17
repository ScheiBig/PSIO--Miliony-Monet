'''
``ansi`` module provides escape sequences for text formatting and cursor manipulation in console.

ANSI escape sequences are supported in almost every *nix terminal,
in PowerShell 5 (might need a flag to be activated) and Windows Terminal app (natively).

Within this module, you will find objects for:
* ``ansi.fg`` - changing font color,
* ``ansi.bg`` - changing background color (directly behind font),
* ``ansi.fmt`` - changing font decoration style,
* ``ansi.clear`` - clearing console,
* ``ansi.cur`` - moving cursor,
* ``ansi.progress`` - progress bar printing utility.

Please note, that **escape sequences modify console state** and so **format will probably outlive 
application** that uses it, therefore application using this module **should 
``print(ansi.fmt.reset)`` in cleanup code** to restart format of console.
'''

import typing as _t

__all__ = [
	"fg",
	"bg",
	"fmt",
	"clear",
	"cur",
	"progress",
]

class fg:
	'''Foreground colors'''
	BLACK   = "\u001b[30m"
	RED     = "\u001b[31m"
	GREEN   = "\u001b[32m"
	YELLOW  = "\u001b[33m"
	BLUE    = "\u001b[34m"
	MAGENTA = "\u001b[35m"
	CYAN    = "\u001b[36m"
	WHITE   = "\u001b[37m"

	class b:
		'''Bright foreground colors'''
		BLACK   = "\u001b[90m"
		RED     = "\u001b[91m"
		GREEN   = "\u001b[92m"
		YELLOW  = "\u001b[93m"
		BLUE    = "\u001b[94m"
		MAGENTA = "\u001b[95m"
		CYAN    = "\u001b[96m"
		WHITE   = "\u001b[97m"

	@classmethod
	def rgb(cls, r: int, g: int, b: int): return f"\u001b[38;2;{r};{g};{b}m"


class bg:
	'''Background colors'''
	BLACK   = "\u001b[40m"
	RED     = "\u001b[41m"
	GREEN   = "\u001b[42m"
	YELLOW  = "\u001b[43m"
	BLUE    = "\u001b[44m"
	MAGENTA = "\u001b[45m"
	CYAN    = "\u001b[46m"
	WHITE   = "\u001b[47m"

	class b:
		'''Bright background colors'''
		BLACK   = "\u001b[100m"
		RED     = "\u001b[101m"
		GREEN   = "\u001b[102m"
		YELLOW  = "\u001b[103m"
		BLUE    = "\u001b[104m"
		MAGENTA = "\u001b[105m"
		CYAN    = "\u001b[106m"
		WHITE   = "\u001b[107m"

	@classmethod
	def rgb(cls, r: int, g: int, b: int): return f"\u001b[48;2;{r};{g};{b}m"


class fmt:
	'''Text formatting'''
	RESET     = "\u001b[0m"
	BOLD      = "\u001b[1m"
	UNDERLINE = "\u001b[4m"
	REVERSE   = "\u001b[7m"

class clear:
	'''Utilities for console clearing'''

	ALL    = "\u001b[2J"
	SCROLL = "\u001b[3J"
	LN     = "\u001b[2K"

class cur:
	'''Utilities for cursor position'''
	UP    = "\u001b[1A"
	DOWN  = "\u001b[1B"
	RIGHT = "\u001b[1C"
	LEFT  = "\u001b[1D"

	NEXT_LN = "\u001b[1E" 
	PREV_LN = "\u001b[1F" 

	BEGIN     = "\u001b[0G"
	TOP_BEGIN = "\u001b[0;0H"

	@classmethod
	def pos(cls, col: int = 1, row: _t.Optional[int] = None):
		if row == None:
			return f"\u001b[{col}G"
		else:
			return f"\u001b[{row};{col}H"
		
	def next(cls, lines: int = 1):
		return f"\u001b[{lines}E"
	
	def prev(cls, lines: int = 1):
		return f"\u001b[{lines}F"

def progress(
	progress: int,
	size: int,
	*,
	frame: str | None = "[]",
	bar: str = f"{fg.BLUE}#{fmt.RESET}",
	returns: bool = False,
	label: str | None = None,
	final: str | None = None
) -> None:
	
	'''
	Prints progress bar in console
	:param progress: how many bars should be filled - 0 <= progress <= size
	:param size: how many bars are to be filled
	:param frame: two-character string - frame around progress bar (if None, will be frameless)
	:param returns: whether progress bar should return to previous line \
		(print in place - shouldn't use with multiple bars)
	:param label: - label after progress bar - should always be same-width (if None, will print %)\
	:param final: - label that will replace previous label (still same width, None will not replace)
	'''

	after = label if label is not None else f"{progress * 100 / size:6.2f}%"
	if progress == size and final is not None:
		after = final

	print(
		cur.PREV_LN if returns else "",
		frame[0] if frame is not None else "",
		bar * progress,
		" " * (size - progress),
		frame[1] if frame is not None else "",
		" ",
		after,
		sep=""
	)

