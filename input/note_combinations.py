import random as r
import functools as fn
import operator as op
import typing as _t

note_deno = _t.Literal[10,20,50,100,200,500]
notes: list[note_deno] = [
	10, 20, 50,
	100, 200, 500
]

note_combinations: list[list[note_deno]] = []

for _ in range(42):
	one_combination: list[note_deno] = notes.copy()
	for _ in range(r.randint(0, 3)):
		one_combination[r.randint(0, len(notes) - 1)] = notes[r.randint(0, len(notes) - 1)]
	note_combinations.append(one_combination)

note_count: dict[note_deno, int] = { k: 0 for k in notes }

for comb in note_combinations:
	for note in comb:
		note_count[note] += 1

print(note_count)

expected_note_count: dict[note_deno, int] = { k: 50 for k in notes }

needed_note_count: dict[note_deno, int] = { k: expected_note_count[k] - note_count[k] 
	for k in notes 
}
print(needed_note_count)

remaining_notes: list[note_deno] = fn.reduce(
	op.concat, # type: ignore [arg-type] # plz for love of god fix your type inference mypy
	[[k] * v 
		for k, v in needed_note_count.items()
	],
	[]
)

r.shuffle(remaining_notes)

for rem in remaining_notes:
	while True:
		rand_i: int = r.randint(0, len(note_combinations) - 1)
		if len(note_combinations[rand_i]) < 8:
			note_combinations[rand_i].append(rem) # type: ignore [arg-type] # seems like a bug of type-checker?
			break

note_count = { k: 0 for k in notes }

for comb in note_combinations:
	for note in comb:
		note_count[note] += 1

print(note_count)

with open("note_combinations.txt", "w") as f:
	for comb in note_combinations:
		f.write(f"{len(comb)}:: {comb}\n")
