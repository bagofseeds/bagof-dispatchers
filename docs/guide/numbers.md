---
icon: fontawesome/solid/hashtag
---

# Dispatching on numbers

Dispatch matches the **runtime type** of a value, so a `float` overload accepts
only actual `float` values. An `int` is **not** a `float`: Python's numeric
tower (`int` → `float` → `complex`) is a static type-checking convention, and
dispatch does not apply it. Annotate a parameter `float` only when you mean an
actual float and nothing else:

```pycon
>>> from bagof.dispatchers import dispatch, NoMethodError
>>> @dispatch
... def halve(x: float) -> float:
...     return x / 2
>>> halve(2.0)
1.0
>>> try:
...     halve(3)          # an int is not a float
... except NoMethodError:
...     print("no overload for int")
no overload for int
```

To accept **any real number**, annotate with `numbers.Real` instead of `float`.
It matches `int`, `float`, `bool`, `fractions.Fraction`, and third-party reals
such as NumPy scalars — every type registered under the `numbers` abstract base
classes:

```pycon
>>> from bagof.dispatchers import dispatch
>>> from numbers import Real
>>> @dispatch
... def classify(x: Real) -> str:
...     return "a real number"
>>> classify(3)           # int
'a real number'
>>> classify(2.5)         # float
'a real number'
```

Use `numbers.Integral` for integers (`int`, `bool`, and NumPy ints),
`numbers.Complex` to also accept `complex`, or spell the set out explicitly with
`Union[int, float]`.
