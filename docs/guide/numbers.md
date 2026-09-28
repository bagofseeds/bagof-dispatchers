# Dispatching on numbers

Dispatch matches the runtime type of a value, so an overload annotated
`float` accepts only actual `float` instances, and an `int` is not a
`float` under this rule. Python's numeric tower, described in
[PEP 484](https://peps.python.org/pep-0484/), treats `int` as a kind of
`float`, which is in turn a kind of `complex`, but that promotion is a
convention static type checkers follow. Dispatch does not apply it, so
annotate a parameter `float` only when the overload is meant for an actual
float value and nothing else:

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

To accept any real number, annotate the parameter with `numbers.Real`
instead of `float`. `numbers.Real` matches `int`, `float`, `bool`,
`fractions.Fraction`, and any third-party numeric type registered under the
`numbers` abstract base classes, such as NumPy scalars:

```pycon
>>> from numbers import Real
>>> @dispatch
... def classify(x: Real) -> str:
...     return "a real number"
>>> classify(3)           # int
'a real number'
>>> classify(2.5)         # float
'a real number'
```

Use `numbers.Integral` to accept integers, which include `int`, `bool`, and
NumPy's integer types. Use `numbers.Complex` to also accept `complex`, or
spell the accepted set out explicitly with a union such as
`Union[int, float]`.
