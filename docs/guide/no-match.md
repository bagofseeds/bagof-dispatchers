# When nothing matches

When no registered overload accepts a call, dispatch raises
[`NoMethodError`][bagof.dispatchers.NoMethodError]. Because `NoMethodError`
is also a `TypeError`, code that already catches `TypeError` keeps working
without modification. The error carries the function's name and the
overloads that came closest to matching:

```pycon
>>> from bagof.dispatchers import dispatch, NoMethodError, DispatchError
>>> @dispatch
... def area(width: int, height: int) -> int:
...     return width * height
>>> try:
...     area("triangle")
... except NoMethodError as error:
...     print(error.function, isinstance(error, DispatchError), isinstance(error, TypeError))
area True True
```

`NoMethodError` is also a [`DispatchError`][bagof.dispatchers.DispatchError],
the common base it shares with
[`AmbiguousMethodError`][bagof.dispatchers.AmbiguousMethodError], so code
that wants to catch either failure without distinguishing between them can
catch `DispatchError` instead.
