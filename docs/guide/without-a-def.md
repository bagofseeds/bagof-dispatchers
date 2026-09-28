# Without a `def`

When the overloads already exist as a plain mapping from type to callable,
[`Function.from_mapping`][bagof.dispatchers.Function.from_mapping] builds a
dispatched function from that mapping directly, with no `def` needed for any
of the overloads. A tuple key gives one positional hint per element:

```pycon
>>> from bagof.dispatchers import Function
>>> size = Function.from_mapping({(int,): abs, (str,): len}, name="size")
>>> size(-3)
3
>>> size("abcd")
4
```

A plain tuple of hints can only describe ordinary positional parameters. For
a signature shape it cannot spell, such as positional-only parameters,
`*args`, or keyword-only parameters, pass a
[`Signature`][bagof.dispatchers.Signature] as the key instead:

```pycon
>>> from bagof.dispatchers import Function, Signature
>>> total = Function.from_mapping({Signature.from_hints(int, int): lambda a, b: a + b})
>>> total(2, 3)
5
```
