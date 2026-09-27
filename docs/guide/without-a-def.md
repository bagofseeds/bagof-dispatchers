---
icon: fontawesome/solid/table
---

# Without a `def`

For a table of type-to-callable with no functions to write, build one from a
mapping. A tuple key gives one positional hint per element:

```pycon
>>> from bagof.dispatchers import Function
>>> size = Function.from_mapping({(int,): abs, (str,): len}, name="size")
>>> size(-3)
3
>>> size("abcd")
4
```

For an exotic shape — positional-only, `*args`, keyword-only — that a plain
tuple of hints cannot spell, pass a `Signature` as the key:

```pycon
>>> from bagof.dispatchers import Function, Signature
>>> total = Function.from_mapping({Signature.from_hints(int, int): lambda a, b: a + b})
>>> total(2, 3)
5
```
