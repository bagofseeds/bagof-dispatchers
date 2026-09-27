---
icon: fontawesome/solid/database
---

# Your own registry

`@dispatch` is a ready-made registry that keeps functions **separate per
module** — the same name in two modules is two independent functions. Build a
`Dispatcher` when you want a **shared** function several modules extend: a
`Dispatcher` keys a function by its qualified name alone, so every module
registering that name into the one instance extends a single function.

Reach a function to hand around through the `functions` namespace. It
**get-or-creates**, so a registry and its callers can name the same function in
any order:

```pycon
>>> from bagof.dispatchers import Dispatcher
>>> registry = Dispatcher()
>>> render = registry.functions.render      # an empty function, for now
>>> @registry
... def render(x: int) -> str:
...     return "int:%d" % x
>>> registry.functions["render"] is render is registry.functions.render
True
>>> render(7)
'int:7'
```

`functions` is a bare namespace with **no methods of its own**, so a function
may be named like anything — `register`, `items`, `map` — without colliding:

```pycon
>>> registry = Dispatcher()
>>> items = registry.functions["items"]
>>> @items.register
... def _(x: int) -> str:
...     return "one item"
>>> registry.functions.items(0)
'one item'
>>> "items" in registry.functions
True
```

Attribute access ignores names beginning with `_`, so a REPL or a tool probing
for dunders never mints an empty function; item access does not, so
`registry.functions["_x"]` still names one. For the same reason, do not register
an anonymous overload (`@dispatch def _`, or a `lambda`) directly on the
module-level `dispatch`: they all key the one `"_"` or `"<lambda>"` name and
collapse into a single function. Give each overload a real name, or overlay
hints onto a named `def`.
