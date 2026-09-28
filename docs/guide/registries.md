# Your own registry

[`dispatch`][bagof.dispatchers.dispatch] is a ready-made registry that keeps
functions separate per module, so the same name defined in two different
modules produces two independent functions. A
[`Dispatcher`][bagof.dispatchers.Dispatcher] built directly behaves
differently: it keys a function by its qualified name alone, so every module
that registers that name into the same `Dispatcher` instance extends a
single, shared function. Build one when several modules need to contribute
overloads to what is conceptually one generic function.

A function held by a `Dispatcher` is reached through its `functions`
namespace, which get-or-creates: naming a function that does not exist yet
creates it empty, so a registry and the modules that register onto it can
name the same function in either order:

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

The `functions` namespace exposes no methods of its own, so a function may be
given any name, including one that would otherwise collide with a namespace
method, such as `register`, `items`, or `map`:

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

Attribute access ignores names beginning with an underscore, so a REPL or a
tool probing for dunder attributes never mints an empty function by
accident. Item access does not apply that rule, so `registry.functions["_x"]`
still names a function whose name starts with an underscore. This distinction
matters for anonymous overloads too: registering `@dispatch def _` or a
`lambda` directly keys the function by the literal name `"_"` or
`"<lambda>"`, so several such registrations on the same registry collapse
into a single function instead of staying independent. Give each overload a
real name, or overlay hints onto a named `def`, to avoid the collision.
