---
icon: fontawesome/solid/sitemap
---

# Part of `bagof`

`bagof.dispatchers` is the low-level root of the `bagof` family: it owns the hint
subtype relation and introspection helpers (under `bagof.dispatchers.core`) that
the other bags build on. Its only dependency is
[`typing_extensions`](https://typing-extensions.readthedocs.io/), and it supports
Python 3.8 through current.

The design is written up in
[RFC 0001](../rfc/0001-hint-informed-multiple-dispatch.md).
`bagof` is a namespace package of small, focused tools; see the
[project overview](https://bagofseeds.github.io/bagof/).
