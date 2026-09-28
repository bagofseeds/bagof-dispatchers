---
icon: fontawesome/solid/sitemap
---

# Part of `bagof`

`bagof.dispatchers` is the low-level root of the `bagof` family of packages.
It owns the hint subtype relation and the introspection helpers, under
`bagof.dispatchers.core`, that the other bags build on. Its only dependency
is [`typing_extensions`](https://typing-extensions.readthedocs.io/), and it
supports Python 3.8 through the current release.

The design behind the dispatch engine is written up in
[RFC 0001](../rfc/0001-hint-informed-multiple-dispatch.md). `bagof` itself is
a namespace package of small, focused tools; see the
[project overview](https://bagofseeds.github.io/bagof/) for how the family
fits together.
