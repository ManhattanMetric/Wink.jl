# Safety model

Wink hands a language model real capabilities inside your session, so the
boundaries are explicit:

## The confirmation gate

`eval_code` and `edit_file` are gated: the exact code (or a diff-style edit
preview) is printed and Wink waits for `y`/`N` on stdin. Anything except an
explicit yes — including `Ctrl-C` — is a denial. Denials are reported to the
model as `DECLINED` with instructions not to retry verbatim.

- `Wink.autoeval!(true)` disables the gate for the session (`:autoeval on` in
  `ai>` mode). Non-interactive sessions auto-deny unless autoeval is on.
- The gate itself is swappable: `Wink.CONFIG.confirm` is any
  `(kind::Symbol, preview::String) -> Bool`.

## Perimeters

- **Reading** (`read_file`) is limited to the active project, directories of
  loaded packages, Julia's own source tree, and the package depot.
- **Editing** (`edit_file`) is limited to the active project and packages
  loaded for development. Stdlib and installed depot copies are read-only.
  Only existing `.jl` files can be edited; files are never created.
- Every edit first writes a backup copy to a scratch directory
  (`Wink.backups()` lists them) — though git remains the real safety net.

## Type-expression evaluation

Tools accept type strings like `"Vector{<:Real}"`, which must be evaluated to
resolve dispatch. Wink walks the parsed expression first and only permits type
syntax: names, dotted paths, curly braces, `where`, `<:`, tuples, literals,
and `typeof(name)`. Arbitrary calls in type position (e.g.
``typeof(run(`...`))``) are rejected before any evaluation.

## What this is not

Wink is not a sandbox. An *approved* `eval_code` runs with your full
privileges, exactly as if you had typed it — that is the point of the tool.
The gate exists so nothing runs without you seeing it first; review what you
approve, especially with `autoeval` on.
