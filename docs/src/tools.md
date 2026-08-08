# Tools

These are the tools Wink exposes to the model. Read-only tools run without
asking; **gated** tools show you the exact code or edit and wait for y/n
confirmation (unless `Wink.autoeval!(true)`).

## Read-only introspection

| Tool | What it does |
|---|---|
| `list_methods(signature)` | Method table with signatures and Revise-aware source locations. |
| `methods_with(type_name, module_name, supertypes)` | The reverse lookup: methods accepting an argument of a given type ("what can I call with an X?"). Module-scoped queries include unexported functions. |
| `get_source(signature)` | The actual source text of a method via CodeTracking; falls back to clearly-marked lowered IR when no source file exists (e.g. `eval`'d code). |
| `get_ir(signature, level)` | Compiler IR at `lowered`, `typed`, `warntype`, `llvm`, or `native`. |
| `get_doc(name)` | Docstring for a function, type, macro, constant, or module. |
| `type_info(type_name)` | Kind, mutability, parameters, fields, supertype chain, subtypes. |
| `module_info(module_name, include_private)` | Module contents grouped by kind. |
| `list_variables(module_name, pattern)` | Bindings a module currently holds with sizes and value summaries (`varinfo`); `""` shows your `Main` workspace. |
| `where_defined(signature)` | `file:line` of a method's current definition. |
| `read_file(path, first_line, last_line)` | Line-numbered file reading, restricted to the active project, loaded packages, and Julia's source tree. |
| `list_names(pattern)` | Keyword search over loaded docstrings (`apropos`). |
| `search_docs(query, top_k)` | Semantic docstring search over everything loaded (embedding index; keyword fallback without an embedding provider). |
| `search_packages(pattern)` | Offline name search over the on-disk package registries (General, ~14k packages): latest version, repo URL, and whether each hit is already in your project or loaded. |

Signature strings accept bare names (`"sort"`), qualified names
(`"Base.Docs.doc"`), and typed forms (`"sort(::Vector{Int})"`,
`"combine(Int, b::Int)"`). Type expressions are evaluated only after a
whitelist check — see [Safety model](safety.md).

## Gated actions

| Tool | What it does |
|---|---|
| `eval_code(code)` | Evaluates code in your live `Main` with REPL soft-scope semantics; returns the value plus captured stdout/stderr. |
| `edit_file(path, old_string, new_string)` | Replaces one exact, unique occurrence in a source file; Revise applies the change to the running session. A backup is saved first. |

A declined gate returns a `DECLINED` message to the model, which is instructed
not to retry the identical action.
