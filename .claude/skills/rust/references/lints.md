# Lint rules

The mechanically checkable half of the guidelines, as ast-grep rules. Paths
below are relative to the skill directory (the one holding `SKILL.md`).

`rules/` holds one rule per file and `utils/` the shared matchers they compose
(test code, `spawn_blocking` closures). Pass this skill's own config so the
rules work in any project, whether or not it has an `sgconfig.yml`, and resolve
the path through symlinks first:

```sh
# the skill dir with every symlink resolved - see "Resolve the path" below
SKILL=$(dirname "$(realpath <this skill dir>/sgconfig.yml)")

# every rule, against a crate
ast-grep scan -c "$SKILL/sgconfig.yml" src/

# a single rule
ast-grep scan -c "$SKILL/sgconfig.yml" --filter '^no-unwrap$' src/
```

## Resolve the path

ast-grep loads `ruleDirs`/`utilDirs` with a directory walker that does not
follow symlinks, and `--follow` only applies to the code being scanned, not to
rule discovery. So when the skill is installed as symlinks - Nix/home-manager,
stow, any dotfiles manager - a scan through the link path loads **zero rules**,
prints nothing and exits 0. That is indistinguishable from a clean run, so the
lint silently passes on every file. `realpath` resolves to the real files and
the rules load.

To confirm rules are loaded, `--filter` fails loudly when they are not:

```sh
ast-grep scan -c "$SKILL/sgconfig.yml" --filter '^no-unwrap$' src/
# Error: Rule not found: ^no-unwrap$   <- nothing loaded, fix the path
```

Use `--filter` rather than `-r rules/no-unwrap.yml` for a single rule: `-r`
bypasses `sgconfig.yml` and so never reads `utilDirs`, and every rule that
composes a util fails with ``Rule `is-test` is not defined``.

| Rule | Severity | Enforces |
| --- | --- | --- |
| `no-static-mut` | error | 7 - no global mutable state |
| `no-lazy-global` | error | 7 - no `LazyLock`/`OnceLock`/`once_cell` global |
| `no-ambient-singleton` | warning | 7 - no `Mutex`/`RwLock`/`AtomicX` singleton |
| `no-blocking-in-async` | error | 11 - no blocking call, `std` or rayon, in an async fn |
| `no-unwrap` | error | 15 - propagate the error instead |
| `no-expect` | error | 15 - propagate the error instead |
| `no-panic-macros` | error | 15 - `panic!`/`todo!`/`unimplemented!` - return a `Result` |
| `no-as-cast` | warning | 16 - `From`/`TryFrom` over a silent `as` cast |
| `no-stringly-errors` | warning | 15 - keep errors typed, not `String` |
| `no-println` | warning | 13 - `tracing` over `println!` |
| `no-dbg` | warning | Debugging leftover |

Most rules exempt test code - `#[test]` functions, `#[cfg(test)]` modules, and
the `tests/` and `benches/` directories - and `no-println` also exempts
`build.rs` and `examples/`, whose stdout is the point. `no-static-mut` and
`no-dbg` fire everywhere on purpose: a mutable global is the same hazard in a
test harness, and a `dbg!` is a leftover wherever it lands.

A hit is a starting point, not a verdict. If the exception is deliberate, say
why in a comment rather than reshaping the code to satisfy the rule.
