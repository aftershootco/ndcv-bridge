# Tooling

Usage for the tools referenced in SKILL.md. Both are extra cargo subcommands:

```sh
cargo install cargo-mutants cargo-hack
```

## cargo-mutants — verifying tests actually catch bugs

Mutates your source, then reruns the tests. A mutant that survives means no test
covers that behaviour. Run it after adding tests, and read the result before
moving on.

Never run it unscoped over a whole crate — it is O(mutants x test-suite) and will
take tens of minutes. Always narrow it:

```sh
# only the code changed on this branch (preferred)
git diff origin/main.. > /tmp/pr.diff
cargo mutants --in-diff /tmp/pr.diff

# only one file
cargo mutants -f src/parser.rs

# a subset, minus generated or vendored code
cargo mutants -f 'src/domain/**' -e 'src/**/generated.rs'
```

| Flag | Meaning |
| --- | --- |
| `--in-diff FILE` | Only mutate lines present in a unified diff file. |
| `-f GLOB`, `--file GLOB` | Only mutate matching files. Repeatable. |
| `-e GLOB`, `--exclude GLOB` | Skip matching files. Repeatable. |

Reading the output: `caught` is good, `missed` means a surviving mutant, so add
a test. `unviable` (did not compile) and `timeout` are usually noise.

## cargo-hack — verifying every feature combination builds

Features must be additive: turning one on adds behaviour, never removes or
changes it. cargo-hack checks that by running a command once per feature
combination.

```sh
# every combination (the thorough one)
cargo hack check --feature-powerset --no-dev-deps

# one feature at a time — use when the powerset is too slow
cargo hack check --each-feature --no-dev-deps

# cap the powerset when there are many features
cargo hack check --feature-powerset --depth 2 --no-dev-deps

# run the tests, not just check, across the workspace
cargo hack test --each-feature --workspace
```

`--no-dev-deps` avoids [cargo#4866] and is recommended whenever the goal is
checking features rather than running tests.

### Genuinely conflicting features

Prefer telling cargo-hack about the conflict so it skips those combinations:

```sh
cargo hack check --feature-powerset --mutually-exclusive-features tls-rustls,tls-native
```

If the conflict must also be a hard error for downstream users, guard it at the
top of `lib.rs` / `main.rs`. The `cfg` attribute is required — an unguarded
`compile_error!` breaks every build:

```rust
#[cfg(all(feature = "tls-rustls", feature = "tls-native"))]
compile_error!("features `tls-rustls` and `tls-native` are mutually exclusive");
```

[cargo#4866]: https://github.com/rust-lang/cargo/issues/4866
