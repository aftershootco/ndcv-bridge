---
name: rust
description: Rust coding conventions - edition 2024, functional style, struct-based APIs, crate preferences (thiserror/error-stack, tokio, tracing, axum, reqwest, nom), testing with cargo-mutants, and ast-grep lint rules enforcing them. Use when writing, reviewing, linting, or scaffolding any Rust code, crate, or Cargo workspace.
---

# Rust programming guidelines

1. Use rust edition 2024
2. Prefer a functional style over an imperative one:
   - iterators / streams instead of hand-written loops
   - method chaining instead of sequential calls Stop when it hurts readability,
     error propagation, or hot-path performance. A plain `for` loop with `?`
     beats a contorted chain.
3. Prefer associated methods on a struct over free floating functions whenever
   there is a natural receiver type. Free functions are fine when there is
   none - don't invent a namespacing struct just to hold them.
4. Before making a new crate always define the external api / interface for the
   crate. The dev-ex for the crate matters more than anything.
5. Make sure to add examples to crates that are to be externally consumed. Add
   them under examples/foo.rs as well as part of the main doc comments.
6. If doing any unsafe code properly explain why this is required and what are
   the drawbacks / alternatives.
7. No global mutable state - it makes code untestable and dependent on invisible
   conditions. Inject dependencies instead. `const` and `static` holding
   immutable data are fine. Never `static mut`, no lazily initialised global
   (`OnceLock`, `LazyLock`, `once_cell`), and no interior-mutable global
   (`Mutex`, `RwLock`, `AtomicX`) used as an ambient singleton.
8. Testing. Test behaviour, not implementation. Cover critical and user-facing
   paths only - hundreds of low-value tests just slow down CI and local
   iteration, which only hurts us in the end. Verify the tests you add with
   `cargo-mutants` and read the result immediately: a surviving mutant in
   critical code means write a test, one in incidental code means leave it.
   Never run it over the whole crate - see `references/tools.md` for scoped
   invocations.
9. Use abstractions. Like for multiplatform crates unify all the system
   dependent items into a trait and implement them per-system
10. Keep features additive - enabling one adds behaviour, never removes or
    changes it. Check every combination with `cargo-hack`. If two features
    genuinely conflict, declare it to cargo-hack and guard it with a `cfg`-gated
    `compile_error!` in lib.rs / main.rs. See `references/tools.md`.
11. Async:
    - No blocking code in async functions. Use `spawn_blocking` if you need to
      do any sync call in async.
    - Use native `async fn` in traits. Don't reach for the `async-trait` macro
      it boxes and heap-allocates every call. It is only justified when you
      genuinely need `dyn` dispatch, since native async fns are not
      dyn-compatible. Where callers need the future to be `Send`, write
      `-> impl Future<Output = T> + Send` rather than going back to the macro.
12. Self-documenting types: take `user_id: UserId` not `user_id: u64`, where
    `struct UserId(u64);`. The newtype turns "passed the wrong id" into a
    compile error instead of a runtime bug. Same for `bool` parameters - use a
    two-variant enum.
13. Crate preferences (error handling, async, logging, config, http, parsing):
    see `references/crates.md`. Read it before adding any dependency.
14. The mechanically checkable half of these guidelines ships as ast-grep rules
    in `rules/`, driven by `sgconfig.yml`. Run them on any Rust code you write
    or review, always through the symlink-resolved path, since a symlinked skill
    dir loads zero rules and passes silently:
    `ast-grep scan -c "$(realpath <this skill dir>/sgconfig.yml)" src/` - see
    `references/lints.md` for the rule list and the single-rule invocation.
15. Don't panic in code that ships. No `unwrap`, `expect`, `panic!`, `todo!`
    or `unimplemented!` outside tests - each one aborts a caller that could have
    handled the case. Return a `Result` and keep the error typed: a `thiserror`
    variant in a library, an `error-stack` context in a binary. Never flatten
    one to a `String` (`.map_err(|e| e.to_string())`) - that throws away the
    source chain and leaves callers matching on prose. `unreachable!` is fine
    where it asserts an invariant the types cannot express; name the invariant
    in the message.
16. Convert with `From` / `TryFrom`, not `as`. An `as` cast between integers
    truncates, wraps and flips sign in silence, and float-to-int saturates.
    `From` for the widening direction, `TryFrom` for the narrowing one, so the
    loss is a compile error or a `Result`. Pointer casts and `as _` in FFI have
    no trait equivalent - those stay, with a comment saying why.

If invoked without any arguments you are to review the codebase to match these
guidelines.
