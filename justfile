ndcv-version := `cargo read-manifest --manifest-path ./ndcv-bridge/Cargo.toml | jq -r .version`
auth-token := `op item get "Cargo registry token" --reveal --fields password`


publish:
    cargo publish --registry kellnr -p bounding-box
    cargo publish --registry kellnr -p ndcv-bridge
    cargo publish --registry kellnr -p ndarray-image
    cargo publish --registry kellnr -p ndarray-resize
    cargo publish --registry kellnr -p ndarray-safetensors

ndcv-docs:
    # Package documentation for the upload
    cargo doc -p ndcv-bridge --target-dir ./target --no-deps
    cd ./target && zip -r doc.zip ./doc
    curl -sS -L -H "Authorization: {{auth-token}}" https://crates.darksailor.dev/api/v1/docs/ndcv-bridge/{{ndcv-version}} --upload-file target/doc.zip
