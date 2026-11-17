# Prompts

The nine LLM system prompts of CGScout (P1-P9, in the order they run), exported as plain text so they can be read, cited and diffed without opening the code.

**The code is the source.** Each prompt is a string constant in a module (the table in [INDEX.md](INDEX.md) gives the symbol, the length and the SHA-256 of each one, and the version
constant where there is one). These files are generated copies:

```bash
python prompts/export_prompts.py --write     # regenerate P*.txt, SHA256SUMS and INDEX.md from the code
python prompts/export_prompts.py --check     # exit 1 if a copy differs from its constant
```

`tests/test_prompts_export.py` runs the check, so a prompt cannot be edited in the code without regenerating these files. `SHA256SUMS` is in the `sha256sum -c` format.

Cache keys: the cache entries of P1, P2, P3, P5 and P8 contain the prompt text (an edit invalidates exactly those entries); the entries of P4, P6, P7 and P9 contain a version constant
instead, so an edit of those four needs a version bump (`INDEX.md` lists them).
