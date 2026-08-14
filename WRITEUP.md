# Writeup

## Approach

I started by reading through `knowledge_base.py` to understand what was already
built (chunking, embeddings, FAISS store) so I knew exactly what `ask_question()`
needed to plug into. From there I implemented the two TODOs in order:

1. `ask_question()`: retrieve top-3 chunks, join them into a context string,
   format the provided prompt template, call the LLM, return `{answer, sources}`.
2. `main()`: build the knowledge base, load the LLM, then loop on user input
   until `quit`.

I verified each piece with `pytest tests/ -v` as I went, then manually ran the
CLI (`python -m src.pipeline`) with a mix of expected questions, edge cases,
and off-topic questions to see how it actually behaved outside the test suite.

## Bonus items

- **`--query` flag** for single-question, non-interactive mode.
- **Input validation**: `ask_question()` raises `ValueError` on empty/whitespace
  input, caught and printed cleanly in `main()` instead of crashing.
- **Error handling**: missing `data/` directory is checked before doing any
  work; `Ctrl+C`/`Ctrl+D` in the interactive loop exit cleanly instead of
  throwing a traceback.
- **Type hints**: added where they clarify intent (e.g. `print_result`).
- **Extra tests**: added coverage for input validation and for the source-text
  truncation logic in `print_result()`, since neither had any test coverage
  in the original suite.

I extracted a `print_result()` helper since the same source/answer formatting
was needed in both `--query` mode and the interactive loop, so there's one place to
change the output format instead of two copies drifting apart.

## Bugs I caught while testing manually

Manual testing surfaced two things the automated tests couldn't catch, since
none of the tests call `main()` at all (they exercise `ask_question()` and
`get_llm()` directly against a fixture-built vector store, so anything that
only lives inside `main()`'s control flow has zero test coverage):

- `--query ""` was silently falling through into interactive mode instead of
  raising the validation error, because `if args.query:` treats an empty
  string as falsy. Fixed by checking `args.query is not None` instead, so an
  explicitly passed empty string is still routed through `ask_question()` and
  hits the `ValueError` path.
- The missing `data/` directory check (renamed `data/` temporarily to confirm)
  worked correctly and printed a clean error instead of a traceback, but this
  is another case only manual testing could confirm, since `main()` isn't
  under test.

## Retrieval observations

The `data/` folder includes files beyond the marketing-agency docs described
in the README (`company_handbook.txt`, `product_faq.txt`, internal-looking
content like PTO policy and on-call rotations). Asking about contract
cancellation retrieved the correct pricing/FAQ chunk as the top result, but
the other two slots were pulled from these unrelated internal docs. The
answer was still correct because the top-ranked chunk had the real content,
but it's a sign that `k=3` over a small, topically mixed corpus doesn't
guarantee all three retrieved chunks are actually relevant, just that the
best match usually is. I left this as-is rather than filtering by file or
tuning `k`, since the assignment says not to modify `knowledge_base.py` and
I didn't want to risk breaking the multi-source retrieval tests by changing
retrieval behavior.

I also noticed inconsistent grounding in the LLM's answers. Asking "What's
the capital of China?" (completely unrelated to the retrieved marketing
chunks) got a confident, correct answer ("Beijing") despite the prompt
explicitly instructing the model to say "I don't have enough information"
when the answer isn't in the context. Asking a different off-topic question
("tallest mountain in the world") correctly triggered the "I don't have
enough information" fallback. This tells me `flan-t5-base` doesn't reliably
stay grounded in the provided context; general pretraining knowledge can
leak through and override the instruction when the model happens to already
know the answer. That's a real limitation of the model, not something fixable
in the retrieval or prompt-formatting code.