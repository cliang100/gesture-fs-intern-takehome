"""
Document Q&A Pipeline — YOUR WORK GOES HERE.

The knowledge base (loading, chunking, vector store) is already built
for you in knowledge_base.py. Your job is to:

  1. Retrieve relevant chunks and generate an answer
  2. Wire it up into an interactive CLI

Useful docs:
  - Vector store search: https://python.langchain.com/docs/how_to/vectorstores/
  - HuggingFace pipelines: https://python.langchain.com/docs/integrations/llms/huggingface_pipelines/
"""

import os
import argparse
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.knowledge_base import build_knowledge_base


# ──────────────────────────────────────────────
# Provided: local LLM (no API key needed)
# ──────────────────────────────────────────────
def get_llm():
    """Return a callable local LLM using flan-t5-base.

    Downloads ~1GB on first run, then cached.
    Usage:
        llm = get_llm()
        result = llm("What color is the sky?")
        print(result[0]["generated_text"])  # "blue"
    """
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    def generate(prompt):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_new_tokens=150)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return [{"generated_text": text}]

    return generate


# ──────────────────────────────────────────────
# Provided: prompt template
# ──────────────────────────────────────────────
PROMPT_TEMPLATE = """You are a helpful assistant for a marketing agency. Use the following context to answer the client's question.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Client question: {question}

Answer:"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TODO 1: Implement ask_question
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def ask_question(vector_store, llm, question: str) -> dict:
    """Retrieve relevant chunks and generate an answer.

    Steps:
      1. Use vector_store.similarity_search(question, k=3) to get
         the top 3 most relevant document chunks.
      2. Combine the chunk text into a single context string.
         (Hint: each chunk has a .page_content attribute)
      3. Format the PROMPT_TEMPLATE with the context and question.
      4. Pass the formatted prompt to llm(...) and extract the
         generated text from the result.

    Args:
        vector_store: FAISS vector store from knowledge_base.py
        llm: Callable from get_llm()
        question: The user's question string

    Returns:
        dict with two keys:
            "answer"  -> str: the generated answer
            "sources" -> list[str]: the chunk texts that were retrieved
    """
    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty.")

    docs = vector_store.similarity_search(question, k=3)
    sources = [doc.page_content for doc in docs]
    context = "\n\n".join(sources)

    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    answer = llm(prompt)[0]["generated_text"].strip()

    return {"answer": answer, "sources": sources}


def print_result(result: Dict[str, object]) -> None:
    """Print a Q&A result: numbered sources, then the answer."""
    print("\n📄 Sources:")
    for i, source in enumerate(result["sources"], start=1):
        preview = source if len(source) <= 150 else source[:150] + "..."
        print(f"  {i}. {preview}")
    print(f"\n💬 Answer: {result['answer']}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TODO 2: Complete the interactive loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    parser = argparse.ArgumentParser(description="Ask the marketing agency's Q&A bot a question.")
    parser.add_argument("--query", help="Ask a single question and exit, instead of looping.")
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    if not os.path.isdir(data_dir):
        print(f"Error: data directory not found at {data_dir}")
        return

    print("Building knowledge base...")
    vector_store = build_knowledge_base(data_dir)
    print("Loading LLM...")
    llm = get_llm()

    if args.query is not None:
        try:
            print_result(ask_question(vector_store, llm, args.query))
        except ValueError as e:
            print(f"Error: {e}")
        return

    print("\nReady! Ask a question (or type 'quit' to exit).")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if question.lower() == "quit":
            break
        try:
            print_result(ask_question(vector_store, llm, question))
        except ValueError as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()