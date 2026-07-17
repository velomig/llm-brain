"""Interactive chat with long-term memory (terminal version).

The full loop, every turn:
    your prompt -> retrieve relevant memories -> inject as context ->
    LLM responds -> capture (prompt, response) back into the Neural Map.

Commands:
    /save <text>          store a one-off memory (category=note, importance=5)
    /summarize <category> compress a category into one profile memory
    /stats                how many memories are stored
    /recall <text>        show what WOULD be retrieved (no LLM call)
    /prune                forget old, low-value, never-recalled episodes
    /reflect              run the "sleep" pass: insights + proactive reminders
    /quit
"""
from __future__ import annotations

import config
from brain import log
from brain import reflect as reflect_mod
from brain import store
from brain.embeddings import EmbeddingUnavailable, embed_document, text_only_embedding
from brain.memory import MemoryUnit
from brain.pipeline import answer, capture, reinforce_recall
from brain.retriever import retrieve
from brain.summarize import summarize_category


def memory_count() -> int:
    return store.count_all()


def main() -> None:
    print(f"🧠 Neural Map  |  chat={config.CHAT_MODEL}  embed={config.embedding_model(config.CHAT_MODEL)}")
    print(f"   {memory_count()} memories stored.  Commands: /save /summarize /stats /recall /prune /reflect /quit\n")

    history: list[dict] = []
    while True:
        try:
            user = input("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue

        if user in ("/quit", "/exit"):
            break
        if user == "/stats":
            print(f"   {memory_count()} memories stored.\n")
            continue
        if user == "/prune":
            result = store.prune_episodes()
            print(f"   pruned {result['pruned']} faded episodes.  {memory_count()} memories remain.\n")
            continue
        if user == "/reflect":
            r = reflect_mod.reflect()
            print(f"   reflected: {r['insights']} insight(s), {r['reminders']} reminder(s).\n")
            continue
        if user.startswith("/save "):
            text = user[len("/save "):].strip()
            unit = MemoryUnit(text=text, category="note", importance=5, extra={"source_type": "user_entered"})
            try:
                embedding = embed_document(unit.text)
            except EmbeddingUnavailable:
                store.add(unit, text_only_embedding(), text_only=True)
            else:
                store.add(unit, embedding)
            print(f"   saved memory {unit.id[:8]}.\n")
            continue
        if user.startswith("/summarize "):
            cat = user[len("/summarize "):].strip()
            summary = summarize_category(cat)
            print(f"   {summary}\n" if summary else f"   no summary produced for '{cat}' (empty category or empty model output; nothing was changed).\n")
            continue
        if user.startswith("/recall "):
            for m in retrieve(user[len("/recall "):].strip()):
                print(f"   [{m['score']:.3f}] sim={m['similarity']} kw={m['keyword']} rec={m['recency']} :: {m['text'][:70]}")
            print()
            continue

        mems = retrieve(user)
        reinforce_recall(mems)
        result = answer(user, mems, history=history[-config.HISTORY_TURNS * 2:])
        reply = result["reply"]
        print(f"\nbrain › {reply}\n")
        if result.get("tools_used"):
            print("   ⚙ used: " + ", ".join(c["tool"] for c in result["tools_used"]) + "\n")
        if mems:
            print("   ↳ recalled: " + " | ".join(f"{m['text'][:38].replace(chr(10), ' ')}…" for m in mems[:3]) + "\n")

        history.append({"role": "user", "content": user})
        history.append({"role": "assistant", "content": reply})
        try:
            capture(user, reply, mems=mems, thinking=result.get("thinking"))
        except Exception as exc:
            log.fallback("capture_failed", exc, phase="cli")
            print(f"   warning: memory capture failed: {exc}\n")


if __name__ == "__main__":
    main()
