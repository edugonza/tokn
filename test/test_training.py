import json
import time
from pprint import pprint

from tokn import Tokn


def measure_performance(f, desc: str):
    start = time.perf_counter_ns()
    res = f()
    elapsed_time = (time.perf_counter_ns() - start) * 1e-9
    print(f"{desc}: {elapsed_time}s")
    return res


def test_training():
    with open("data/jokes.json", "rt") as fd:
        jokes = json.load(fd)

    jokes_str = [joke.get("body", "") for joke in jokes]

    tk = Tokn(100000)
    tk.train(jokes_str)

    pprint(
        {
            "Vocab size": len(tk.decode_map.keys()),
            # "token values": sorted(tk.decode_map.values()),
        }
    )

    sample = "This is a sample sentence."
    pprint([tk.decode([ti]) for ti in tk.encode(sample)])
    assert tk.decode(tk.encode(sample)) == sample

    sample = (
        "This is a sample sentence, but the real thing is that the content must"
        " get encoded and decoded."
    )
    pprint([tk.decode([ti]) for ti in tk.encode(sample)])
    assert tk.decode(tk.encode(sample)) == sample

    large_sample = (sample + " ") * 1000000

    old_tokens = measure_performance(
        lambda: tk.simple_encode(large_sample), desc="Simple tokenizer"
    )
    new_tokens = measure_performance(
        lambda: tk.tree_encode(large_sample), desc="Tree tokenizer"
    )

    print("Simple tokenizer: ", len(old_tokens))
    print("Tree tokenizer: ", len(new_tokens))

    assert tk.decode(tk.tree.tokenize(sample)) == sample
