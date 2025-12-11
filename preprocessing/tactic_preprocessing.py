import json
from pathlib import Path
import markdown_it

def main():
    with open("tactics.md", "r", encoding="utf-8") as file:
        tactics_md = file.read()

    md = markdown_it.MarkdownIt()
    tokens = md.parse(tactics_md)

    tactics = set()

    for token in tokens:
        if token.type == 'heading_open' and token.tag == 'h2':
            next_token = tokens[tokens.index(token) + 1]
            if next_token.type == 'inline' and next_token.content != "_" and next_token.content != "(" and next_token.content != "admit" and next_token.content != "sorry":
                tactic = next_token.content
                tactics.add(tactic)
                
    tactics = sorted(list(tactics))
    print(f"Saving {len(tactics)} tactics to data/cache/tactic_vocab.json")

    Path("data/cache").mkdir(parents=True, exist_ok=True)
    with open("data/cache/tactic_vocab.json", "w") as f:
        json.dump(tactics, f, indent=2)

if __name__ == "__main__":
    main()
