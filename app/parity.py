# Run the Python and the TypeScript versions on the same queries and packs and compare their results.
#   python app/parity.py ["query" ...]
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
QUERIES = ["Elena Ourgaud", "Louis Abraham", "Marie Curie", "Jean-Paul Sartre", "Zoë Ångström",
           "Ali Khan", "Wolfgang Amadeus Mozart", "Łukasz Żółć", "aurélie coursimault"]
PACKS = [[], ["--country", "FR"], ["--country", "JP"], ["--country", "US"]]
DETAILS = ("real", "first_people", "first_countries", "last_people", "last_countries", "real_countries")
LIMIT = 50


def run(cmd, args, queries):
    out = subprocess.run([*cmd, "--json", "-n", str(LIMIT), *args, *queries], capture_output=True, text=True, check=True)
    return [json.loads(line) for line in out.stdout.splitlines()]


def main():
    queries = sys.argv[1:] or QUERIES
    ok = True
    for args in PACKS:
        py = run([sys.executable, f"{HERE}/anagram.py"], args, queries)
        ts = run(["node", f"{HERE}/cli.ts"], args, queries)
        for a, b in zip(py, ts, strict=True):
            names = lambda res: [(r["first"], r["last"], r["country"], *(r[k] for k in DETAILS)) for r in res["results"]]
            countries = lambda res: [o["country"] for o in res["origin"] or []]
            same = all(a[k] == b[k] for k in ("found", "real_names", "real_people")) and names(a) == names(b)
            same &= countries(a) == countries(b)
            diff = max([abs(x["log_p"] - y["log_p"]) for x, y in zip(a["results"], b["results"])]
                       + [abs(x["p"] - y["p"]) for x, y in zip(a["origin"] or [], b["origin"] or [])], default=0.0)
            ok &= same and diff < 1e-9
            print(f"{'same' if same else 'DIFFERENT'}  {' '.join(args) or 'world'}  {a['query']}: {a['found']:,} names,"
                  f" {a['real_names']} real, largest difference {diff:.1e}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
