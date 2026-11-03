# %%
from collections import Counter, defaultdict

from unidecode import unidecode


def load_first():
    ans = Counter()
    with open("nat2022.csv", "r") as file:
        file.readline()
        for line in file:
            line = line.split(";")
            name = unidecode(line[1])
            count = int(line[3])
            ans[name] += count
    ans.pop("_PRENOMS_RARES")
    return ans


def load_last():
    ans = Counter()
    with open("noms2008nat_txt.txt", "r") as file:
        n_splits = len(file.readline().split()) - 1
        for line in file:
            name, *counts = line.rsplit(None, n_splits)
            name = unidecode(name)
            count = sum(map(int, counts))
            ans[name] += count
    ans.pop("AUTRES NOMS")
    return ans


first = load_first()
last = load_last()
total_first = sum(first.values())
total_last = sum(last.values())


# Index: sorted chars of last name -> list of last names
last_index = defaultdict(list)
for lastname in last:
    key = "".join(sorted(lastname.replace(" ", "").replace("-", "").replace("'", "")))
    last_index[key].append(lastname)

# Pre-compute character counts for first names
first_counts: list[tuple[str, Counter]] = []
for firstname in first:
    normalized = firstname.replace(" ", "")
    count = Counter(normalized)
    first_counts.append((firstname, count))


def find_anagram_pair(query):
    """Find first+last name pair that is an anagram of query string.

    Args:
        query: Target string to find anagram for
        first_counts: Pre-computed first name counts from build_anagram_index
        last_index: Pre-computed last name index from build_anagram_index

    Returns:
        tuple: (firstname, lastname) if found, None otherwise
    """
    # Normalize and count characters in query
    query_normalized = query.upper().replace(" ", "")
    query_count = Counter(query_normalized)

    # Try each first name
    for firstname in first:
        count = Counter(firstname.replace(" ", "").replace("-", "").replace("'", ""))
        if not count <= query_count:
            continue
        remaining = query_count - count
        remaining_key = "".join(sorted(remaining.elements()))

        # Look up matching last names
        for lastname in last_index[remaining_key]:
            yield firstname, lastname


# %%


def sort_joint(item):
    firstname, lastname = item
    freq_first = first[firstname] / total_first
    freq_last = last[lastname] / total_last
    return freq_first * freq_last


def sort_first(item):
    firstname, lastname = item
    freq_first = first[firstname] / total_first
    freq_last = last[lastname] / total_last
    return freq_first, freq_last


if __name__ == "__main__":
    best = sorted(list(find_anagram_pair("elena Ourgaud")), key=sort_joint, reverse=True)
    for firstname, lastname in best:
        print(firstname.title(), lastname.title())
# %%
