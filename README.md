# Anagranames

Who and where are your anagranames? Type your full name: the site finds the full names with exactly its letters, in
105 countries. Real people come from name_dataset (the 2021 Facebook leak); synthetic names come from a model of
first and last names per country. Everything runs in the browser.

Site: https://louisabraham.github.io/anagranames/ · How it works: [web/how.html](web/how.html)

## Layout

- `prepare_world.py`, `make_fr_insee.py`, `pairmodel.py`, `calibrate.py`, `quantize.py`, `export_app.py`: the data and
  the model, from name_dataset and INSEE (the base of France) to the packs in `app/data` (not in git: the file
  `app-data.tar` of the release `data-v2`).
- `app/anagram.ts`: the search (the site and the CLI `app/cli.ts`); `app/anagram.py`: the same search in Python, the
  reference of the tests (`app/parity.py`).
- `web/`: the site; `node web/build.mjs` builds `web/anagram.js` and `web/data`.

## Run

```sh
gh release download data-v2 --pattern app-data.tar --output - | tar -x -C app
node web/build.mjs
python3 -m http.server -d web 8000
```

Tests: `uv run --no-project --with playwright --with numpy python web/e2e.py` (Chromium; `--all`: 3 browsers and 5
phone and tablet profiles), and `uv run --no-project --with numpy python app/parity.py`.

Made by [Louis Abraham](https://louisabraham.github.io/).
