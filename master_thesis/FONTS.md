# Times New Roman w finalnym PDF

Wymagany przez PJATK krój Times New Roman jest własnościowym fontem i nie jest
zapisywany w repozytorium. Target `make thesis` oczekuje czterech plików w
`/home/kuba/.cache/alphapulse-fonts/` (katalog można zmienić zmienną
`TIMES_NEW_ROMAN_DIR`):

- `times.ttf` — Regular,
- `timesbd.ttf` — Bold,
- `timesi.ttf` — Italic,
- `timesbi.ttf` — Bold Italic.

Na komputerze użytym do finalnej kompilacji fonty pobrano z archiwum Microsoft
Core Fonts `times32.exe` wskazanego przez pakiet Ubuntu
`ttf-mscorefonts-installer`. Przed rozpakowaniem zweryfikowano SHA-256 archiwum:

```text
db56595ec6ef5d3de5c24994f001f03b2a13e37cee27bc25c58f6f43e8f807ab
```

`make thesis` przerywa budowę, jeżeli brakuje którejkolwiek odmiany, dzięki czemu
PDF do oddania nie powstanie przypadkowo z fontem zastępczym. Po kompilacji można
potwierdzić osadzenie poleceniem:

```bash
pdffonts output/pdf/AlphaPulse_master_thesis.pdf
```
