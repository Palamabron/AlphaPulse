# Rekordy historycznej serii HPO

Pliki w tym katalogu są generowane przez `master_thesis/generate_figures.py`
z lokalnej bazy końcowej serii HPO. Utrwalają rekordy dwóch kandydatów opisanych
w pracy oraz protokół badania, ponieważ źródłowy katalog `artifacts/` jest
ignorowany przez Git.

- `historical_protocol.json` — zapis protokołu i ograniczenia proweniencji.
- `trial_04_record.json` — pełna płaska konfiguracja i metryki kandydata MMC.
- `trial_22_record.json` — pełna płaska konfiguracja i metryki kandydata CORR.

Pola `audit_notes` odróżniają wartości zapisane w bazie od parametrów, które
historycznie nie wpływały na trening. Rekordy pozwalają audytować opublikowane
wyniki, ale nie zastępują brakującego snapshotu historycznego kodu i nie są
gotowymi konfiguracjami dla poprawionej implementacji.
