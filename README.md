# The Toulouse public library dataset
**Echoes of a city’s quiet catalog**

![header image](</img/mediatheque_jose_cabanis.jpg> "The toulouse public library.")
<br>*Médiathèque José Cabanis, Tolosa.*

## Genesis

> From the slow, patient stacks where the pink city’s breath settles into paper and dust, this catalogue was born. Not as a tidy ledger but as a tangled testimony of lives and seasons that bears the stains and slips of human passage. Children with bright breath, adults with careful hands all interfering the patient circulation of the months and years. This is a narrative of what a town borrows and forgets; so it might be read again.

## Table of Contents

<details>
<summary> Contents: (click to expand) </summary>

- [The Toulouse public library dataset](#the-toulouse-public-library-dataset)
  - [Genesis](#genesis)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Encountered issues](#encountered-issues)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Dataset](#dataset)
    - [Description](#description)
    - [Input CSV layout](#input-csv-layout)
    - [Sample JSON output](#sample-json-output)
  - [Configuration](#configuration)
    - [Default replacements mapping](#default-replacements-mapping)
  - [Code structure and function reference](#code-structure-and-function-reference)
    - [Module overview](#module-overview)
    - [Utilities](#utilities)
    - [LoanAnalysis class](#loananalysis-class)
  - [Contributing](#contributing)
  - [Disclaimer \& License](#disclaimer--license)
  - [Acknowledgments](#acknowledgments)

</details>

## Overview

This project processes CSV exports of library loan counts and produces a set of analytical JSON outputs per category (e.g., movies, books, music). It first cleans mojibake and CSV artifacts and then provides insights by aggregating counts, detecting trends (progressions/regressions), extracting shooting stars, and computing top-n lists in several types of groupings.

The analysis is available as:

- A standalone python script
- A didactic jupyter notebook

## Encountered issues

**Mojibake** and **misaligned CSV fields** quietly break analyses. Garbled characters ([mojibake](https://en.wikipedia.org/wiki/Mojibake)) turn informations into unreadable strings, while misplaced delimiters or extra semicolons shift columns so numeric counts and category flags land in the wrong fields. Together they lead to **silent data corruption** — resulting in wrong aggregations and incorrect classifications.

Repairing these issues requires careful encoding detection and targeted string fixes, plus defensive parsing (line cleanup and column coercion) so the analysis recovers true data without discarding valuable records.

> [!TIP]
> The provided [Jupyter notebook](https://github.com/brooks-code/toulouse-biblio-chronicle/blob/main/toulouse_public_library.ipynb) details these issues with pedagogy.

## Requirements

- Python 3.10+
- jupyterlab notebook if you want to run the notebook.
- Packages:
  - pandas
  - chardet

## Installation

1. Clone or download the repository:
   - `git clone https://github.com/brooks-code/toulouse-biblio-chronicle.git`
  
2. *(Recommended)* Within the script's location, create and activate a virtual environment :
   - `python -m venv venv`
   - `source venv/bin/activate`  (macOS / Linux) or `venv\Scripts\activate` (Windows)
  
3. Install the dependencies:
   - `pip install pandas chardet`

## Usage

Prerequisites:

- Check the CSV files are in a folder named `dataset` (or set a different folder in the *Config*). If not, they are available in this [repository](https://github.com/brooks-code/toulouse-biblio-chronicle) or [here](https://data.toulouse-metropole.fr/explore/?q=m%C3%A9diath%C3%A8que&sort=modified) (search for the top-500 datasets).
- CSV filenames should contain one of the keywords (default: `films`, `imprimes`, `cds`) so the script can find them via glob "*{discriminator}*.csv".

    - Run the `toulouse_public_library.ipynb` notebook.

    - Or run the script from the command line:

```python
python toulouse_library.py
```

- Outputs are written to `output/results_{discriminator}.json`.

> [!NOTE]
> By default the script iterates `("films", "imprimes", "cds")`.

## Dataset

### Description

Expected CSV: semi-colon delimited files containing rows with at least:

| Column name   | Type    | Description                     | Example      |
|---------------|---------|---------------------------------|--------------|
| ANNEE         | Integer | Year of the record              | 2019         |
| Nbre de prêts | Integer | Number of loans                 | 93           |
| TITRE         | String  | Title of the work               | Enfantillages|
| AUTEUR        | String  | Author                          | Aldebert     |
| Editeur       | String  | Publisher                       | Skyzo Music |
| Indice        | String  | Index                 | S099.2            |
| BIB           | String  | Library code                    | CABANIS       |
| COTE          | String  | Location label    | E 780.2 ALD     |
| Cat 1         | String  | Category label 1 (Audience)                | E       |
| Cat 2         | String  | Category label 2    (Media type)            | CD          |

- “–” represents missing data.
- category column (default: `Cat 1`) contains audience markers such as `A`, `E`, `BB`, `P`.
- Media type depends of the dataset type.

### Input CSV layout

```csv
ANNEE;Nbre de prêts;TITRE;AUTEUR;Editeur;Indice;BIB;COTE;Cat 1;Cat 2
2020;95;La promesse de l'aube;Barbier, Eric;Paris : Pathé, 2018;PROM;CABANIS;PROM;A;-
```

**Processing:**
> [!NOTE]
>- Lines may include artifact `;-;` before a Cat 1 marker; the loader will fix it.
>- Text fields may contain mojibake substrings (*see configurable replacements*) which will be fixed.
>- Only rows with Cat 1 in the `cat1_selector` (default `("A", "E")`) are kept.

### Sample JSON output

```json
{
    "top10_all_years_all_cat1": {
        "question": "Top ten most popular items overall",
        "result": [
            {
                "Item_ID": "Le Journal de Mickey --- -",
                "Nbre de prêts": 28857,
                "Cat 1": "E"
            }
            ]
    }
}
```

## Configuration

| Option | Purpose | Default value |
|---|---:|---|
| dataset_folder | Directory containing input CSV files | `"dataset"` |
| output_folder | Directory where JSON results are written | `"output"` |
| discriminator | Identifier used to select a CSV and name outputs (set per run) | `None` (*set per run*, e.g., `"films"`) |
| discriminator_list | Discriminators iterated by the main entrypoint | `("films", "imprimes", "cds")` |
| top_n | Number of top items to keep in each list | `10` |
| output_json | Output filename pattern (computed from discriminator) | `results_{discriminator}.json` |
| year_field | CSV column name containing the year | `"ANNEE"` |
| popularity_field | CSV column name containing popularity / loan counts | `"Nbre de prêts"` |
| title_field | CSV column name containing item title | `"TITRE"` |
| author_field | CSV column name containing item author | `"AUTEUR"` |
| cat1_field | CSV column name for audience category | `"Cat 1"` |
| cat1_selector | Cat1 values to keep (filter) | `("A", "E")` |
| composite_field | Generated composite key (title + author) | `"Item_ID"` |
| replacements | Mojibake repair mapping (wrong -> correct) | see below |

### Default replacements mapping

- `'ãa'` → `'â'`  
- `'ãe'` → `'ê'`  
- `'ãi'` → `'î'`  
- `'ão'` → `'ô'`  
- `'ãu'` → `'û'`  
- `'âe'` → `'é'`  
- `'áa'` → `'à'`  
- `'áe'` → `'è'`  
- `'ðc'` → `'ç'`

## Code structure and function reference

### Module overview

Script layout (high-level):

```txt
 LoanAnalysis Module
 ├─ Config (dataclass)
 ├─ Utilities
 │  ├─ find_first_matching_file(pattern)
 │  ├─ detect_encoding(filepath, sample_bytes=200)
 │  ├─ fix_line(line)
 │  └─ apply_replacements(val, repl)
 ├─ LoanAnalysis (class)
 │  ├─ load_and_clean_csv(filepath, encoding=None)
 │  ├─ aggregate_by(df, groupby_fields, key_field, value_field)
 │  ├─ top_n_from_agg(agg_df, value_field, top_n)
 │  ├─ group_top_n(df, groupby_fields, key_field, value_field, top_n)
 │  ├─ _build_pivot(df, index, columns, values)
 │  ├─ compute_trends(pivot_df, trend="progression", top_n=None)
 │  ├─ compute_trend_by_group(df, group_field, value_field, trend)
 │  ├─ compute_sudden_disappearances(pivot_df, top_n=None)
 │  ├─ compute_disappearances_by_group(df, group_field, value_field)
 │  ├─ attach_cat1(record)
 │  ├─ update_records_with_cat(records)
 │  ├─ flatten_top_lists(input_data)
 │  └─ run_all(df)
 └─ main(cfg=None)
```

### Utilities

<details>
<summary> Details: (click to expand) </summary>

- `find_first_matching_file(pattern: str) -> str`
  - Return the first file path matching glob pattern or raise FileNotFoundError.
  
    ```python
    pattern = os.path.join("dataset", "*films*.csv")
    path = find_first_matching_file(pattern)
    ```

- `detect_encoding(filepath: str, sample_bytes: int = 200) -> str`
  - Uses chardet on a byte sample and returns encoding (fallback `'utf-8'`).

    ```python
    enc = detect_encoding("dataset/sample.csv")
    ```

- `fix_line(line: str) -> str`
  - Strip trailing whitespace and fix `;-;` artifact when it appears before a `Cat 1` marker.

- `apply_replacements(val: object, repl: Mapping[str, str]) -> object`
  - Replace substrings in strings according to mapping; returns unchanged non-strings.
  
    ```python
    clean = apply_replacements("Author Ãa", {"Ãa": "Â"})
    ```

</details>

### LoanAnalysis class

Instantiate it with a Config object:
`analysis = LoanAnalysis(cfg)`

<details>
<summary> Key methods: (click to expand) </summary>

- `load_and_clean_csv(filepath: str, encoding: Optional[str] = None) -> pandas.DataFrame`
  - Reads a semicolon-delimited CSV, fixes lines, applies mojibake replacements, coerces numeric columns (`ANNEE`, `Nbre de prêts`) into nullable Int64, builds `Item_ID` as "TITLE --- AUTHOR", constructs internal cat1_map (composite -> cat1).
  - Returns cleaned DataFrame filtered by `cat1_selector`.
  
    ```python
    df = analysis.load_and_clean_csv("dataset/loans_films.csv", encoding="utf-8")
    ```

- `aggregate_by(df, groupby_fields, key_field, value_field) -> DataFrame`
  - Groups by groupby_fields + key_field and sums value_field.

    ```python
    agg = analysis.aggregate_by(df, [cfg.year_field], cfg.composite_field, cfg.popularity_field)
    ```

- `top_n_from_agg(agg_df, value_field, top_n) -> DataFrame`
  - Return top_n rows ordered by value_field descending.

- `group_top_n(df, groupby_fields, key_field, value_field, top_n) -> Dict[str, List[dict]]`
  - Produce top-n per group. Useful for per-year or per-cat results.

    ```python
    top_by_year_cat1 = analysis.group_top_n(df, [cfg.year_field, cfg.cat1_field], cfg.composite_field, cfg.popularity_field, 10)
    ```

- `_build_pivot(df, index, columns, values) -> DataFrame`
  - Create pivot with composite key as index, years as columns, counts filled with zeros.

- `compute_trends(pivot_df, trend="progression", top_n=None) -> DataFrame`
  - Detect progression/regression items:
    - Keeps items with >= 2 positive years.
    - Finds first and last positive year and sums values at those years.
    - Returns rows with diff > 0 for progressions, diff < 0 for regressions.
  - Returned columns: Item_ID, first_year, last_year, sum_first, sum_last, diff

    ```python
    pivot = analysis._build_pivot(df, cfg.composite_field, cfg.year_field, cfg.popularity_field)
    top_prog = analysis.compute_trends(pivot, trend="progression")
    ```

- `compute_trend_by_group(df, group_field, value_field, trend) -> Dict[str, List[dict]]`
  - Runs compute_trends per group value (e.g., per Cat1).

- `compute_sudden_disappearances(pivot_df, top_n=None) -> DataFrame`
  - Finds items whose last positive year is followed by an existing year with zero counts (sudden disappearance).
  - Returns columns: Item_ID, last_year, loan_last

    ```python
    disp = analysis.compute_sudden_disappearances(pivot)
    ```

- `compute_disappearances_by_group(df, group_field, value_field) -> Dict[str, List[dict]]`
  - Runs compute_sudden_disappearances per group value.

- `attach_cat1(record: dict) -> dict`
  - Attach a `Cat 1` value to a record based on its `Item_ID` using the internal `cat1_map`.

- `update_records_with_cat(records: Iterable[dict]) -> List[dict]`
  - Apply attach_cat1 to multiple records.

- `flatten_top_lists(input_data) -> List[dict]`
  - Recursively flatten nested list/dict containers of records to a flat list.

- `run_all(df: DataFrame) -> Dict[str, dict]`
  - Run a full suite of analyses (aggregations, trends, shooting stars) and return a structured result dict.

    ```python
    results = analysis.run_all(df)
    ```

</details>

## Contributing

Contributions are *welcome!* I appreciate your support: each contribution and feedback helps me **grow and improve**.

This project is intended as a practice on a real world use case, feel free to play with it. I'm open to any suggestion that will improve the code quality and deepen my software programming skills. If you'd like to contribute to this project, please fork the repository and submit a pull request with your changes.

## Disclaimer & License

This project is provided as-is, without warranty. Use at your own risk.

It is provided under a [Creative Commons CC0 license](https://creativecommons.org/publicdomain/zero/1.0/). See the [LICENSE](/LICENSE) file for details.

## Acknowledgments

![footer image](</img/toulouse_arche_vue_jean_jaurès.jpg> "View of the Jean Jaurès alley from the arch.")

Big thanks to the [Toulouse Metropolis open data portal](https://data.toulouse-metropole.fr/pages/accueil/) team for making these datasets publicly available:

- [top 500 books loans](https://data.toulouse-metropole.fr/explore/dataset/top-500-des-imprimes-les-plus-empruntes-a-la-bibliotheque-de-toulouse/information/)
- [top 500 movies loans](https://data.toulouse-metropole.fr/explore/dataset/top-500-des-films-les-plus-empruntes-a-la-bibliotheque-de-toulouse/information/)
- [top 500 songs loans](https://data.toulouse-metropole.fr/explore/dataset/top-500-des-cds-les-plus-empruntes-a-la-bibliotheque-de-toulouse/information/)

They are provided under a `Licence Ouverte v2.0 (Etalab)` [license](https://www.etalab.gouv.fr/wp-content/uploads/2017/04/ETALAB-Licence-Ouverte-v2.0.pdf).

Done in september 2025 by [brk🂽](github.com/brooks-code).

じゃあまた
