#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# File name: toulouse_library.py
# Author: J.M. Barrie
# Date created: 2025-09-29
# Version = "1.0"
# License =  "CC0"
# Read =  "Peter Pan; or, the Boy Who C/Wouldn't Grow Up"
# Look-at =  "Alice In Wonderland Fashion Editorial by Annie Leibovitz"
# =============================================================================
""" Standalone analysis script for the Toulouse public library dataset."""
# =============================================================================

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple
from io import StringIO
import os
import glob
import json

import chardet
import pandas as pd
from collections import Counter, defaultdict

# -----------------------------
# Configuration
# -----------------------------
@dataclass(frozen=True)
class Config:
    """Configuration for a single analysis run.

    Attributes:
        dataset_folder: Path to the dataset directory.
        output_folder: Path to the output directory.
        discriminator: Identifier used to match CSV filenames and to name output files.
        discriminator_list: Tuple of available discriminators to iterate over.
        top_n: Number of top items to keep in each list.
        output_json: Output JSON filename (constructed from discriminator).
        year_field: CSV column name containing the year.
        popularity_field: CSV column name containing the popularity/loan counts.
        title_field: CSV column name containing item title.
        author_field: CSV column name containing item author.
        cat1_field: CSV column name containing category 1.
        cat1_selector: Tuple of Cat 1 values to keep (filter).
        composite_field: Name of the composite key field (title + author).
        replacements: Mapping of mojibake substrings to replacements.
    """

    dataset_folder: str = "dataset"
    output_folder: str = "output"
    discriminator: str = None
    discriminator_list: Tuple[str, ...] = ("films", "imprimes", "cds")
    top_n: int = 10
    output_json: str = "" # set in __post_init__
    year_field: str = "ANNEE"
    popularity_field: str = "Nbre de prêts"
    title_field: str = "TITRE"
    author_field: str = "AUTEUR"
    cat1_field: str = "Cat 1"
    cat1_selector: Tuple[str, ...] = ("A", "E")
    composite_field: str = "Item_ID"
    replacements: Mapping[str, str] = None

    def __post_init__(self) -> None:
        """Post-initialization to set computed defaults.

        Constructs the output filename from the discriminator and ensures
        a default replacements mapping is present.
        """
        object.__setattr__(self, "output_json", f"results_{self.discriminator}.json")
        if self.replacements is None:
            object.__setattr__(self, "replacements", {
                'ãa': 'â', 'ãe': 'ê', 'ãi': 'î', 'ão': 'ô',
                'ãu': 'û', 'âe': 'é', 'áa': 'à', 'áe': 'è',
                'ðc': 'ç'
            })


# -----------------------------
# Utilities
# -----------------------------
def find_first_matching_file(pattern: str) -> str:
    """Find the first filesystem path matching a glob pattern.

    Args:
        pattern: Glob pattern to search for.

    Returns:
        The first matching filepath.

    Raises:
        FileNotFoundError: If no matches are found.
    """
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No file matching pattern: {pattern!r}")
    return matches[0]


def detect_encoding(filepath: str, sample_bytes: int = 200) -> str:
    """Detect the text encoding of a file using a sample of bytes.

    Args:
        filepath: Path to the file to inspect.
        sample_bytes: Number of bytes to read for detection (default: 200).

    Returns:
        The detected encoding string or 'utf-8' as fallback.
    """
    with open(filepath, "rb") as f:
        raw = f.read(sample_bytes)
    result = chardet.detect(raw)
    return result.get("encoding") or "utf-8"


def fix_line(line: str) -> str:
    """Clean a single CSV line by stripping trailing whitespace and fixing a specific artifact.

    The function removes trailing newline/whitespace and replaces the first occurrence of
    the ';-;' artifact with ';' only if the line ends with a Cat 1 marker (A, E, B, P).

    Args:
        line: Raw line from the CSV file.

    Returns:
        The cleaned line.
    """
    s = line.rstrip()
    if s and s[-1] in {"A", "E", "B", "P"} and ";-;" in s:
        s = s.replace(";-;", ";", 1)
    return s


def apply_replacements(val: object, repl: Mapping[str, str]) -> object:
    """Apply string replacements to repair mojibake artifacts.

    Args:
        val: Value to process (only strings are changed).
        repl: Mapping of substrings to replace (wrong -> correct).

    Returns:
        The modified string if input was a string; otherwise the original value.
    """
    if not isinstance(val, str):
        return val
    for wrong, right in repl.items():
        val = val.replace(wrong, right)
    return val


# -----------------------------
# Analysis Class
# -----------------------------
class LoanAnalysis:
    """Perform loan data analysis and produce structured results.

    The class encapsulates loading, cleaning, aggregation, trend detection,
    and result serialization logic for one discriminator (Config).
    """

    def __init__(self, cfg: Config) -> None:
        """Initialize the analysis instance.

        Args:
            cfg: Configuration for the analysis run.
        """
        self.cfg = cfg
        self.cat1_map: Dict[str, str] = {}

    def load_and_clean_csv(self, filepath: str, encoding: Optional[str] = None) -> pd.DataFrame:
        """Load CSV, fix common artifacts, coerce types, and build composite keys.

        Args:
            filepath: Path to the CSV file.
            encoding: Optional encoding to use; if None, encoding will be detected.

        Returns:
            A cleaned pandas DataFrame ready for analysis.

        Raises:
            ValueError: If the CSV is empty.
        """
        enc = encoding or detect_encoding(filepath)
        with open(filepath, "r", encoding=enc, errors="replace") as fin:
            lines = fin.readlines()
        if not lines:
            raise ValueError("Empty CSV file")
        header = lines[0].rstrip("\n\r")
        data_lines = [fix_line(l) for l in lines[1:]]
        unified = StringIO("\n".join([header] + data_lines))

        df = pd.read_csv(
            unified,
            delimiter=";",
            engine="python",
            encoding=enc,
            encoding_errors="replace"
        )

        for c in df.select_dtypes(include=["object"]).columns:
            df[c] = df[c].apply(lambda v: apply_replacements(v, self.cfg.replacements))

        df[self.cfg.year_field] = pd.to_numeric(df[self.cfg.year_field], errors="coerce").astype("Int64")
        df[self.cfg.popularity_field] = pd.to_numeric(df[self.cfg.popularity_field], errors="coerce").astype("Int64")

        df[self.cfg.composite_field] = (
            df[self.cfg.title_field].astype(str)
            + " --- "
            + df[self.cfg.author_field].astype(str)
        )

        grp = df.groupby(self.cfg.composite_field)[self.cfg.cat1_field].unique()
        self.cat1_map = {
            k: (", ".join(sorted(v)) if len(v) > 1 else v[0])
            for k, v in grp.to_dict().items()
        }

        df = df[df[self.cfg.cat1_field].isin(self.cfg.cat1_selector)]

        return df

    def aggregate_by(self, df: pd.DataFrame, groupby_fields: Iterable[str], key_field: str,
                     value_field: str) -> pd.DataFrame:
        """Aggregate the DataFrame by given fields summing the value field.

        Args:
            df: Source DataFrame.
            groupby_fields: Iterable of fields to group by (may be empty).
            key_field: Field to include as key in the aggregation (composite).
            value_field: Field whose values will be summed.

        Returns:
            Aggregated DataFrame with groupby_fields + key_field and summed value_field.
        """
        fields = list(groupby_fields) + [key_field]
        return df.groupby(fields, as_index=False)[value_field].sum()

    def top_n_from_agg(self, agg_df: pd.DataFrame, value_field: str, top_n: int) -> pd.DataFrame:
        """Return top-n rows from an aggregated DataFrame ordered by value_field descending.

        Args:
            agg_df: Aggregated DataFrame with value_field present.
            value_field: Column used to rank entries.
            top_n: Number of top records to return.

        Returns:
            DataFrame containing the top_n rows.
        """
        return agg_df.sort_values(value_field, ascending=False).head(top_n)

    def group_top_n(self, df: pd.DataFrame, groupby_fields: Iterable[str],
                    key_field: str, value_field: str, top_n: int) -> Dict[str, List[dict]]:
        """Compute top-n items per group.

        Args:
            df: Source DataFrame.
            groupby_fields: Fields used to partition the dataset (e.g., Cat1 or (Year, Cat1)).
            key_field: Composite key field name.
            value_field: Field with numeric values to aggregate.
            top_n: Number of top items to keep per group.

        Returns:
            Mapping from group key (string) to a list of record dicts (top-n for that group).
        """
        agg = self.aggregate_by(df, groupby_fields, key_field, value_field)
        result: Dict[str, List[dict]] = {}
        gb = list(groupby_fields)
        for group_vals, sub in agg.groupby(gb):
            key = ", ".join(map(str, group_vals)) if isinstance(group_vals, tuple) else str(group_vals)
            result[key] = self.top_n_from_agg(sub, value_field, top_n).to_dict(orient="records")
        return result

    def _build_pivot(self, df: pd.DataFrame, index: str, columns: str, values: str) -> pd.DataFrame:
        """Create a pivot table used for trend and disappearance calculations.

        Args:
            df: Source DataFrame.
            index: Field to use as pivot index (composite key).
            columns: Field to use as pivot columns (year).
            values: Field to aggregate into pivot cells.

        Returns:
            Pivoted DataFrame with zeros filled for missing combinations.
        """
        return df.pivot_table(index=index, columns=columns, values=values, aggfunc="sum", fill_value=0)

    def compute_trends(self, pivot_df: pd.DataFrame, trend: str = "progression", top_n: Optional[int] = None) -> pd.DataFrame:
        """Compute items with upward or downward trends.

        The algorithm:
        - Keeps only items with >= 2 positive years.
        - Finds the first and last year with positive counts per item.
        - Computes sum at first and last positive year and their difference.
        - Returns top-n items with positive diffs for 'progression' or negative for 'regression'.

        Args:
            pivot_df: Pivot table with composite items as index and years as columns.
            trend: Either 'progression' or 'regression'.
            top_n: Optional override for number of rows to return.

        Returns:
            DataFrame with columns [composite_field, first_year, last_year, sum_first, sum_last, diff].
        """
        top_n = top_n or self.cfg.top_n
        mask = (pivot_df > 0).sum(axis=1) >= 2
        pivot_filtered = pivot_df[mask].copy()
        if pivot_filtered.empty:
            return pd.DataFrame(columns=[self.cfg.composite_field, "first_year", "last_year", "sum_first", "sum_last", "diff"])

        first_year = pivot_filtered.apply(lambda r: int(r[r > 0].index.min()), axis=1)
        last_year = pivot_filtered.apply(lambda r: int(r[r > 0].index.max()), axis=1)

        sum_first = []
        sum_last = []
        for idx, fy, ly in zip(pivot_filtered.index, first_year, last_year):
            sum_first.append(int(pivot_filtered.at[idx, fy]))
            sum_last.append(int(pivot_filtered.at[idx, ly]))

        df_trend = pd.DataFrame({
            self.cfg.composite_field: pivot_filtered.index,
            "first_year": first_year.values,
            "last_year": last_year.values,
            "sum_first": sum_first,
            "sum_last": sum_last,
            "diff": [s_l - s_f for s_f, s_l in zip(sum_first, sum_last)]
        }).reset_index(drop=True)

        if trend == "progression":
            return df_trend[df_trend["diff"] > 0].sort_values("diff", ascending=False).head(top_n)
        if trend == "regression":
            return df_trend[df_trend["diff"] < 0].sort_values("diff", ascending=True).head(top_n)
        raise ValueError("trend must be 'progression' or 'regression'.")

    def compute_trend_by_group(self, df: pd.DataFrame, group_field: str, value_field: str, trend: str) -> Dict[str, List[dict]]:
        """Compute top trend items for each group value.

        Args:
            df: Source DataFrame.
            group_field: Field to group by (e.g., Cat1).
            value_field: Numeric field used for trend computation.
            trend: 'progression' or 'regression'.

        Returns:
            Mapping from group value to list of trend record dicts.
        """
        result: Dict[str, List[dict]] = {}
        for grp, sub in df.groupby(group_field):
            pivot = self._build_pivot(sub, self.cfg.composite_field, self.cfg.year_field, value_field)
            trend_df = self.compute_trends(pivot, trend=trend)
            for col in ("sum_first", "sum_last", "diff"):
                if col in trend_df:
                    trend_df[col] = trend_df[col].astype(int)
            result[str(grp)] = trend_df.to_dict(orient="records")
        return result

    def compute_sudden_disappearances(self, pivot_df: pd.DataFrame, top_n: Optional[int] = None) -> pd.DataFrame:
        """Detect items that suddenly disappear after a last positive year.

        An item is considered a sudden disappearance if:
        - It has at least one positive year.
        - The year immediately after its last positive year exists in pivot columns and has a zero value.

        Args:
            pivot_df: Pivot table with composite index and year columns.
            top_n: Optional override for maximum results.

        Returns:
            DataFrame with columns [composite_field, last_year, loan_last] sorted by loan_last descending.
        """
        top_n = top_n or self.cfg.top_n
        records = []
        for item in pivot_df.index:
            nonzero = pivot_df.columns[pivot_df.loc[item] > 0]
            if len(nonzero) == 0:
                continue
            last_year = nonzero.max()
            if (last_year + 1) in pivot_df.columns and pivot_df.loc[item, last_year + 1] == 0:
                records.append({
                    self.cfg.composite_field: item,
                    "last_year": int(last_year),
                    "loan_last": int(pivot_df.loc[item, last_year])
                })
        if not records:
            return pd.DataFrame(columns=[self.cfg.composite_field, "last_year", "loan_last"])
        return pd.DataFrame(records).sort_values("loan_last", ascending=False).head(top_n)

    def compute_disappearances_by_group(self, df: pd.DataFrame, group_field: str, value_field: str) -> Dict[str, List[dict]]:
        """Compute sudden disappearances per group.

        Args:
            df: Source DataFrame.
            group_field: Field to group by (e.g., Cat1).
            value_field: Numeric field used to build the pivot.

        Returns:
            Mapping from group value to disappearance record dicts.
        """
        result: Dict[str, List[dict]] = {}
        for grp, sub in df.groupby(group_field):
            pivot = self._build_pivot(sub, self.cfg.composite_field, self.cfg.year_field, value_field)
            df_disp = self.compute_sudden_disappearances(pivot)
            result[str(grp)] = df_disp.to_dict(orient="records")
        return result

    def attach_cat1(self, record: dict) -> dict:
        """Attach Cat 1 value to a result record based on the composite key.

        Args:
            record: Result record dictionary which may contain the composite key.

        Returns:
            The same record with Cat 1 attached when available.
        """
        item = record.get(self.cfg.composite_field)
        if item in self.cat1_map:
            record[self.cfg.cat1_field] = self.cat1_map[item]
        return record

    def update_records_with_cat(self, records: Iterable[dict]) -> List[dict]:
        """Attach Cat1 to multiple records.

        Args:
            records: Iterable of record dicts.

        Returns:
            List of records with Cat1 attached when available.
        """
        return [self.attach_cat1(dict(r)) for r in records]

    @staticmethod
    def flatten_top_lists(input_data) -> List[dict]:
        """Flatten nested result structures (lists/dicts) into a single list of dicts.

        This helper is used to count nominations across various nested result containers.

        Args:
            input_data: Nested structure containing lists or dicts of records.

        Returns:
            Flat list of record dicts.
        """
        flat: List[dict] = []
        if isinstance(input_data, list):
            flat.extend(input_data)
        elif isinstance(input_data, dict):
            for v in input_data.values():
                if isinstance(v, list):
                    flat.extend(v)
                elif isinstance(v, dict):
                    flat.extend(LoanAnalysis.flatten_top_lists(v))
        return flat

    def run_all(self, df: pd.DataFrame) -> Dict[str, dict]:
        """Run the full suite of analyses and return a structured results dict.

        The produced keys mirror the original script:
        - topN_all_years_all_cat1
        - topN_by_year_all_cat1
        - topN_all_years_separated_cat1
        - topN_by_year_separated_cat1
        - topN_titles_occurring_most_all_cat1
        - topN_titles_occurring_most_separated_cat1
        - topN_progressions_separated_cat1
        - topN_progressions_all_cat1
        - topN_regressions_separated_cat1
        - topN_regressions_all_cat1
        - topN_sudden_disappearances_separated_cat1
        - topN_sudden_disappearances_all_cat1
        - topN_most_nominated_entries

        Args:
            df: Cleaned DataFrame for this discriminator.

        Returns:
            Dictionary mapping analysis names to result dicts.
        """
        R = {}
        C = self.cfg

        # topN overall across all years (aggregated)
        agg_all = self.aggregate_by(df, [], C.composite_field, C.popularity_field)
        topN_all = self.top_n_from_agg(agg_all, C.popularity_field, C.top_n).to_dict(orient="records")
        R["topN_all_years_all_cat1"] = {
            "question": f"Top {C.top_n} most popular items overall",
            "result": self.update_records_with_cat(topN_all)
        }

        # Top per year (merged across Cat1)
        agg_year = self.aggregate_by(df, [C.year_field], C.composite_field, C.popularity_field)
        by_year: Dict[int, List[dict]] = {}
        for yr, sub in agg_year.groupby(C.year_field):
            recs = self.top_n_from_agg(sub, C.popularity_field, C.top_n).to_dict(orient="records")
            by_year[int(yr)] = self.update_records_with_cat(recs)
        R["topN_by_year_all_cat1"] = {
            "question": f"Top {C.top_n} most popular items per year, all audiences merged",
            "result": by_year
        }

        # Tops separated by Cat1
        R["topN_all_years_separated_cat1"] = {
            "question": f"Top {C.top_n} most popular items across all years, categorized by audience (A: adults, E: kids)",
            "result": self.group_top_n(df, [C.cat1_field], C.composite_field, C.popularity_field, C.top_n)
        }
        R["topN_by_year_separated_cat1"] = {
            "question": f"Top {C.top_n} most popular items per year, categorized by audience (A: adults, E:kids)",
            "result": self.group_top_n(df, [C.year_field, C.cat1_field], C.composite_field, C.popularity_field, C.top_n)
        }

        # Titles appearing most often across yearly topNs (merged)
        merged_counter = Counter()
        for recs in R["topN_by_year_all_cat1"]["result"].values():
            for rec in recs:
                merged_counter[rec[C.composite_field]] += 1
        R["topN_titles_occurring_most_all_cat1"] = {
            "question": f"Top {C.top_n} titles that appear most frequently among top tens (all audiences)",
            "result": [
                {C.composite_field: k, "appearances": v, C.cat1_field: self.cat1_map.get(k, "")}
                for k, v in merged_counter.most_common(C.top_n)
            ]
        }

        # Titles appearing most often per Cat1
        temp_counter: Dict[str, Counter] = defaultdict(Counter)
        for group_key, recs in R["topN_by_year_separated_cat1"]["result"].items():
            cat = group_key.split(",")[-1].strip() if "," in group_key else group_key
            for rec in recs:
                temp_counter[cat][rec[C.composite_field]] += 1

        sep_occ: Dict[str, List[dict]] = {}
        for cat, counter in temp_counter.items():
            sep_occ[cat] = [
                {C.composite_field: k, "appearances": v, C.cat1_field: cat}
                for k, v in counter.most_common(C.top_n)
            ]
        R["topN_titles_occurring_most_separated_cat1"] = {
            "question": f"Top {C.top_n} titles that appear most frequently among top tens (categorized by audiences)",
            "result": sep_occ
        }

        # Progressions/regressions and disappearances
        R["topN_progressions_separated_cat1"] = {
            "question": f"Top {C.top_n} progressions, categorized by audience",
            "result": self.compute_trend_by_group(df, C.cat1_field, C.popularity_field, trend="progression")
        }

        pivot_all = self._build_pivot(df, C.composite_field, C.year_field, C.popularity_field)
        prog_all = self.compute_trends(pivot_all, trend="progression")
        prog_all = prog_all.astype({"sum_first": int, "sum_last": int, "diff": int})
        R["topN_progressions_all_cat1"] = {
            "question": f"Top {C.top_n} progressions, overall",
            "result": self.update_records_with_cat(prog_all.to_dict(orient="records"))
        }

        R["topN_regressions_separated_cat1"] = {
            "question": f"Top {C.top_n}, downward trend, categorized by audience",
            "result": self.compute_trend_by_group(df, C.cat1_field, C.popularity_field, trend="regression")
        }

        reg_all = self.compute_trends(pivot_all, trend="regression")
        reg_all = reg_all.astype({"sum_first": int, "sum_last": int, "diff": int})
        R["topN_regressions_all_cat1"] = {
            "question": f"Top {C.top_n} downward trend, overall",
            "result": self.update_records_with_cat(reg_all.to_dict(orient="records"))
        }

        R["topN_sudden_disappearances_separated_cat1"] = {
            "question": f"Top {C.top_n} shooting stars categorized by audience",
            "result": self.compute_disappearances_by_group(df, C.cat1_field, C.popularity_field)
        }

        disp_all = self.compute_sudden_disappearances(pivot_all)
        R["topN_sudden_disappearances_all_cat1"] = {
            "question": f"Top {C.top_n} shooting stars, overall",
            "result": self.update_records_with_cat(disp_all.to_dict(orient="records"))
        }

        # Nomination counting across many result lists to find frequently nominated titles
        keys = [
            "topN_all_years_all_cat1",
            "topN_by_year_all_cat1",
            "topN_all_years_separated_cat1",
            "topN_by_year_separated_cat1",
            "topN_sudden_disappearances_all_cat1",
            "topN_sudden_disappearances_separated_cat1",
            "topN_titles_occurring_most_separated_cat1",
            "topN_progressions_all_cat1",
            "topN_progressions_separated_cat1",
            "topN_regressions_all_cat1",
            "topN_regressions_separated_cat1"
        ]
        nom_counter = Counter()
        for k in keys:
            data = R.get(k, {}).get("result", {})
            for rec in self.flatten_top_lists(data):
                nom_counter[rec[C.composite_field]] += 1

        topN_nominated = [
            {C.composite_field: k, "nominations": v, C.cat1_field: self.cat1_map.get(k, "")}
            for k, v in nom_counter.most_common(C.top_n)
        ]
        R["topN_most_nominated_entries"] = {
            "question": f"Top {C.top_n} most nominated entries",
            "result": topN_nominated
        }

        return R


# -----------------------------
# Entrypoint
# -----------------------------
def main(cfg: Optional[Config] = None) -> None:
    """Main entrypoint: iterate configured discriminators and run analysis for each.

    For each discriminator this function:
    - finds a CSV matching "*{discriminator}*.csv"
    - loads and cleans the CSV
    - runs the analysis suite
    - writes results to results_{discriminator}.json

    Args:
        cfg: Optional Config instance; if not provided a default Config is used.
    """
    cfg = cfg or Config()
    for disc in cfg.discriminator_list:
        local_cfg = Config(
            dataset_folder=cfg.dataset_folder,
            output_folder=cfg.output_folder,
            discriminator=disc,
            discriminator_list=cfg.discriminator_list,
            top_n=cfg.top_n,
            year_field=cfg.year_field,
            popularity_field=cfg.popularity_field,
            title_field=cfg.title_field,
            author_field=cfg.author_field,
            cat1_field=cfg.cat1_field,
            cat1_selector=cfg.cat1_selector,
            composite_field=cfg.composite_field,
            replacements=cfg.replacements
        )

        pattern = f"*{local_cfg.discriminator}*.csv"
        try:
            filepath = find_first_matching_file(os.path.join(local_cfg.dataset_folder, pattern))
        except FileNotFoundError:
            print(f"No CSV found for discriminator {local_cfg.discriminator!r} (pattern: {pattern}). Skipping.")
            continue

        enc = detect_encoding(filepath)
        print(f"[{local_cfg.discriminator}] Detected encoding: {enc}")

        analysis = LoanAnalysis(local_cfg)
        df = analysis.load_and_clean_csv(filepath, encoding=enc)
        
        results = analysis.run_all(df)

        output_filepath = os.path.join(local_cfg.output_folder, "results_" + local_cfg.discriminator + ".json")
        # Create output folder if it doesn't exist
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        with open(output_filepath, "w", encoding="utf-8") as fout:
            json.dump(results, fout, ensure_ascii=False, indent=4)
        print(f"Wrote results to {output_filepath}")

if __name__ == "__main__":
    main()
