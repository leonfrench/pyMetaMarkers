"""CSV import/export in the format used by the R package (one header line + CSV)."""

from __future__ import annotations

import datetime
import gzip as gzip_module
import re

import numpy as np
import pandas as pd

R_RESERVED = {
    "if", "else", "repeat", "while", "function", "for", "next", "break", "TRUE", "FALSE", "NULL",
    "Inf", "NaN", "NA", "NA_integer_", "NA_real_", "NA_character_", "in",
}


def today() -> str:
    return datetime.date.today().isoformat()


def format_number(x) -> str:
    """Format a number like R's as.character (15 significant digits)."""
    if isinstance(x, (bool, np.bool_)):
        return "TRUE" if x else "FALSE"
    return f"{x:.15g}"


def make_names(name) -> str:
    """Python version of R's make.names() for a single string."""
    name = re.sub(r"[^0-9A-Za-z._]", ".", str(name))
    if not name or not re.match(r"[A-Za-z]|\.(?![0-9])", name):
        name = "X" + name
    if name in R_RESERVED:
        name = name + "."
    return name


def _format_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Logical columns as TRUE/FALSE (as data.table::fwrite); to_csv formats the rest."""
    result = df.copy(deep=False)
    for column in df.columns:
        values = df[column]
        if pd.api.types.is_bool_dtype(values.dtype) or _is_boolean_object(values):
            text = np.where(values.fillna(False).to_numpy(dtype=bool), "TRUE", "FALSE").astype(object)
            text[values.isna().to_numpy()] = ""
            result[column] = text
    return result


def _is_boolean_object(values: pd.Series) -> bool:
    if values.dtype != object:
        return False
    present = values.dropna()
    return len(present) > 0 and all(isinstance(v, (bool, np.bool_)) for v in present)


def write_table(df: pd.DataFrame, filename, header: str, gzip: bool = True) -> str:
    path = str(filename) + (".gz" if gzip else "")
    if gzip:
        handle = gzip_module.open(path, "wt", newline="", compresslevel=6)  # R's default level
    else:
        handle = open(path, "w", newline="")
    with handle:
        handle.write(header + "\n")
        _format_for_csv(df).to_csv(handle, index=False, lineterminator="\n", float_format="%.15g", na_rep="")
    return path


def read_table(filename, header_size: int = 1) -> pd.DataFrame:
    result = pd.read_csv(
        filename, skiprows=header_size, keep_default_na=False, na_values=[""], float_precision="round_trip",
        true_values=["TRUE", "True", "true"], false_values=["FALSE", "False", "false"],
        dtype={"group": str, "cell_type": str, "gene": str},
        low_memory=False,  # infer each column's type from all rows, not chunk by chunk
    )
    for column in result.columns:
        values = result[column]
        if values.dtype == object and _is_boolean_object(values):
            if values.isna().any():
                result[column] = values.astype("boolean")
            else:
                result[column] = values.astype(bool)
    return result
