from dataclasses import dataclass, field
import numbers
from typing import Callable, Dict
import warnings

import numpy as np
import pandas as pd
from tabulate import tabulate


@dataclass
class ColumnFormat:
    name: str
    fun: Callable = field(default_factory=lambda: lambda x: x)


def merge_to_multiindex_column(dfs: Dict[str, pd.DataFrame], index=None) -> pd.DataFrame:
    merged_df = None
    for name, df in dfs.items():
        # copy and create multiindex column
        df = df.copy()

        # set index
        if index is not None:
            df.set_index(index, inplace=True)

        # create multiindex column
        df.columns = pd.MultiIndex.from_product([[name], df.columns])

        # merge
        if merged_df is None:
            merged_df = df
        else:
            merged_df = merged_df.merge(df, left_index=True, right_index=True, how='outer')

    return merged_df


def _is_missing(x):
    if x is None:
        return True
    elif isinstance(x, numbers.Number):
        return np.isnan(x)
    elif isinstance(x, np.ndarray):
        return np.any(np.isnan(x))
    else:
        return x in ['nan', 'NaN', 'None']


def print_results_table(df, column_formats=None, mode=None):
    # format table
    if column_formats is None:
        df_print = df
    else:
        # format values
        df_print = df.copy(deep=True)
        for col, col_fmt in column_formats.items():
            indices = df_print.columns.get_level_values(level=-1) == col
            df_print.loc[:, indices] = df_print.loc[:, indices].applymap(
                lambda x: '-' if _is_missing(x) else col_fmt.fun(x))

        # reorder columns
        if df_print.columns.nlevels == 1:
            try:
                col_names = [col for col in column_formats.keys() if col in df_print.columns]
                df_print = df_print.loc[:, col_names]
            except KeyError as err:
                warnings.warn(f"Could not reorder columns: {err}")
        else:
            columns = []
            for col_fmt_name in column_formats.keys():
                columns.extend([col for col in df_print.columns if col[-1] == col_fmt_name])
            df_print = df_print.loc[:, columns]

            # resort all level except the last
            df_print.sort_index(axis=1, level=range(df_print.columns.nlevels - 1), sort_remaining=False, inplace=True)

        # rename columns
        column_remap = {col: col_fmt.name for col, col_fmt in column_formats.items()}
        df_print.rename(columns=column_remap, inplace=True)

    # print table
    headers = list(map(lambda x: '\n'.join(x) if isinstance(x, (list, tuple)) else x,
                       df_print.columns.tolist()))

    if mode == 'raw':
        pd_options = [
            'display.colheader_justify', 'right',
            'display.max_columns', None,
            'display.max_rows', 5,
            'display.width', None,
        ]
        with pd.option_context(*pd_options):
            print(df_print)

    elif mode == 'latex':
        print(tabulate(df_print, headers=headers, tablefmt='latex'))

    else:
        print(tabulate(df_print, headers=headers, tablefmt='fancy_grid'))
