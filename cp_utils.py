import builtins
import re
import functools
from typing import Any, Union

import pyspark
from pyspark.sql import types
from pyspark.sql import functions as F

from cp_tidy import TidyDataFrame
from loguru import logger


RE_AMOUNT_FIELDS = "^.*_[eog]c$"
RE_NET_AMOUNT_FIELDS = "^net_.*_[eog]c$"
RE_BALANCE_FIELDS = "^.*_balance_[eog]c"
RE_DATE_FIELDS = "^.*_date.*$"


def cp_error(message, local_vars, local_keys):
    """Simple error handling given caller's local scope"""
    params_passed = [f"{param:<10} {local_vars.get(param)}" for param in local_keys]
    logger.warning(f"{message}!\n\t" + "\n\t".join(params_passed))


def get_columns(
    data: pyspark.sql.DataFrame,
    columns: list[str] = None,
    pattern: str = None,
    dtype: Any = None,
) -> list[str]:
    """
    Returns list of column names with various filtering methods.

    If `columns` is provided, all valid column names are returned. If `pattern`
    is provided, all column names are filtered by the pattern, only returning
    column names that match. If `dtype` is provided, all columns with the exact
    Spark Type are returned.

    The parameters are ranked in order of strictness, where the `columns` filter
    is more lenient than the `pattern` filter, and the `pattern` filter is more
    lenient than the `dtype` filter.
    """
    if columns is not None:
        return [column for column in data.columns if column in columns]
    if pattern is not None:
        return get_columns_by_pattern(data, pattern)
    if dtype is not None:
        return get_columns_by_dtype(data, dtype)
    return data.columns


def get_columns_by_dtype(
    data: pyspark.sql.DataFrame, dtype=types.StringType
) -> list[str]:
    """Extract columns from data by data type (`dtype`)."""
    dtype_columns = builtins.filter(
        lambda sf: isinstance(sf.dataType, dtype), data.schema
    )
    return list(map(lambda sf: sf.name, dtype_columns))


def get_columns_by_pattern(data: pyspark.sql.DataFrame, pattern: str) -> list[str]:
    """Extract columns from data by regular expression (`pattern`)."""
    pattern_columns = builtins.filter(
        lambda sf: isinstance(re.match(pattern, sf.name), re.Match), data.schema
    )
    return list(map(lambda sf: sf.name, pattern_columns))


def format_columns(func: callable):
    """
    Decorator for formatting columns of various data types.

    Responsible for handling pyspark.sql.DataFrame objects as well as
    TidyDataFrame objects, coercing to the latter if the input is not
    of that type. Additionally, handles multiple formats for the
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        ### handle case of mixed arguments/keyword arguments
        data = args[0] if args else kwargs.get("data")
        columns = args[1] if args else kwargs.get("columns")
        del kwargs["data"], kwargs["columns"]

        ### coerce to TidyDataFrame, track input DataFrame type
        _return_tidydataframe = True
        if not isinstance(data, TidyDataFrame):
            data = TidyDataFrame(data)
            _return_tidydataframe = False

        ### handle multiple cases of columns parameter
        if isinstance(columns, str):
            columns = columns.split(",")
            if isinstance(columns, list):
                columns = [c.strip() for c in columns]
            else:
                columns = [columns]

        ### perform result
        result = func(data=data, columns=columns, **kwargs)

        ### return type of input DataFrame
        if not _return_tidydataframe:
            return result.data
        return result

    return wrapper


@format_columns
def format_numeric(
    data: Union[pyspark.sql.DataFrame, TidyDataFrame],
    columns: Union[str, list[str]],
    dtype: Any = types.DecimalType(precision=38, scale=2),
) -> Union[pyspark.sql.DataFrame, TidyDataFrame]:
    """Format numeric column(s)"""
    return (
        data
        # extract values with meaningful numerical representation
        .withColumns(
            {
                key: F.regexp_extract(str=F.col(key), pattern="([\-\(\d\.]+)", idx=0)
                for key in columns
            }
        )
        # replace values with appropriate numerical representation
        .withColumns(
            {
                key: F.regexp_replace(str=F.col(key), pattern="\(", replacement="-")
                for key in columns
            }
        )
        # convert values to numerical representation
        .withColumns({key: F.col(key).cast(dtype) for key in columns})
    )


@format_columns
def format_datetime(
    data: Union[pyspark.sql.DataFrame, TidyDataFrame],
    columns: Union[str, list[str]],
    format: str = "MM/dd/yyyy",
    include_time: bool = False,
) -> Union[pyspark.sql.DataFrame, TidyDataFrame]:
    """Format datetime column(s)"""
    CDM_DATETIME_FORMAT = "yyyy-MM-dd HH:mm:ss"
    data = data.withColumns(
        {key: F.to_timestamp(F.col(key), format) for key in columns}
    )
    if not include_time:
        return data.withColumns({key: F.to_date(F.col(key)) for key in columns})
    return data.withColumns(
        {key: F.date_format(F.col(key), CDM_DATETIME_FORMAT) for key in columns}
    )


# def which_files(
#     path: str = "User Imported Data/EFT_UP",
#     pattern: str = ".*",
#     recursive: bool = False,
# ):
#     """
#     List files with (optional) pattern parameter.

#     Examples
#     ========
#     >>> which_files()
#     >>> which_files(pattern="*.csv")
#     >>> which_files(path="GL", pattern="*.xlsx", recursive=True)
#     """
#     RE_FILE_PATTERN = re.compile(pattern, flags=re.IGNORECASE)
#     is_valid_file = lambda fp: isinstance(re.search(RE_FILE_PATTERN, fp), re.Match)

#     all_files = cp.fs.ls(path=path, recursive=recursive, only_files=True)
#     filtered_files = list(builtins.filter(is_valid_file, all_files))

#     if len(filtered_files) < 1:
#         cp_error(
#             message="No files returned",
#             local_vars=locals(),
#             local_keys=["path", "pattern", "recursive"],
#         )
#         return None
#     if len(filtered_files) < 2:
#         return filtered_files[0]
#     return filtered_files


# def map_readin(
#     path: str,
#     pattern: str = ".*",
#     recursive: bool = False,
#     union_func: str = "unionByName",
#     cp_options: dict[str, str] = None,
#     **kwargs,
# ) -> pyspark.sql.DataFrame:
#     """
#     Functional multi-file readin alternative.

#     Note: the current implementation of multifile_readin does not allow for
#     multiple options to be passed for its default arguments (i.e. header).
#     """
#     cp_read = functools.partial(cp.read, **kwargs)
#     raw_files = which_files(path=path, pattern=pattern, recursive=recursive)
#     if isinstance(raw_files, str):
#         return cp_read(raw_files)
#     if not any(map(lambda kw: isinstance(kw, Iterable), kwargs.values())):
#         return functools.reduce(getattr(pyspark.sql.DataFrame, union_func), map(cp_read, raw_files))
