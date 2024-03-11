### VERSION     0.0.1
### CREATED     February 22, 2024
### MODIFIED    February 24, 2024
### AUTHOR      Lucas Nelson
### CONTRIBUTOR Anusha Venkata Jami, Phung Pham, Lida Zhang


from typing import Iterable
from attrs import define, field, validators
from warnings import warn
from itertools import filterfalse

import pyspark
from pyspark.sql.functions import lit, col

from loguru import logger

### grouping operations
#   + customized, user inclinced to use .display() after
#       + e.g. select
#   + not-so-customized, user not so inclined to use .display() after
#       + e.g. filter
### testing
#   + see what engagements are doing; include functions in TidyDataFrame
#       + package common operations in class


class EmptyDataFrameWarning(UserWarning):
    pass


class NoRowsReturnedWarning(UserWarning):
    pass


class MethodNotFoundWarning(UserWarning):
    pass


@define
class TidyDataFrame:
    data: pyspark.sql.DataFrame = field(
        validator=validators.instance_of(pyspark.sql.DataFrame),
        repr=lambda self: f"DataFrame [{self.n_rows if self.n_rows is not None else '???'} rows x {self.n_cols if self.n_cols is not None else '???'} cols]",
    )
    toggle_count: bool = field(default=True, validator=validators.instance_of(bool))
    toggle_display: bool = field(default=True, validator=validators.instance_of(bool))
    toggle_timer: bool = field(default=True, validator=validators.instance_of(bool))
    n_rows: int = field(default=None)
    n_cols: int = field(default=-1)

    # def __getattr__(self, attr):
    #     '''Undefined methods sourced to data'''
    #     return getattr(self.data, attr)

    def _log_operation(
        self, operation="custom", message="method not covered by TidyDataFrame", level='INFO'
    ):
        """Simple logger invoked by masked methods."""
        logger_func = getattr(logger, level.strip().lower())
        logger_func(f"#> {operation}: {message}")

    def _format_message(self, content):
        message = content
        if isinstance(content, Iterable):
            N_CHAR_THRESHOLD = 50
            # if len(content) > 5:
            #     content = content[:5]
            message = ", ".join(content)
            if len(message) > N_CHAR_THRESHOLD:
                message = message[: (N_CHAR_THRESHOLD - 5)] + "..."
        return message
    
    @property
    def columns(self):
        """Return all column names as a list"""
        return self.data.columns
    
    @property
    def dtypes(self):
        """Return all column names and data types as a list"""
        return self.data.dtypes
    
    @property
    def describe(self, *cols):
        """Compute basic statistics for numeric and string columns."""
        return self.data.describe(*cols)


    def display(self):
        """
        Control execution of display method

        This method masks the `pyspark.sql.DataFrame.display` method. This method does not
        mask the native PySpark display function.

        Often, the `.display()` method will need to be disabled for logging purposes. Similar
        to toggling the `.count()` method, users can temporarily disable a DataFrame's
        ability to display to the console by passing `toggle_display = True`.
        """
        if not self.toggle_display:
            self._log_operation(operation="display", message="feature toggled off", level="WARNING")
        else:
            self.data.display()

    def count(self, result: pyspark.sql.DataFrame = None):
        """
        Retrieve number of rows from DataFrame-like object

        The `.count()` method in PySpark has proven to be a benchmark's nightmare. In theory, this
        is due to a DataFrame persisting across multiple clusters, and coordinating a single result
        (e.g. row count) goes against the benefits of distributing computing. Rather than avoiding
        the problem altogether, this solution performs a layman's cache to reduce the need to
        invoke the `.count()` method.

        Depending on the nature of the request, the `.count()` method may not need to be invoked.
        This is controlled by the state of the `n_rows` attribute and `result` parameter. The first
        time `TidyDataFrame.count` is called, `n_rows` will be `None` - hence, a count will need
        to be computed. If a `result` is passed, this implies that the underlying `data` has
        changed, meaning `n_rows` is no longer accurate and `count` will need to be computed. If
        `n_rows` is initialized (not `None`) and no change in `data` is detected, then `n_rows` is
        simply retrieved and returned without the need for computing row count.

        Additionally, a handler layers the function to bypass retrieving the count. This can be
        controlled by the user when initializing a TidyDataFrame by passing the `toggle_count`
        parameter. (Contributed by Lida Zhang)
        """
        if self.toggle_count:
            if self.n_rows is None:  # not yet defined, compute row count
                self.n_rows = self.data.count()
                if self.n_rows == 0:
                    warn("Provided data is empty", EmptyDataFrameWarning)
            if result is not None:  # result computed, recompute row count
                self.n_rows = result.count()
                if self.n_rows == 0:
                    warn("Query returned no results", NoRowsReturnedWarning)
            return self.n_rows  # defined and no new result, return row count
        return None

    def select(self, *cols, _deprecated=True):
        """Select columns from DataFrame"""
        if not _deprecated:
            cols_pre = self.data.columns
            result = self.data.select(*cols)
            cols_post = result.columns

            def get_message(pre: list[str], post: list[str]) -> str:
                n_pre = len(pre)
                n_post = len(post)
                if n_pre == n_post:
                    return "no columns dropped"
                if n_post == 0:
                    return "all columns dropped"
                return f"selected {n_post:,} column(s) ({self._format_message(set(post))})"

            self._log_operation(
                operation="select", message=get_message(pre=cols_pre, post=cols_post)
            )
            return TidyDataFrame(
                result,
                toggle_count=self.toggle_count,
                toggle_display=self.toggle_display,
                n_rows=self.n_rows,
            )

    def filter(self, condition):
        """Filter observations from DataFrame based on condition"""

        n_rows_pre = self.count()
        result = self.data.filter(condition)
        n_rows_post = self.count(result)

        def get_message(pre: int, post: int):
            if not self.toggle_count:
                return f"count not performed"
            assert post <= pre, "Unsure how filter returned more rows?"
            if pre == post:
                return f"no rows removed, {pre:,} remaining"
            if post == 0:
                return f"all rows removed, {post:,} remaining"
            diff = pre - post
            return f"contains {pre:,} rows, removed {diff:,} rows ({diff / pre:.2%}), {post:,} remaining"

        self._log_operation(
            operation="filter", message=get_message(pre=n_rows_pre, post=n_rows_post)
        )
        return TidyDataFrame(
            data=result,
            toggle_count=self.toggle_count,
            toggle_display=self.toggle_display,
            n_rows=n_rows_post,
        )

    def where(self, condition):
        """Alias for `filter`"""
        return self.filter(condition)

    def drop(self, cols):
        """Drop columns from DataFrame (using `.select()`)"""
        all_cols = self.data.columns
        drop_cols = set(all_cols).difference(set(cols))
        return self.select(*drop_cols)

    def withColumn(self, colName, col):
        # needed to prevent masking of `col` function by `col` parameter
        import pyspark.sql.functions as f

        n_rows = self.data.count() if self.toggle_count else None
        result = TidyDataFrame(
            self.data.withColumn(colName=colName, col=col),
            toggle_count=self.toggle_count,
        )
        n_null = (
            result.data.select(colName).filter(f.col(colName).isNull()).count()
            if self.toggle_count
            else None
        )
        col_info = filterfalse(lambda t: t[0] != colName, result.data.dtypes)

        def get_message(total: int, invalid: int, col_info: tuple[str]):
            col_name, col_type = col_info
            if not self.toggle_count:
                qualification = "no count performed"
            else:
                assert invalid <= total, "Unsure how mutate returned more rows?"
                if invalid == total:
                    qualification = "all values are NULL"
                elif invalid == 0:
                    qualification = "no values are NULL"
                else:
                    qualification = f"with {invalid:,} ({(total - invalid) / total:.2%}) NULL values"
            return f"created '{col_name}' ({col_type}), {qualification}"

        self._log_operation(
            operation="mutate",
            message=get_message(
                total=n_rows,
                invalid=n_null,
                col_info=tuple(col_info)[0],  # yield, unnest
            ),
        )
        return result

    def withColumnRenamed(self, existing: str, new: str):
        result = self.data.withColumnRenamed(existing=existing, new=new)
        self._log_operation(
            operation="rename", message=f"column '{existing}' renamed to '{new}'"
        )
        return TidyDataFrame(
            result,
            toggle_count=self.toggle_count,
            toggle_display=self.toggle_display,
            n_rows=self.n_rows,
        )

    def join(self, other, on=None, how="inner"):
        n_rows_self = self.count()
        n_rows_other = other.count()
        result = self.data.join(other=other, on=on, how=how)
        n_rows_joined = self.count(result)

        def get_message(
            how: str, n_rows_left: int, n_rows_right: int, n_rows_joined: int
        ):
            if not self.toggle_count:
                return "count not performed"
            # assert statement vary by join type
            return f"({how}) matched {n_rows_joined:,} ({n_rows_joined / n_rows_left:.2%}) rows"

        self._log_operation(
            operation=f"join",
            message=get_message(
                how=how,
                n_rows_left=n_rows_self,
                n_rows_right=n_rows_other,
                n_rows_joined=n_rows_joined,
            ),
        )
        return TidyDataFrame(
            result,
            toggle_count=self.toggle_count,
            toggle_display=self.toggle_display,
            n_rows=self.n_rows,
        )

    def drop_duplicates(self, subset=None):
        """Drop duplicates (based on subset) from DataFrame"""
        n_rows_pre = self.count()
        result = self.data.drop_duplicates(subset=subset)
        n_rows_post = self.count(result)

        def get_message(pre: int, post: int):
            if not self.toggle_count:
                return "count not performed"
            return f"removed {pre - post:,} ({(pre - post) / pre:.2%}) duplicates"

        self._log_operation(
            operation="distinct", message=get_message(pre=n_rows_pre, post=n_rows_post)
        )
        return TidyDataFrame(
            result,
            toggle_count=self.toggle_count,
            toggle_display=self.toggle_display,
            n_rows=self.n_rows,
        )


    def dropDuplicates(self, subset=None):
        """Alias for `drop_duplicates`"""
        return self.drop_duplicates(subset=subset)
    

    def dropna(self, how='any', thresh=None, subset=None):
        """Return DataFrame omitting rows with null values"""
        return self.data.dropna(how=how, thresh=thresh, subset=subset)


    def distinct(self):
        n_rows_pre = self.count()
        result = self.data.distinct()
        n_rows_post = self.count(result)

        def get_message(pre: int, post: int):
            if not self.toggle_count:
                return "count not performed"
            return f"preserved {post:,} ({(pre - post) / pre:.2%}) rows"

        self._log_operation(
            operation="distinct", message=get_message(pre=n_rows_pre, post=n_rows_post)
        )
        return TidyDataFrame(
            result,
            toggle_count=self.toggle_count,
            toggle_display=self.toggle_display,
            n_rows=self.n_rows,
        )

    
    def union(self, other):
        result = self.data.union(other)
        self._log_operation(
            operation="union", message="...original"
        )
        return TidyDataFrame(
            result,
            toggle_count=self.toggle_count,
            toggle_display=self.toggle_display,
            n_rows=self.n_rows,
        )

    def unionAll(self, other):
        return self.union(other=other)
    
    def unionByName(self, other, allowMissingColumns = False):
        result = self.data.unionByName(other=other, allowMissingColumns=allowMissingColumns)
        self._log_operation(
            operation="union", message="...by name"
        )
        return TidyDataFrame(
            result,
            toggle_count=self.toggle_count,
            toggle_display=self.toggle_display,
            n_rows=self.n_rows,
        )

    def __getattr__(self, name):
        return getattr(self.data, name)
