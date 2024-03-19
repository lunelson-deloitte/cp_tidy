from attrs import define, field, validators
from warnings import warn

import functools
import itertools

import pyspark
import pyspark.sql.functions as F

from loguru import logger

### grouping operations
#   + customized, user inclinced to use .display() after
#       + e.g. select
#   + not-so-customized, user not so inclined to use .display() after
#       + e.g. filter
### testing
#   + see what engagements are doing; include functions in TidyDataFrame
#       + package common operations in class
### (future) resources
#   + different notebooks for pyspark.sql.DataFrame and TidyDataFrame
#       + demo for value add
#   + comparison of "typical/daily" workflow and proposed workflow


@define
class TidyDataFrame:
    _data: pyspark.sql.DataFrame = field(validator=validators.instance_of(pyspark.sql.DataFrame))
    toggle_options: dict[str, bool] = field(default=dict(count=True, display=True))
    _n_rows: int = field(default=-1, validator=validators.instance_of(int))
    _n_cols: int = field(default=-1, validator=validators.instance_of(int))


    def __attrs_post_init__(self):
        self._n_rows = self.count()
        self._n_cols = len(self._data.columns)
        self._log_operation(">> enter >>", self.__repr__(data_type=type(self).__name__))

    def __repr__(self, data_type: str):
        """String representation of TidyDataFrame"""
        data_repr = f"{data_type}[{self.n_rows:,} rows x {self.n_cols:,} cols]"
        disabled_options_string = ""
        if data_type == 'TidyDataFrame':
            disabled_options = itertools.compress(self.toggle_options.keys(), self.toggle_options.values())
            disabled_options_string = f"(disabled: {', '.join(disabled_options)})"
        return f"{data_repr} {disabled_options_string}"

    def _log_operation(
        self,
        operation: str,
        message: str,
        level: str = "info",
    ):
        """Simple logger invoked by decorated methods."""
        getattr(logger, level)(f"#> {operation}: {message}")

    @property
    def data(self):
        self._log_operation("<< exit <<", self.__repr__(data_type=type(self._data).__name__))
        return self._data

    @property
    def columns(self):
        """Return all column names as a list"""
        return self._data.columns

    @property
    def dtypes(self):
        """Return all column names and data types as a list"""
        return self._data.dtypes

    @property
    def describe(self, *cols):
        """Compute basic statistics for numeric and string columns."""
        return self._data.describe(*cols)
    
    @property
    def _unknown_dimension():
        return "???"

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
            self._log_operation(
                operation="display", message="feature toggled off", level="warning"
            )
        else:
            self._data.display()

    def count(self, result: pyspark.sql.DataFrame = None):
        """
        Retrieve number of rows from DataFrame-like object

        The `.count()` method in PySpark has proven to be a benchmark's nightmare. In theory, this
        is due to a DataFrame persisting across multiple clusters, and coordinating a single result
        (e.g. row count) goes against the benefits of distributing computing. Rather than avoiding
        the problem altogether, this solution performs a layman's cache to reduce the need to
        invoke the `.count()` method.

        Depending on the nature of the request, the `.count()` method may not need to be invoked.
        This is controlled by the state of the `_n_rows` attribute and `result` parameter. The first
        time `TidyDataFrame.count` is called, `_n_rows` will be `None` - hence, a count will need
        to be computed. If a `result` is passed, this implies that the underlying `data` has
        changed, meaning `_n_rows` is no longer accurate and `count` will need to be computed. If
        `_n_rows` is initialized (not `None`) and no change in `data` is detected, then `_n_rows` is
        simply retrieved and returned without the need for computing row count.

        Additionally, a handler layers the function to bypass retrieving the count. This can be
        controlled by the user when initializing a TidyDataFrame by passing the `toggle_count`
        parameter. (Contributed by Lida Zhang)
        """
        if not self.toggle_count:
            self._n_rows = self._unknown_dimension
        else:
            if self._n_rows is None:  # not yet defined, compute row count
                self._n_rows = self.data.count()
            if result is not None:  # result computed, recompute row count
                self._n_rows = result.count()
            return self._n_rows  # defined and no new result, return row count

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
                return (
                    f"selected {n_post:,} column(s) ({self._format_message(set(post))})"
                )

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

        n_rows = self.count()
        n_null_before = -1
        if colName in self.data.columns:
            n_null_before = (
                self.data.select(colName).filter(f.col(colName).isNull()).count()
                if self.toggle_count
                else None
            )
        result = self.data.withColumn(colName=colName, col=col)
        n_null_after = (
            result.select(colName).filter(f.col(colName).isNull()).count()
            if self.toggle_count
            else None
        )
        col_info = filterfalse(lambda t: t[0] != colName, result.dtypes)

        def get_message(
            total: int,
            invalid_before: int = -1,
            invalid_after: int = 0,
            col_info: tuple[str] = tuple(),
        ):
            col_name, col_type = col_info
            if not self.toggle_count:
                qualification = "no count performed"
            else:

                def n_invalid(n, total=total, when="before"):
                    if n == -1:
                        return ""
                    return f"({when}) {n:,} ({(n)/total:.2%}) values are NULL"

                qualification = f"{n_invalid(invalid_before)}{'' if invalid_before == -1 else ', '}{n_invalid(invalid_after, when='after')}"
            return f"created '{col_name}' ({col_type}): {qualification}"

        self._log_operation(
            operation="mutate",
            message=get_message(
                total=n_rows,
                invalid_before=n_null_before,
                invalid_after=n_null_after,
                col_info=tuple(col_info)[0],  # yield, unnest
            ),
        )
        return TidyDataFrame(
            result,
            toggle_count=self.toggle_count,
            toggle_display=self.toggle_display,
            n_rows=self.n_rows,
        )

    def withColumns(self, *colsMap):
        result = self.data.withColumns(*colsMap)
        self._log_operation(operation="mutate", message=f"message not yet created")
        return TidyDataFrame(
            result,
            toggle_count=self.toggle_count,
            toggle_display=self.toggle_display,
            n_rows=self.n_rows,
        )

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

    def dropna(self, how="any", thresh=None, subset=None):
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
        self._log_operation(operation="union", message="...original")
        return TidyDataFrame(
            result,
            toggle_count=self.toggle_count,
            toggle_display=self.toggle_display,
            n_rows=self.n_rows,
        )

    def unionAll(self, other):
        return self.union(other=other)

    def unionByName(self, other, allowMissingColumns=False):
        result = self.data.unionByName(
            other=other, allowMissingColumns=allowMissingColumns
        )
        self._log_operation(operation="union", message="...by name")
        return TidyDataFrame(
            result,
            toggle_count=self.toggle_count,
            toggle_display=self.toggle_display,
            n_rows=self.n_rows,
        )

    def __getattr__(self, name):
        return getattr(self.data, name)
