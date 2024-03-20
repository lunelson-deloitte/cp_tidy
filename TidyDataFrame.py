import functools
import itertools

from attrs import define, field, validators
from loguru import logger

import pyspark


@define
class TidyDataFrame:
    """Decorator class enhancing data pipelines with in-process logging messages

    The TidyDataFrame extends the native pyspark.sql.DataFrame (DataFrame) by
    giving users immediate feedback as their code executes. Depending on the
    nature of a command, users can observe how a method alters their DataFrame's
    dimensions, schema, and more.

    TidyDataFrame offers an array of helpful methods out of the box with minimal
    changes to any existing code. Once wrapped around a DataFrame, it can "toggle"
    many common options, such as counting, displaying, timing, messaging, and so on.
    """

    _data: pyspark.sql.DataFrame = field(
        validator=validators.instance_of(pyspark.sql.DataFrame)
    )
    toggle_options: dict[str, bool] = field(factory=dict)
    _n_rows: int = field(default=None)
    _n_cols: int = field(default=None)

    def __attrs_post_init__(self):
        """Coerce initialization to normalized state, with greeting message"""
        self.toggle_options.setdefault("count", True)
        self.toggle_options.setdefault("display", True)
        self._n_rows = (
            self._data.count()
            if self.toggle_options.get("count")
            else self._unknown_dimension
        )
        self._n_cols = len(self._data.columns)
        self._log_operation(
            ">> enter >>", self.__repr__(data_type=type(self).__name__), level="success"
        )

    def __repr__(self, data_type: str):
        """String representation of TidyDataFrame"""
        n_rows_repr = (
            f"{self._n_rows:,}" if isinstance(self._n_rows, int) else self._n_rows
        )
        data_repr = f"{data_type}[{n_rows_repr} rows x {self._n_cols:,} cols]"
        disabled_options_repr = ""
        if data_type == "TidyDataFrame":
            disabled_options = itertools.compress(
                self.toggle_options.keys(),
                map(lambda x: not x, self.toggle_options.values()),
            )
            options_string = ", ".join(disabled_options)
            disabled_options_repr = (
                f"(disabled: {options_string})" if options_string != "" else ""
            )
        return f"{data_repr} {disabled_options_repr}"

    def _log_operation(self, operation, message, level="info"):
        # consider alias for users; maybe .comment()?
        getattr(logger, level)(f"#> {operation}: {message}")
        return self

    def _tdf_controller(message: str, alias: str = None):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                if hasattr(self, func.__name__):
                    result = func(self, *args, **kwargs)
                    self._n_cols = len(result._data.columns)
                    if not kwargs.get("disable_message", False):
                        self._log_operation(
                            operation=func.__name__ if alias is None else alias,
                            message=eval(f"f'{message}'"),
                        )
                    return result

            return wrapper

        return decorator

    @property
    def data(self):
        self._log_operation(
            "<< exit <<",
            self.__repr__(data_type=type(self._data).__name__),
            level="success",
        )
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
    def _unknown_dimension(self):
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
        if not self.toggle_options.get("display"):
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
        if not self.toggle_options.get("count"):
            self._n_rows = self._unknown_dimension
            return 0
        else:
            if self._n_rows == self._unknown_dimension:  # not defined, compute
                self._n_rows = self._data.count()
            if result is not None:  # result computed, recompute row count
                self._n_rows = result._data.count()
            return self._n_rows  # defined and no new result, return row count

    ### FILTERING OPERATIONS
    @_tdf_controller(
        message="contains {self.count():,} rows, removed {self.count() - self.count(result):,} rows, remaining {self.count():,} rows"
    )
    def filter(self, condition, disable_message: bool = False):
        self._data = self._data.filter(condition)
        return self

    def where(self, condition, disable_message: bool = False):
        return self.filter(condition)

    @_tdf_controller(
        message="removed {self.count() - self.count(result):,} duplicates",
        # alias="filter_dups"
    )
    def drop_duplicates(self, subset=None, disable_message: bool = False):
        self._data = self._data.drop_duplicates(subset=subset)
        return self

    def dropDuplicates(self, subset=None, disable_message: bool = False):
        return self.drop_duplicates(subset=subset)

    @_tdf_controller(
        message="removed {self.count() - self.count(result):,} NAs",
        # alias="filter_na"
    )
    def dropna(
        self, how="any", thresh=None, subset=None, disable_message: bool = False
    ):
        self._data = self._data.dropna(how=how, thresh=thresh, subset=subset)
        return self

    @_tdf_controller(
        message="removed {self.count() - self.count(result):,} duplicate rows"
    )
    def distinct(self, disable_message: bool = False):
        self._data = self._data.distinct()
        return self

    ### COLUMN SELECTING OPERATIONS
    @_tdf_controller(message="selected {self._n_cols} columns")
    def select(self, *cols, disable_message: bool = False):
        self._data = self._data.select(*cols)
        return self

    def drop(self, cols, disable_message: bool = False):
        all_cols = self._data.columns
        drop_cols = set(all_cols).difference(set(cols))
        return self.select(*drop_cols)

    ### JOIN OPERATIONS
    @_tdf_controller(message="appended {self.count(result):,} rows")
    def union(self, other, disable_message: bool = False):
        self._data = self._data.union(other)
        return self

    def unionAll(self, other, disable_message: bool = False):
        return self.union(other)

    @_tdf_controller(
        message="appended {(self.count() - self.count(result)) * -1:,} rows"
    )
    def unionByName(
        self, other, allowMissingColumns=False, disable_message: bool = False
    ):
        self._data = self._data.unionByName(
            other, allowMissingColumns=allowMissingColumns
        )
        return self

    @_tdf_controller(
        message="matched {(self.count() - self.count(result)) * -1:,} rows"
    )
    def join(self, other, on=None, how="inner", disable_message: bool = False):
        self._data = self._data.join(other=other, on=on, how=how)
        return self

    ### COLUMN EDITING OPERATIONS
    @_tdf_controller(
        message='created `{args[0] if args else kwargs.get("colName")}` (< type >)',  # update to "created" or "edited"?
        alias="mutate",
    )
    def withColumn(self, colName, col, disable_message: bool = False):
        self._data = self._data.withColumn(colName=colName, col=col)
        return self

    @_tdf_controller(
        message="creating multiple columns",
        alias="rename",
    )
    def withColumns(self, *colsMap, disable_message: bool = False):
        self._data = self._data.withColumns(*colsMap)
        return self

    @_tdf_controller(
        message='column `{args[0] if args else kwargs.get("existing")}` renamed to `{args[1] if args else kwargs.get("new")}`',
        alias="rename",
    )
    def withColumnRenamed(self, existing, new, disable_message: bool = False):
        self._data = self._data.withColumnRenamed(existing=existing, new=new)
        return self

    ### CATCH ALL OPERATION
    def __getattr__(self, attr):
        if hasattr(self._data, attr):

            def wrapper(*args, **kwargs):
                result = getattr(self._data, attr)(*args, **kwargs)
                if isinstance(result, pyspark.sql.DataFrame):
                    self._data = result
                    self._log_operation(
                        operation=attr, message="not yet implemented", level="warning"
                    )
                    return self
                else:
                    return self

            return wrapper
        ### TODO: validate if this logging operation is legit
        ### TODO: mark as unstable (sometimes get notebook dependencies caught in this; generates long message)
        # self._log_operation(operation=attr, message="method does not exist", level="error")
        raise AttributeError(
            f"'{type(self._data).__name__}' object has no attribute '{attr}'"
        )
