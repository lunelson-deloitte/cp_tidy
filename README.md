# cp_tidy

The `TidyDataFrame` is a decorated class variant of the `pyspark.sql.DataFrame`. When enabled, TidyDataFrame generates logging statements in real time, tracking transformations made to a DataFrame as they occur. Additional features can be enabled/disabled depending on the user's preferences, including timing commands, displaying datasets, and controlling messages.


## About the TidyDataFrame

This class was heavily inspired by the work done by [Benjamin Elbers](https://github.com/elbersb) in their [tidylog](https://github.com/elbersb/tidylog) project. By just loading the package, any project making use of conventional tidyverse functions can benefit from in-process, extremely helpful logging statements. If interested, I highly encourage checking out the project to learn more about it.

The `TidyDataFrame` aims to mimic `tidylog` as much as possible:

- Decorate existing functionality to enhance data processing

- Develop minimal code and generate minimal overhead

- Offer a solution that users can simply "plug in" to their existing code


## Implementing the TidyDataFrame

Suppose you are working with the following example:

```{python}
import pyspark

df = (
    df
    .select(...)
    .filter(...)
    .withColumns(...)
)
```

If installed correctly, one could use `TidyDataFrame` to receive the following outputs with minimal changes to their code:

```{python}
import pyspark
from tidydataframe import TidyDataFrame

df = (
    TidyDataFrame(df)
    .select(...)
    .filter(...)
    .withColumn(...)
)

#> select: kept X columns
#> filter: removed X rows (%), N - X remaining
#> mutate: created column (type), X null values
```

The behavior of `TidyDataFrame` can be controlled in initialization:

```{python}
import pyspark
from tidydataframe import TidyDataFrame

df = (
    TidyDataFrame(df, toggle_count=False, toggle_timer=True)
    .select(...)
    .filter(...)
    .withColumns(...)
)

#> select (0.05s): kept X columns
#> filter (0.78s): count not performed
#> mutate (0.13s): created column (type), count not performed
```


## Considerations

Given this class decorates a `pyspark.sql.DataFrame`, this class is meant to run in a distributed computing environment. If operating on large datasets (greater than 200M rows), `TidyDataFrame` could be an unwanted performance bottleneck.

If looking for ways to implement `TidyDataFrame` without introducing needless overhead, consider the following:

- Run on reasonable samples of your DataFrame. The less data, the better. As long as you believe `TidyDataFrame`'s messages can be scaled back to the full size of your data, we believe no *approximate truth* is lost.

- Consider your cluster configurations. If counting operations are enabled (`TidyDataFrame(..., toggle_count=True)`, the default), collecting these results will take some time. Although not possible for most, reconsider how many workers are required for your workload.

We understand some of these considerations aren't entirely in the user's control (nor is it in ours). Where possible, we try to address these possibilities in development and feedback. If you believe there are areas we have not yet addressed or improvements we could make to minimize the need for any consideration, please contribute using the guidelines below.


## Development Team

This class was originally developed by Lucas Nelson, with the help of Anusha Venkata Jami, Phung Pham, and Lida Zhang.


## Contribution Guidliness

Please create a pull request.
