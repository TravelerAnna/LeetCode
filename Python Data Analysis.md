# Python Data Analysis



## Pandas Method

### .shape

df.shape: return arrays of the number of rows and the number of columns. 





### .loc[ , ]

==**Slice the data by exact column or row index. **==

==**Not copy, just locate some values, do not change the original dataframe, but can change the value**==

```python
students.loc[ students['Name'] == 'Anna', ['X', 'Y']]
```





### .drop_duplicates(subset=[ ])

==drop duplicates based on specific columns, use subset = ['a']==

axis = 0 (row, top to bottom ) / axis = 1 (column, left to right )

inplace = True



### .dropna( )

subset = ['name']



### .columns

rename columns : 

​	df.columns = ['student_id', 'first_name', 'last_name', 'age_in_years']

rename columns specifically: 

​	**df.rename( columns = {'id' : 'student_id'} )**



### .astype( )

change type : **df.astype( dtype = {"grade" : int } )**



### .fillna( )

fill the missing value with 0 : **df.fillna( value = {'quantity' : 0} )**



### .concat( )

concatenate tables : **pd.concat( [df1, df2])**

axis = 0 / axis = 1



### .sort_values(by=)

- **by = [' ']** : sort by specified column

- **ascending = False** : descending order





### .pivot( )

pivot a table, specify index, column and values.

**df.pivot( index='month', columns='city', values='temperature')**

pd.pivot(data=weather, columns="city", index="month", values="temperature")



### .melt( )

**id_vars**：invariable column index, can be more than one.

**value_vars**：columns need to be melt

**var_name**：new columns name after melt

**value_name**：new values name after name

eg. df.melt(  id_vars = 'product',

​        value_vars = ['quarter_1', 'quarter_2', 'quarter_3', 'quarter_4'],

​        var_name = 'quarter',

​        value_name = 'sales' )



### .merge( )

df.merge( ) / pd.merge( df1, df2, left_on=' ', right_on=' ', how=' ')

The `how` parameter can be set to different types of joins:

- `'inner'`: Only include rows with keys present in **both** DataFrames.
- `'left'`: **Include all rows from the left DataFrame**, and matching rows from the right DataFrame.
- `'right'`: **Include all rows from the right DataFrame**, and matching rows from the left DataFrame.
- `'outer'`: **Include all rows from both DataFrames**, filling in `NaN` where there are no matches.



### .isnull( )

determine whether it is null.



### .rename( )

==DataFrame.rename(mapper=None, index=None, columns=None, axis=None, copy=True, inplace=False, level=None, errors='ignore')==



- **mapper**: dict-like or function, optional. A mapping of labels to new labels.
- **index**: dict-like or function, optional. A mapping of index labels to new labels.
- **columns**: dict-like or function, optional. A mapping of column labels to new labels.
- **axis**: int or str, optional. Axis to target with mapper. Can be either 0/'index' or 1/'columns'.
- **copy**: bool, default True. Also copy underlying data.
- **inplace**: bool, default False. Whether to return a new DataFrame or modify the existing one.
- **level**: int or level name, default None. In case of a MultiIndex, only rename labels in the specified level.
- **errors**: {'ignore', 'raise'}, default 'ignore'. Control raising of exceptions on invalid data.

```python
df = df.rename(index={0: 'first', 1: 'second', 2: 'third'})
df = df.rename(columns={'A': 'Alpha', 'B': 'Beta'})
df = df.rename(columns=str.lower)
```





### len( )

- For strings, `len()` returns the number of characters.
- For lists, tuples, and sets, `len()` returns the number of elements.
- For dictionaries, `len()` returns the number of key-value pairs.

```python
s = "Hello, World!"
length = len(s)
print(length)  # Output: 13

d = {'a': 1, 'b': 2, 'c': 3}
length = len(d)
print(length)  # Output: 3
```





### **.str**.**startswith**()

==**prefix, start, end**==

check if a string starts with a specified prefix. It returns `True` if the string starts with the specified prefix, otherwise it returns `False`.

```python
text = "Hello, world!"
result = text.startswith("Hello")
print(result)  # Output: True

result = text.startswith("world", 7)
print(result)  # Output: True

result = text.startswith("world", 0, 5)
print(result)  # Output: False
```





### ~ operator

- bitwise NOT operator, which inverts all the bits of its operand. This operator is used primarily with integers.

- The `~` operator inverts the boolean values, turning `True` into `False` and `False` into `True`.

```python
	~( employees['name'].str.startswith('M') )
```







## Pandas Skills

### Return Dataframe or series

```python
views.loc[views['author_id'] == views['viewer_id'], ['author_id']]

views.loc[views['author_id'] == views['viewer_id'], 'author_id']

World['area']  /  world [ ['name', 'population', 'area'] ]
```

**1. The first one returns a data frame, and the second one returns a series.**

**2. The first one returns a series, and the second one returns a data frame.**





### Filter by condition

`World[ (world['area'] >= 0) | (world['population'] >= 0) ]` 

**only one square parenthesis / return dataframe**





### Select one or multiple columns

`World['area']  /  world [ ['name', 'population', 'area'] ]` 

**if multiple, two square parenthesis.**





### Combine filter by condition and select columns

```python
products.loc[ (products['low_fats']=='Y') & (products['recyclable']=='Y'), ['product_id']]
```

Use loc for true index name.





### Calculate str length(value / column)

If you want to compute the length of a single value, convert it to a string first and len( ).

The `str.len()` method is used to compute the length of each element in a Series or Index containing strings.

```python
number = 28233
length = len(str(number))

series = pd.Series(['028233', '12345', '6789'])
lengths = series.str.len()
```











## Pandas Exercise

### [183. Customers Who Never Order](https://leetcode.cn/problems/customers-who-never-order/)

==If different columns in different tables need to be compared for filtering, and they are the same meaning, use merge to combine.==

```python
def find_customers(customers: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(customers, orders, left_on='id', right_on ='customerId', how = 'left')
    df = df[ df['customerId'].isnull() ][['name']]
    df = df.rename(columns = {'name': 'Customers'})
    return df
```





### [1873. Calculate Special Bonus](https://leetcode.cn/problems/calculate-special-bonus/)



