# coding: utf-8

# # Pima Indian Diabetes Prediction

import pandas as pd                 # pandas is a dataframe library
import matplotlib.pyplot as plt      # matplotlib.pyplot plots data



# ## Loading and Reviewing the Data

df = pd.read_csv("./pima-data.csv")


df.shape

df.head(5)

df.tail(5) 

#  Check for null values

df.isnull().values.any()


# ### Correlated Feature Check

#  The Function that displays correlation by color.  Red is most correlated, Blue least.


def plot_corr(df, size=11):
    """
    Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot

    Displays:
        matrix of correlation between columns.  Blue-cyan-yellow-red-darkred => less to more correlated
                                                0 ------------------>  1
                                                Expect a darkred line running from top left to bottom right
    """

    corr = df.corr()    # data frame correlation function
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)   # color code the rectangles by correlation value
    plt.xticks(range(len(corr.columns)), corr.columns)  # draw x tick marks
    plt.yticks(range(len(corr.columns)), corr.columns)  # draw y tick marks


plot_corr(df)



df.corr()




df.head(5)


# The skin and thickness columns are correlated 1 to 1.  Dropping the skin column

del df['skin']


df.head(5)


# Check for additional correlations

# In[13]:

plot_corr(df)


# The correlations look good.  There appear to be no coorelated columns.

 
# Inspect data types to see if there are any issues.  Data should be numeric.


df.head(5)


# Change diabetes from boolean to integer, True=1, False=0

diabetes_map = {True : 1, False : 0}
df['diabetes'] = df['diabetes'].map(diabetes_map)


# Verify that the diabetes data type has been changed.


df.head(5)
