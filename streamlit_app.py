# python 3.11.9
# just your imports
import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
# from scipy.stats import zscore
import numpy as np
 
# Style config
st.markdown(
    f"""
    <style>
    body {{
        background-color: #e2e1dc;
        color: #000000;
        font-family: "Serif";
    }}
    .stApp {{
        background-color: #e2e1dc;
    }}
    .stButton>button {{
        background-color: #f38126;
        color: white;
        font-family: "Serif";
    }}
    .stSidebar {{
        background-color: #bcd6f5;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit App Title 
st.title("Simple Data Visualization App") 
 
# File uploader 
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"]) 
 
if uploaded_file is not None: 
    # Read the CSV file 
    df = pd.read_csv(uploaded_file) 

    "_____"

    st.write("# **Data Profile**") 

    # Show the first few rows 
    st.write("### Preview of the Data") 
    st.dataframe(df.head()) 

    # Show short summary statistics 
    st.write("### Short Summary of Statistics") 
    st.write(df.describe())

    # Show  Full summary statistics 
    st.write("### Full Summary of Statistics") 
    st.write(df.describe(include='all'))

    # Show shape 
    st.write("### Shape of the Data") 
    st.info(f'There is {df.shape[0]} row(s) and {df.shape[1]} feature(s) in the Data') 

    #Feature Types
    st.write('### Feature Data Types')
       
    o,f,i,d = st.columns(4)
    with o:
        st.write('##### Object Features')
        st.dataframe(df.select_dtypes(include=["object"]).columns.tolist())
    with f:
        st.write('##### Float Features')
        st.dataframe(df.select_dtypes(include=["float64"]).columns.tolist())
    with i:
        st.write('##### Integer Features')
        st.dataframe(df.select_dtypes(include=["int64"]).columns.tolist())
    with d:
        st.write('##### Datetime Features')
        st.dataframe(df.select_dtypes(include=["datetime64[ns]"]).columns.tolist())
    
    st.write("##### Feature Data Type Counts")
    st.dataframe(df.dtypes.value_counts())
    
    "______"

    st.write("# **Data Values**") 
    left, right= st.columns(2)
    with right:
        #Show Null values
        st.write('### Null Values')
        st.dataframe(df.isnull().sum().sort_values(ascending=False)[df.isnull().sum() > 0])
    with left:
        #Show value Counts 
        st.write("### Feature value counts") 
        columns = st.selectbox("Select X-axis column", df.columns)
        st.dataframe(df[columns].value_counts(ascending=False, dropna=False))  
    with right:
        #Show Number of duplicate Rows
        st.write('### Duplicate Rows')
        st.write('Number of Duplicates signfied by checked box.')
        st.dataframe(df.duplicated().value_counts(dropna=False))

    #filter for a value
    st.write('### Filter for a value in a selected feature')
    col = st.selectbox('Select feature', df.columns)
    df_null = df[df[col].isnull()]
    value = st.selectbox('Select value', df[col].value_counts(dropna=False).sort_index(ascending=False).index.tolist())
    if pd.isnull(value):
        st.dataframe(df_null)
    else:
        st.dataframe(df[df[col] == value])
    

    "______"

    st.write("# **Data Visualization**") 

    # Select columns for visualization 
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist() 
    number_df = df.select_dtypes(include=["number"]).dropna()

    if len(numeric_columns) == 0: 
        st.warning("No numeric columns found in the dataset.")
    
    if len(numeric_columns) >= 1: 
        st.write('### Boxplot')
        boxplot_col = st.selectbox('Select feature for Boxplot',numeric_columns)

        fig,ax = plt.subplots()
        sns.boxplot(df[boxplot_col].dropna(),ax=ax)
        st.write(f'##### Boxplot of {boxplot_col}')
        st.pyplot(fig)

        st.write('##### *Percentiles of Boxplot*')
        percentiles = np.percentile(df[boxplot_col].dropna(),[25,50,75])
        perc1, perc2,perc3 = st.columns(3)
        with perc1:
            st.write(f"**25th percentile:** {percentiles[0]}")
        with perc2:
            st.write(f"**50th percentile (median):** {percentiles[1]}")
        with perc3:
            st.write(f"**75th percentile:** {percentiles[2]}")

        # st. write('##### *Statistical Outliers*')
        # stand_v = df[boxplot_col].apply(zscore)
        # stand_v = pd.concat([stand_v,df.index],axis=1)
        # outliers = (stand_v > 3)
        # outlier_indices = df.index[outliers].tolist()
        # st.dataframe(outliers)
        # st.write('###### ***Outlier Indices***')
        # st.dataframe(outlier_indices)

        "_________"

    if len(numeric_columns) >= 1: 
        st.write("### Histogram") 
        colhist, colbin = st.columns(2) 
        with colhist: 
            hist_col = st.selectbox("Select column for histogram", numeric_columns) 
        with colbin:
            bin = st.selectbox('Select number of bins for histogram', range(10,110,10))
         
        fig, ax = plt.subplots() 
        sns.histplot(df[hist_col].dropna(), bins=bin, kde=True, ax=ax) 
        st.write(f'##### Histogram of {hist_col}')
        st.pyplot(fig) 

        "_________"

    if len(numeric_columns) >= 2: 
        st.write("### Scatter Plot") 
        left_column, right_column = st.columns(2) 
        with left_column: 
            x_col = st.selectbox("Select X-axis column", numeric_columns) 
        with right_column: 
             y_col = st.selectbox("Select Y-axis column", numeric_columns) 
 
        fig, ax = plt.subplots() 
        sns.scatterplot(data=df.dropna(), x=x_col, y=y_col, ax=ax) 
        st.write(f'##### Scatter plot of {x_col} vs {y_col}')
        st.pyplot(fig)  
    
    if len(numeric_columns) >= 2:
        st.write('### Correlation Heatmap')
        lef,righ = st.columns(2)
        with lef:
            heat_col=st.selectbox("Select Target Feature", numeric_columns)
        with righ:
            meth = st.radio('Select Corolation Method', ['pearson','spearman','kendall'])
        thresh = st.selectbox('Select Correlation Threshhold', np.round(np.arange(.0,1.1,.1),1))
        corr = number_df.dropna(subset=[heat_col]).corr(method=meth)
        corr2 = corr[corr > thresh]
        sns.heatmap(corr2)
        st.write('#### ***Features and Correlation Values***',)
        filter_corr2 = corr2[heat_col].sort_values(ascending=False)
        st.dataframe(filter_corr2[filter_corr2 >= thresh])
        st.write(f'##### Heatmap of {heat_col}')
        st.pyplot(fig)
else: 
    st.warning("Please upload a CSV file to proceed.")

    
    
    #zscore
    # account for instances where there may not be a condition in the data set to satisfy the code