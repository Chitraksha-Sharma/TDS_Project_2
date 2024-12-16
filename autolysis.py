import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from fastapi import FastAPI



load_dotenv()
api_key = os.getenv("AZURE_API_KEY")
base_url = os.getenv("AZURE_API_BASE")
deployement_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
api_version = os.getenv("AZURE_API_VERSION")

model = AzureChatOpenAI(
    azure_deployment=deployement_name,
    api_version=api_version,
    api_key=api_key,
    azure_endpoint=base_url
)



def generic_analysis(df,file_path):
    """
    Perform generic analysis on the dataset.
    
    Args:
        file_path (str): Path to the dataset file (CSV format).
    
    Returns:
        dict: Summary statistics, missing values, correlations, outliers, and feature importance.
    """
    # Load dataset
    df = df
    
    # 1. Summary Statistics
    summary_stats = df.describe(include='all')
    
    # 2. Missing Values
    missing_values = df.isnull().sum()
    
    # 3. Correlation Matrix
    correlation_matrix = df.corr(numeric_only=True)
    
    # 4. Outlier Detection using Isolation Forest
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    numeric_df_index = numeric_df.index
    isolation_forest = IsolationForest(contamination=0.05, random_state=42)
    outlier_predictions = isolation_forest.fit_predict(numeric_df)
    # Create a column for outlier detection, initializing with 0 (non-outlier)
    df['Outlier'] = 0
    # Assign predictions to rows with valid numeric data
    df.loc[numeric_df_index, 'Outlier'] = outlier_predictions
    outliers = df[df['Outlier'] == -1]

    if len(numeric_df.columns) > 1:
        try:
            # Ensure there are enough rows for clustering
            if len(numeric_df) >= 3:  # Minimum rows needed for 3 clusters
                kmeans = KMeans(n_clusters=3, random_state=42).fit(numeric_df)
            
            # Initialize cluster column in the original DataFrame
                df['Cluster'] = np.nan
                
                # Assign cluster labels to rows that were used in clustering
                df.loc[numeric_df.index, 'Cluster'] = kmeans.labels_
                
                # print("\nClusters Assigned (KMeans):")
                # print(df['Cluster'].value_counts())
            else:
                print("\nNot enough data points for clustering (minimum 3 rows).")
        except NotFittedError as e:
            print("\nError fitting KMeans:", e)
    else:
        print("\nClustering skipped: not enough numeric columns (minimum 2).")
    
    # 5. Feature Importance (using Random Forest for numeric columns)
    feature_importance = None
    if len(df.select_dtypes(include=np.number).columns) > 1:
        X = df.select_dtypes(include=np.number).dropna()
        y = np.random.choice([0, 1], size=X.shape[0])  # Dummy target for feature ranking
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X, y)
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

    
    directory = os.path.dirname(file_path)
    filename = os.path.basename(file_path)
    column_names = list(df.columns)
    column_types = list(df.dtypes)
    return {
        'filename' : file_path,
        'example_rows' : df.head(3).to_dict(),
        'column_data' : list(zip(column_names, column_types)),
        'summary_stats': summary_stats,
        'missing_values': missing_values,
        'correlation_matrix': correlation_matrix,

        'feature_importance': feature_importance
    }

def infrance_llm_for_code(filename, example_rows, column_data, summary_stats, missing_values,corr_matrix,feature_importance):
    system_template = """ 
    Consider yourself a highly skilled Data analyst with a deep understanding of data analysis and machine learning and python programming. 
    Your task is to find insights from the dataset and provide python code to perform the analysis.
    I will provide you with a dataset named {filename} with the following columns and types:
    {column_data}
        

        Summary Statistics:
        {summary_stats}

        Example Rows:
        {example_rows}

        Missing Values:
        {missing_values}

        Correlation Insights:
        {corr_matrix}

        Feature Importance:
        {feature_importance}    

        Analyze the dataset. 
        Provide python code to perform the analysis.provide the code in json format as code:{{code}}.
"""

    prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{filename}, {column_data}, {summary_stats}, {example_rows}, {missing_values}, {corr_matrix}, {feature_importance}")]
    )
    prompt = prompt_template.invoke({"filename": filename, "column_data": column_data, "summary_stats": summary_stats, "example_rows": example_rows, "missing_values": missing_values, "corr_matrix": corr_matrix, "feature_importance": feature_importance})
    prompt.to_messages()
    try:    
        response = model.invoke(prompt)
        print(f"Response: {response}")
        return response
    except Exception as e:
        print(f"Error invoking model: {e}")
        return None
    

def execute_code_from_string(code_string):
    # Create a temporary Python file
    temp_file = "temp_analysis.py"
    
    # Write the code string to the file
    with open(temp_file, "w") as f:
        f.write(code_string)
    
    # Execute the file
    try:
        exec(code_string)
    except Exception as e:
        print(f"Error executing code: {str(e)}")
    finally:
        # Clean up - remove the temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)


def create_visualizations(analysis_results):
    """
    Generate visualizations for the analysis results.
    
    Args:
        analysis_results (dict): Results of the generic analysis.

    """
    #
    file_path = analysis_results['filename']
    correlation_matrix = analysis_results['correlation_matrix']
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.savefig(file_path.replace(".csv", "_correlation_matrix.png"))
    plt.close()
    # Visualize missing values
    analysis_results['missing_values'].plot(kind='bar', title="Missing Values Count")
    plt.savefig(file_path.replace(".csv", "_missing_values.png"))
    plt.close()
    
    # Visualize feature importance (if available)
    if analysis_results['feature_importance'] is not None:
        sns.barplot(
            data=analysis_results['feature_importance'], 
            x="Importance", 
            y="Feature"
        )
        plt.title("Feature Importance")
        plt.savefig(file_path.replace(".csv", "_feature_importance.png"))
        plt.close()

def get_insights_and_implications(filename, example_rows, column_data, summary_stats, missing_values,corr_matrix,feature_importance):
    system_template = """
    You are an expert to understand the dataset and provide insights and implications.
    I will provide you with a dataset named {filename} with the following columns and types:
    {column_data}
    
    Example Rows:
    {example_rows}
    
    Summary Statistics:
    {summary_stats}
    
    Missing Values:
    {missing_values}
    
    Correlation Insights:
    {corr_matrix}
    
    Feature Importance:
    {feature_importance}

    Based on the above, provide:
    - Key insights about the dataset.
    - Actionable implications based on the insights.
    """
    prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{filename}, {column_data}, {summary_stats}, {example_rows}, {missing_values}, {corr_matrix}, {feature_importance}")]
    )
    prompt = prompt_template.invoke({"filename": filename, "column_data": column_data, "summary_stats": summary_stats, "example_rows": example_rows, "missing_values": missing_values, "corr_matrix": corr_matrix, "feature_importance": feature_importance})
    prompt.to_messages()
    try:    
        response = model.invoke(prompt)
        print(f"Response: {response.content}")
        return response.content
    except Exception as e:
        print(f"Error invoking model: {e}")
        return None
    
def generate_story_and_save(directory,filename,column_data,  summary_stats, insights):
    
    """
    Use the LLM to generate a story from the analysis and save it as README.md.
    
    Args:
        file_path (str): Path to the dataset.
        analysis_results (dict): Results of the generic analysis.
        insights (str): Insights derived from the analysis.
        implications (str): Implications of the findings.
    """
    # Prepare the summary for the LLM
    system_template = f"""
    Write a README.md file summarizing the analysis of the dataset.
    I will provide you with a dataset named {filename} with the following columns and types:
    
    column_data:
    {column_data}
    
    1. Dataset Overview: The dataset contains the following structure:
    {summary_stats}
    
    2. Analysis Performed: We analyzed missing values, correlations, outliers, and feature importance.
    
    3. Key Insights and Implications: {insights}
    
    """
        # directory = os.path.dirname(file_path)
        # filename = os.path.basename(file_path)
    prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{filename}, {column_data}, {summary_stats},  {insights}")]
    )
    prompt = prompt_template.invoke({"filename": filename, "column_data": column_data, "summary_stats": summary_stats, "insights": insights})
    prompt.to_messages()
    try:    
        llm_story = model.invoke(prompt)
        print(f"Response: {llm_story}")
        readme_path = os.path.join(directory, "README.md")
        with open(readme_path, "w") as file:
            file.write(llm_story.content)
    except Exception as e:
        print(f"Error invoking model: {e}")
        return None
    



def main():
    # Step 1: Parse command-line arguments
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)

    # Get the dataset filename from the command-line arguments
    csv_file = sys.argv[1]
    file_path = csv_file
    file_name = os.path.basename(csv_file)
    directory = os.path.dirname(csv_file)

    # Step 2: Validate the file
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found.")
        sys.exit(1)
    if not csv_file.endswith(".csv"):
        print("Error: Only CSV files are supported.")
        sys.exit(1)

    # Step 3: Load the dataset
    try:
        df = pd.read_csv(csv_file,encoding='latin-1')
    except Exception as e:
        print(f"Error: Unable to read the CSV file. {e}")
        sys.exit(1)
    
    
    # Step 4: Perform Generic Analysis
    print("\nPerforming Generic Analysis...")
    analysis_results = generic_analysis(df,csv_file)

    try:
        filename = analysis_results['filename']
        example_rows = analysis_results['example_rows']
        column_data = analysis_results['column_data']
        summary_stats = analysis_results['summary_stats']
        missing_values = analysis_results['missing_values']
        corr_matrix = analysis_results['correlation_matrix']
        feature_importance = analysis_results['feature_importance']
    except Exception as e:
        print(f"Error parsing analysis results: {str(e)}")
        sys.exit(1)
        
    
    # Step 5: Use LLM to Analyze Data
    print("\nUsing LLM to Analyze the Data...")
    llm_analysis = infrance_llm_for_code(filename, example_rows, column_data, summary_stats, missing_values,corr_matrix,feature_importance)

    # Step 6: Parse the JSON output
    parsed_data = None
    analysis_code = None
    try:
        parser = JsonOutputParser()
        parsed_data = parser.parse(llm_analysis.content)
        analysis_code = parsed_data.get("code")
       
    except Exception as e:
        print(f"Error parsing json: {str(e)}")

    if analysis_code:
        try:
            execute_code_from_string(analysis_code)
        except Exception as e:
            print(f"Error executing code: {str(e)}")
    
    
    # Step 6: Visualize Results
    print("\nCreating Visualizations...")
    visualization_outputs = create_visualizations(analysis_results)
    
    # Step 7: Get Insights and Implications
    print("\nGenerating Insights and Implications...")
    insights_and_implications = get_insights_and_implications(filename, example_rows, column_data, summary_stats, missing_values,corr_matrix,feature_importance)

    
    # Step 8: Generate a Story and Save as README.md
    print("\nWriting the Story (README.md)...")
    generate_story_and_save(directory,filename,column_data, summary_stats, insights_and_implications)
    print("\nAnalysis Completed! The results have been saved.")
    print("- Visualizations: Saved in current directory.")
    print("- Story: README.md generated.")
    print("- Use the LLM insights for further interpretations.")

if __name__ == "__main__":
    main()