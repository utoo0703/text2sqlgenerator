import streamlit as st
import os
import logging
import requests
from dotenv import load_dotenv
from agents import router_agent
import pandas as pd
import json
import io
import plotly.express as px
from io import StringIO
import seaborn as sns
import matplotlib.pyplot as plt

# Session state initialization
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'show_schema' not in st.session_state:
    st.session_state.show_schema = False
if 'process_input' not in st.session_state:
    st.session_state.process_input = False
if 'clear_input' not in st.session_state:
    st.session_state.clear_input = False

# Custom Streamlit logging handler
class StreamlitHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        if 'log_data' not in st.session_state:
            st.session_state.log_data = io.StringIO()
        self.log_data = st.session_state.log_data
    
    def emit(self, record):
        msg = self.format(record)
        self.log_data.write(msg + '\n')

streamlit_handler = StreamlitHandler()
streamlit_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
streamlit_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
logger.addHandler(streamlit_handler)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(page_title="ENTERPRISE DATA ASSISTANT", layout="wide")

# Database paths
available_databases = {
    "Enterprise Loan Database": os.getenv('SQLITE_DB_PATH'),
    "Enterprise Computation Database": r"C:path", #Specify the Secondary database path
}

# Function definitions
def process_sql_result(data_json):
    return pd.read_json(StringIO(data_json), orient='records')

def create_visualization(df, chart_type, x_col, y_col=None):
    if df.empty:
        return None, "The dataset is empty. No visualization can be created."
    if x_col not in df.columns or (y_col and y_col not in df.columns):
        return None, f"Selected column(s) not found in the dataset or not suitable for Visualization."
    try:
        if chart_type == "Bar":
            fig = px.bar(df, x=x_col, y=y_col)
        elif chart_type == "Line":
            fig = px.line(df, x=x_col, y=y_col)
        elif chart_type == "Scatter":
            fig = px.scatter(df, x=x_col, y=y_col)
        elif chart_type == "Histogram":
            fig = px.histogram(df, x=x_col)
        elif chart_type == "Box":
            fig = px.box(df, y=x_col)
        elif chart_type == "Violin":
            fig = px.violin(df, y=x_col)
        elif chart_type == "Distribution":
            fig = px.histogram(df, x=x_col, marginal="box")
        else:
            return None, f"Unsupported chart type: {chart_type}"
        
        fig.update_layout(title=f"{chart_type} Chart of {x_col}" + (f" vs {y_col}" if y_col else ""))
        return fig, None
    except Exception as e:
        return None, f"Error creating visualization: {str(e)}"

def auto_select_chart(df):
    if df.empty:
        return None, None, None

    num_columns = df.select_dtypes(include=['int64', 'float64']).columns
    cat_columns = df.select_dtypes(include=['object', 'category']).columns

    if len(num_columns) == 2:
        return "Scatter", num_columns[0], num_columns[1]
    elif len(num_columns) == 1 and len(cat_columns) == 1:
        return "Bar", cat_columns[0], num_columns[0]
    elif len(cat_columns) == 1:
        return "Bar", cat_columns[0], None
    elif len(num_columns) >= 1:
        return "Histogram", num_columns[0], None
    else:
        return None, None, None

def export_chat_history():
    return json.dumps(st.session_state.messages, indent=2)

def on_input_change():
    user_input = st.session_state.user_input
    if user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.process_input = True  # Flag to process input
        st.session_state.clear_input = True  # Flag to clear input

# Layout setup
st.title("ENTERPRISE DATA ASSISTANT")

# Input area
if st.session_state.clear_input:
    st.session_state.user_input = ""
    st.session_state.clear_input = False

user_input = st.text_input("Type your message here...", key="user_input")
col1, col2, col3 = st.columns([1, 1, 5])
send_button = col1.button("Send", on_click=on_input_change)
clear_button = col2.button("Clear Chat")

if clear_button:
    st.session_state.messages = []
    st.session_state.clear_input = True
    st.rerun()

if st.session_state.get('process_input', False):
    user_input = st.session_state.messages[-1]["content"]
    if user_input.strip():  # Only process if input is not empty
        with st.spinner("Processing your query..."):
            try:
                response = router_agent.interact(user_input)
                logger.info(f"Agent response: {response}")
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_message = f"Error: {str(e)}"
                logger.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
            st.session_state.process_input = False
            st.rerun()

chat_container = st.container()

with chat_container:
    for idx, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            st.markdown(f'**User:** {message["content"]}', unsafe_allow_html=True)
        else:
            content = message["content"]
            st.markdown('**ASSISTANT:**', unsafe_allow_html=True)

            if content.startswith("SQL:"):
                parts = content.split("\n\n")
                sql_query = parts[0].replace("SQL:", "").strip()
                data_json = None
                stats_json = None
                inference = None

                for part in parts[1:]:
                    if part.startswith("DATA:"):
                        data_json = part.replace("DATA:", "").strip()
                    elif part.startswith("STATS:"):
                        stats_json = part.replace("STATS:", "").strip()
                    elif part.startswith("Inference:"):
                        inference = part.replace("Inference:", "").strip()

                st.code(sql_query, language="sql")

                if data_json:
                    try:
                        df = process_sql_result(data_json)
                        st.dataframe(df, height=400)

                        if stats_json:
                            try:
                                stats = json.loads(stats_json)
                                st.subheader("Statistical Summary")
                                stats_df = pd.DataFrame(stats).T
                                st.table(stats_df)

                                # Automatic visualization
                                st.subheader("Automatic Data Visualization")
                                auto_chart_type, auto_x_col, auto_y_col = auto_select_chart(df)
                                auto_fig, auto_error_message = create_visualization(df, auto_chart_type, auto_x_col, auto_y_col)
                                if auto_fig:
                                    st.plotly_chart(auto_fig)
                                elif auto_error_message:
                                    st.error(auto_error_message)

                                # Custom visualization options
                                st.subheader("Custom Data Visualization")
                                viz_option = st.radio(
                                    "Choose visualization type",
                                    ["Chart Visualization", "Statistical Visualization", "Correlation Heatmap"],
                                    key=f"viz_option_{idx}"
                                )

                                if viz_option == "Chart Visualization":
                                    chart_types = ["Bar", "Line", "Scatter", "Histogram", "Box", "Violin", "Distribution"]
                                    chart_type = st.selectbox("Select chart type", chart_types, key=f"chart_type_{idx}")

                                    if chart_type in ["Histogram", "Box", "Violin", "Distribution"]:
                                        x_col = st.selectbox("Select column", df.columns, key=f"x_col_{idx}")
                                        y_col = None
                                    else:
                                        x_col = st.selectbox("Select X-axis", df.columns, key=f"x_col_{idx}")
                                        y_col = st.selectbox("Select Y-axis", df.columns, key=f"y_col_{idx}")

                                    if st.button("Generate Custom Chart", key=f"chart_button_{idx}"):
                                        custom_fig, custom_error_message = create_visualization(df, chart_type, x_col, y_col)
                                        if custom_fig:
                                            st.plotly_chart(custom_fig)
                                        elif custom_error_message:
                                            st.error(custom_error_message)

                                elif viz_option == "Statistical Visualization":
                                    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
                                    stat_col = st.selectbox("Select column for statistical visualization",
                                                            numeric_columns,
                                                            key=f"stat_col_{idx}")

                                    if st.button("Generate Statistical Visualization", key=f"stat_button_{idx}"):
                                        fig, axes = plt.subplots(2, 2, figsize=(20, 20))

                                        # Box plot
                                        sns.boxplot(y=df[stat_col], ax=axes[0, 0])
                                        axes[0, 0].set_title(f'Box Plot of {stat_col}')

                                        # Distribution plot
                                        sns.histplot(df[stat_col], kde=True, ax=axes[0, 1])
                                        axes[0, 1].set_title(f'Distribution of {stat_col}')

                                        # Bar plot of calculated statistics
                                        stat_values = stats[stat_col]
                                        stat_values = {k: v for k, v in stat_values.items() if k != 'count'}  # Exclude 'count' for better scaling
                                        sns.barplot(x=list(stat_values.keys()), y=list(stat_values.values()), ax=axes[1, 0])
                                        axes[1, 0].set_title(f'Calculated Statistics for {stat_col}')
                                        axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45)

                                        # Scatter plot with another numeric column (if available)
                                        other_numeric_columns = [col for col in numeric_columns if col != stat_col]
                                        if other_numeric_columns:
                                            other_col = other_numeric_columns[0]  # Choose the first other numeric column
                                            sns.scatterplot(x=df[stat_col], y=df[other_col], ax=axes[1, 1])
                                            axes[1, 1].set_title(f'Scatter Plot: {stat_col} vs {other_col}')
                                        else:
                                            axes[1, 1].axis('off')  # Turn off the last subplot if no other numeric column

                                        plt.tight_layout()
                                        st.pyplot(fig)

                                else:  # Correlation Heatmap
                                    numeric_df = df.select_dtypes(include=['int64', 'float64'])

                                    if numeric_df.empty:
                                        st.warning("Correlation heatmap is not possible as there are no numeric columns in the dataset.")

                                    elif numeric_df.shape[1] < 2:
                                        st.warning("Correlation heatmap requires at least two numeric columns.")

                                    else:
                                        if st.button("Generate Correlation Heatmap", key=f"corr_button_{idx}"):
                                            try:
                                                corr_matrix = numeric_df.corr()
                                                if corr_matrix.empty:
                                                    st.warning("Unable to generate correlation heatmap. The correlation matrix is empty.")
                                                else:
                                                    fig, ax = plt.subplots(figsize=(12, 10))
                                                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
                                                    ax.set_title('Correlation Heatmap')
                                                    st.pyplot(fig)
                                            except Exception as e:
                                                st.error(f"An error occurred while generating the correlation heatmap: {str(e)}")

                            except json.JSONDecodeError:
                                st.warning("Unable to parse statistics data.")

                        if inference:
                            st.markdown(f"**Inference:**\n{inference}")

                    except json.JSONDecodeError:
                        st.error("Error decoding JSON data from the response.")
                        st.text(data_json)  # Display the raw data for debugging

            elif content.startswith("NLP:"):
                nlp_result = content.replace("NLP:", "").strip()
                st.markdown(nlp_result)

            else:
                st.markdown(content)

            st.markdown('---', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("Enterprise Data Assistant")

    selected_db = st.selectbox("Select Database", list(available_databases.keys()))
    if st.button("Switch Database"):
        try:
            new_db_path = available_databases[selected_db]
            router_agent.change_database(new_db_path)
            st.success(f"Switched to {selected_db}")
        except Exception as e:
            st.error(f"Error switching database: {str(e)}")
            st.rerun()

    show_schema = st.checkbox("Show Database Schema", value=st.session_state.show_schema)
    st.session_state.show_schema = show_schema

    if st.session_state.show_schema:
        st.subheader("Database Schema")
        st.text(router_agent.db_schema)

    if st.button("Export Chat History"):
        chat_json = export_chat_history()
        st.download_button(
            label="Download Chat History",
            data=chat_json,
            file_name="chat_history.json",
            mime="application/json"
        )
        st.text_area("Command Log", value=st.session_state.log_data.getvalue(), height=300)

    log_contents = st.session_state.log_data.getvalue()
    st.download_button(
        label="Download Log",
        data=log_contents,
        file_name="nlp_sql_assistant_log.txt",
        mime="text/plain"
    )
