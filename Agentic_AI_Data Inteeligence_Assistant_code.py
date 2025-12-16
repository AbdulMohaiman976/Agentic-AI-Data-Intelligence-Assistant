# ============================================
# 🚀 COMPLETE COLAB CODE - READY TO RUN
# Multi-Agent System with Groq + LangGraph
# ============================================

# STEP 1: INSTALLATION
print("🔧 Installing packages...")
!pip install -q gradio transformers accelerate sentencepiece
!pip install -q langchain-groq langgraph langchain-community tavily-python
!pip install -q openpyxl pandas matplotlib seaborn plotly sqlalchemy

print("✅ Installation complete!\n")

# ============================================
# IMPORTS
# ============================================

import os
import torch
import sqlite3
import re
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Literal, TypedDict, Annotated, Any
from io import StringIO, BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr
import tempfile
from transformers import pipeline
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.tools.tavily_search import TavilySearchResults

print("✅ Libraries imported!\n")

# ============================================
# CONFIGURATION
# ============================================

GROQ_API_KEY = ""
TAVILY_API_KEY = ""

os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

SAMPLE_DB_FILE = "user_database.db"
UPLOADED_DATA_FILE = "uploaded_data.db"

print("🔑 Configuration loaded!\n")

# ============================================
# DATABASE INITIALIZATION
# ============================================

def init_database():
    """Initialize database with sample data."""
    try:
        conn = sqlite3.connect(SAMPLE_DB_FILE)
        cursor = conn.cursor()

        cursor.execute("PRAGMA foreign_keys = ON")

        # Create tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='customers'")
        if not cursor.fetchone():
            cursor.execute("""
            CREATE TABLE customers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                email TEXT UNIQUE,
                country TEXT,
                revenue REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='orders'")
        if not cursor.fetchone():
            cursor.execute("""
            CREATE TABLE orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id INTEGER,
                amount REAL,
                order_date DATE,
                status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (customer_id) REFERENCES customers(id)
            )
            """)

        # Insert sample data
        cursor.execute("SELECT COUNT(*) FROM customers")
        if cursor.fetchone()[0] == 0:
            sample_customers = [
                ('Ali Khan', 'ali@email.com', 'Pakistan', 15000),
                ('Sarah Ahmed', 'sarah@email.com', 'Pakistan', 25000),
                ('John Doe', 'john@email.com', 'USA', 50000),
                ('Maria Garcia', 'maria@email.com', 'Spain', 30000),
                ('Ahmed Hassan', 'ahmed@email.com', 'Egypt', 20000),
            ]

            sample_orders = [
                (1, 5000, '2024-01-15', 'Completed'),
                (1, 10000, '2024-02-20', 'Completed'),
                (2, 15000, '2024-01-10', 'Completed'),
                (2, 10000, '2024-03-05', 'Pending'),
                (3, 50000, '2024-02-28', 'Completed'),
            ]

            cursor.executemany(
                "INSERT INTO customers (name, email, country, revenue) VALUES (?,?,?,?)",
                sample_customers
            )
            cursor.executemany(
                "INSERT INTO orders (customer_id, amount, order_date, status) VALUES (?,?,?,?)",
                sample_orders
            )

        conn.commit()
        conn.close()
        print("✅ Database initialized!\n")
        return True
    except Exception as e:
        print(f"⚠️ Database error: {e}\n")
        return False

init_database()

# ============================================
# LOAD MODELS - GROQ + TRANSLATOR
# ============================================

print("🤖 Loading models...")

translator = None
try:
    translator = pipeline(
        "translation",
        model="facebook/nllb-200-distilled-600M",
        src_lang="urd_Arab",
        tgt_lang="eng_Latn",
        device=0 if torch.cuda.is_available() else -1
    )
    print("✅ Translator loaded\n")
except Exception as e:
    print(f"⚠️ Translator unavailable: {e}\n")

# Groq Models
sql_generator_llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=2048,
    top_p=0.95
)

summarizer_llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.3,
    max_tokens=1024
)

general_llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.7,
    max_tokens=2048
)

data_analyzer_llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.2,
    max_tokens=1024
)

print("✅ All Groq models loaded\n")

# ============================================
# STATE DEFINITION
# ============================================

class AgentState(TypedDict):
    """State for LangGraph workflow."""
    user_input: str
    detected_intent: str
    translated_input: str
    sql_query: Optional[str]
    sql_results: Optional[pd.DataFrame]
    query_status: str
    current_db: str
    uploaded_data: Optional[pd.DataFrame]
    uploaded_file_name: Optional[str]
    data_schema: Optional[str]
    available_tables: List[str]
    web_raw_results: Optional[List[Dict]]
    web_summary: Optional[str]
    visualization_path: Optional[str]
    visualization_prompt: str
    final_response: str
    execution_log: List[str]
    messages: Annotated[list, add_messages]

# ============================================
# NODE 1: INTENT DETECTION
# ============================================

def detect_intent_node(state: AgentState) -> AgentState:
    """Detect user intent using keywords."""
    user_input = state['user_input'].lower()

    intent_patterns = {
        "upload_data": ["upload", "csv", "excel", "import", "load", "file upload"],
        "analyze_data": ["analyze", "summary", "describe", "statistics", "analysis"],
        "create_table": ["create table", "new table", "bana", "table bnao"],
        "drop_table": ["drop table", "delete table", "remove table"],
        "insert_data": ["insert", "add data", "add row"],
        "sql": ["show", "select", "get", "fetch", "find", "list", "display", "customer", "order", "query", "where", "revenue", "amount", "count", "sum", "join"],
        "visualization": ["chart", "graph", "plot", "visualize", "bar", "pie", "line"],
        "web_search": ["search", "find on web", "latest", "news", "what is", "who is", "recent"]
    }

    detected = "general"
    for intent_type, keywords in intent_patterns.items():
        if any(kw in user_input for kw in keywords):
            detected = intent_type
            break

    state['detected_intent'] = detected
    state['execution_log'].append(f"🧠 Intent: {detected}")
    print(f"✅ Intent: {detected}\n")

    return state

# ============================================
# NODE 2: TRANSLATION
# ============================================

def translation_node(state: AgentState) -> AgentState:
    """Translate non-English input."""
    user_input = state['user_input']

    if any(ord(char) > 127 for char in user_input):
        if translator:
            try:
                translated = translator(user_input)[0]["translation_text"]
                state['translated_input'] = translated
                state['execution_log'].append(f"🌍 Translated")
                print(f"🌍 Translated: {translated}\n")
            except:
                state['translated_input'] = user_input
        else:
            state['translated_input'] = user_input
    else:
        state['translated_input'] = user_input

    return state

# ============================================
# NODE 3: SCHEMA EXTRACTION
# ============================================

def extract_data_schema_node(state: AgentState) -> AgentState:
    """Extract database schema."""
    print("📊 Extracting schema...")

    try:
        conn = sqlite3.connect(state['current_db'])
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        state['available_tables'] = tables

        schema = "Available Tables:\n\n"
        for table_name in tables:
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()

            schema += f"{table_name}:\n"
            for col in columns:
                schema += f"  - {col[1]} ({col[2]})\n"
            schema += "\n"

        state['data_schema'] = schema
        state['execution_log'].append(f"📋 Schema ready")
        print(f"✅ Schema extracted\n")
        conn.close()
    except Exception as e:
        state['execution_log'].append(f"❌ Schema error: {e}")

    return state

# ============================================
# NODE 4: SQL GENERATION WITH GROQ
# ============================================

def sql_generation_node(state: AgentState) -> AgentState:
    """Generate SQL using Groq (Best Model!)."""
    question = state['translated_input']
    intent = state['detected_intent']

    print("🔨 Generating SQL with Groq...")

    try:
        schema_info = state.get('data_schema', 'No schema')

        if intent == "create_table":
            prompt = f"""Generate ONLY SQLite CREATE TABLE statement from this request:
Request: {question}
Reply with ONLY SQL:"""
        elif intent == "drop_table":
            prompt = f"""Generate ONLY SQLite DROP TABLE statement:
Request: {question}
Reply with ONLY SQL:"""
        elif intent == "insert_data":
            prompt = f"""Generate ONLY SQLite INSERT statement:
Request: {question}
Database: {schema_info}
Reply with ONLY SQL:"""
        else:
            prompt = f"""Generate ONLY SQLite SELECT query:
Question: {question}
Database: {schema_info}
Reply with ONLY SQL:"""

        response = sql_generator_llm.invoke(prompt)
        sql_query = response.content.strip()

        sql_query = re.sub(r'```sql\n|```\n?|```', '', sql_query).strip()
        if not sql_query.endswith(';'):
            sql_query += ';'

        state['sql_query'] = sql_query
        state['execution_log'].append(f"📝 SQL: {sql_query[:40]}...")
        print(f"✅ SQL: {sql_query}\n")

        # Execute
        conn = sqlite3.connect(state['current_db'])
        cursor = conn.cursor()

        try:
            if intent in ["create_table", "drop_table", "insert_data"]:
                cursor.execute(sql_query)
                conn.commit()
                state['query_status'] = "✅ Operation successful"
                state['sql_results'] = pd.DataFrame({"Status": ["✅ Success"]})
            else:
                state['sql_results'] = pd.read_sql_query(sql_query, conn)
                state['query_status'] = f"✅ {len(state['sql_results'])} rows"

            conn.close()
        except Exception as e:
            conn.close()
            state['query_status'] = f"❌ Error: {str(e)}"
            state['sql_results'] = pd.DataFrame({"Error": [str(e)]})

    except Exception as e:
        state['query_status'] = f"❌ Error: {str(e)}"
        state['sql_results'] = pd.DataFrame({"Error": [str(e)]})

    return state

# ============================================
# NODE 5: DATA UPLOAD
# ============================================

def data_upload_node(state: AgentState) -> AgentState:
    """Handle uploaded CSV/Excel files."""
    print("📥 Processing upload...")

    if state['uploaded_data'] is None:
        return state

    try:
        df = state['uploaded_data']
        file_name = state['uploaded_file_name']

        conn = sqlite3.connect(UPLOADED_DATA_FILE)

        table_name = file_name.split('.')[0].lower().replace(' ', '_')
        table_name = re.sub(r'[^a-z0-9_]', '', table_name)
        if not table_name:
            table_name = "imported_data"

        df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()

        state['current_db'] = UPLOADED_DATA_FILE
        state['execution_log'].append(f"✅ Upload: {len(df)} rows, {len(df.columns)} cols")
        print(f"✅ Upload complete: {len(df)} rows\n")

    except Exception as e:
        state['execution_log'].append(f"❌ Upload error: {e}")

    return state

# ============================================
# NODE 6: DATA ANALYSIS WITH GROQ
# ============================================

def data_analysis_node(state: AgentState) -> AgentState:
    """Analyze data with Groq."""
    print("🔍 Analyzing data...")

    try:
        df = state['uploaded_data']
        question = state['translated_input']

        if df is None or df.empty:
            state['final_response'] = "No data to analyze"
            return state

        data_info = f"""Data Summary:
Shape: {df.shape[0]} rows, {df.shape[1]} columns
Columns: {', '.join(df.columns.tolist())}

Sample:
{df.head(5).to_string()}

Stats:
{df.describe().to_string()}"""

        prompt = f"""Analyze this data and answer the question:
Question: {question}

{data_info}

Provide detailed analysis:"""

        response = data_analyzer_llm.invoke(prompt)
        state['final_response'] = response.content
        state['execution_log'].append("✅ Analysis complete")
        print("✅ Analysis complete\n")

    except Exception as e:
        state['final_response'] = f"Analysis error: {e}"

    return state

# ============================================
# NODE 7: VISUALIZATION
# ============================================

def visualization_node(state: AgentState) -> AgentState:
    """Create visualization."""
    print("📊 Creating chart...")

    try:
        df = state['sql_results'] or state['uploaded_data']

        if df is None or df.empty:
            state['final_response'] = "No data for visualization"
            return state

        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        if len(numeric_cols) < 1:
            state['final_response'] = "Need numeric data"
            return state

        plt.figure(figsize=(12, 6))

        if len(numeric_cols) >= 2:
            plt.scatter(range(len(df)), df[numeric_cols[0]])
            plt.scatter(range(len(df)), df[numeric_cols[1]])
        else:
            plt.bar(range(len(df)), df[numeric_cols[0]])

        plt.title("Data Visualization")
        plt.tight_layout()

        chart_path = f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()

        state['visualization_path'] = chart_path
        state['execution_log'].append("✅ Chart created")
        print("✅ Chart created\n")

    except Exception as e:
        state['execution_log'].append(f"❌ Chart error: {e}")

    return state

# ============================================
# NODE 8: WEB SEARCH
# ============================================

def web_search_node(state: AgentState) -> AgentState:
    """Search web and summarize."""
    query = state['translated_input']

    print(f"🌐 Searching web...")

    try:
        search_tool = TavilySearchResults(max_results=5)
        raw_results = search_tool.invoke(query)

        contents = []
        for i, result in enumerate(raw_results, 1):
            contents.append(f"Source {i}: {result.get('title', 'N/A')}\n{result.get('content', '')}")

        combined = "\n\n".join(contents)

        summarize_prompt = f"""Summarize these web results clearly:

{combined}

Provide comprehensive summary:"""

        summary_response = summarizer_llm.invoke(summarize_prompt)
        state['web_summary'] = summary_response.content
        state['final_response'] = state['web_summary']
        state['execution_log'].append("✅ Web search done")
        print("✅ Web search complete\n")

    except Exception as e:
        error_msg = f"Web search error: {str(e)}"
        state['web_summary'] = error_msg
        state['final_response'] = error_msg

    return state

# ============================================
# NODE 9: GENERAL CHAT
# ============================================

def general_chat_node(state: AgentState) -> AgentState:
    """General conversation."""
    question = state['translated_input']

    print(f"💬 Chatting...")

    try:
        response = general_llm.invoke(question)
        state['final_response'] = response.content
        state['execution_log'].append("✅ Response generated")
    except Exception as e:
        state['final_response'] = f"Error: {str(e)}"

    return state

# ============================================
# NODE 10: FORMAT RESPONSE
# ============================================

def format_response_node(state: AgentState) -> AgentState:
    """Format final response."""

    if state['detected_intent'] in ["sql", "create_table", "insert_data", "drop_table"]:
        if state['sql_results'] is not None:
            df = state['sql_results']
            sql_query = state.get('sql_query', 'N/A')

            table_str = df.to_string(index=False, max_rows=25)

            response = f"""✅ **SQL Executed Successfully**

**Query:**
```sql
{sql_query}
```

**Status:** {state['query_status']}

**Results:**
```
{table_str}
```
"""
            if len(df) > 25:
                response += f"\n*(Showing 25 of {len(df)} rows)*"

            state['final_response'] = response

    return state

# ============================================
# ROUTER FUNCTION
# ============================================

def route_based_on_intent(state: AgentState) -> str:
    """Route based on intent."""
    intent = state['detected_intent']

    routing = {
        "upload_data": "data_upload",
        "analyze_data": "data_analysis",
        "create_table": "extract_schema",
        "drop_table": "extract_schema",
        "insert_data": "extract_schema",
        "sql": "extract_schema",
        "visualization": "visualization",
        "web_search": "web_search",
        "general": "general_chat"
    }

    return routing.get(intent, "general_chat")

# ============================================
# BUILD LANGGRAPH WORKFLOW
# ============================================

print("🔧 Building LangGraph workflow...")

workflow = StateGraph(AgentState)

# Add all nodes
workflow.add_node("detect_intent", detect_intent_node)
workflow.add_node("translation", translation_node)
workflow.add_node("extract_schema", extract_data_schema_node)
workflow.add_node("sql_generation", sql_generation_node)
workflow.add_node("data_upload", data_upload_node)
workflow.add_node("data_analysis", data_analysis_node)
workflow.add_node("visualization", visualization_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("general_chat", general_chat_node)
workflow.add_node("format_response", format_response_node)

# Add edges
workflow.add_edge(START, "detect_intent")
workflow.add_edge("detect_intent", "translation")

# Conditional routing
workflow.add_conditional_edges(
    "translation",
    route_based_on_intent,
    {
        "data_upload": "data_upload",
        "data_analysis": "data_analysis",
        "extract_schema": "extract_schema",
        "visualization": "visualization",
        "web_search": "web_search",
        "general_chat": "general_chat"
    }
)

# Path connections
workflow.add_edge("extract_schema", "sql_generation")
workflow.add_edge("sql_generation", "format_response")
workflow.add_edge("data_upload", "data_analysis")
workflow.add_edge("visualization", "format_response")
workflow.add_edge("data_analysis", END)
workflow.add_edge("web_search", END)
workflow.add_edge("general_chat", END)
workflow.add_edge("format_response", END)

app = workflow.compile()

print("✅ Workflow compiled!\n")

# ============================================
# MAIN PROCESSOR FUNCTION
# ============================================

def process_query(user_input: str, uploaded_file=None) -> tuple:
    """Process query through LangGraph."""

    print(f"\n{'='*70}")
    print(f"📥 INPUT: {user_input}")
    print(f"{'='*70}\n")

    uploaded_df = None
    file_name = None

    if uploaded_file is not None:
        try:
            file_path = uploaded_file.name if hasattr(uploaded_file, 'name') else str(uploaded_file)
            file_name = file_path.split('/')[-1]

            if file_path.endswith('.csv'):
                uploaded_df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                uploaded_df = pd.read_excel(file_path)

            print(f"✅ File: {len(uploaded_df)} rows, {len(uploaded_df.columns)} cols\n")
        except Exception as e:
            print(f"❌ File error: {e}\n")

    initial_state = {
        'user_input': user_input,
        'detected_intent': '',
        'translated_input': '',
        'sql_query': None,
        'sql_results': None,
        'query_status': '',
        'current_db': SAMPLE_DB_FILE,
        'uploaded_data': uploaded_df,
        'uploaded_file_name': file_name,
        'data_schema': None,
        'available_tables': [],
        'web_raw_results': None,
        'web_summary': None,
        'visualization_path': None,
        'visualization_prompt': '',
        'final_response': '',
        'execution_log': [],
        'messages': []
    }

    try:
        final_state = app.invoke(initial_state)

        response = final_state.get('final_response', 'No response')
        chart = final_state.get('visualization_path')

        log_text = "\n\n### 📋 Execution Flow:\n" + "\n".join(final_state['execution_log'])
        full_response = response + log_text

        return full_response, chart

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(error_msg)
        return error_msg, None

# ============================================
# GRADIO INTERFACE
# ============================================

print("🎨 Building Gradio interface...\n")

with gr.Blocks(theme=gr.themes.Soft(), title="Multi-Agent AI System") as demo:

    gr.Markdown("""
    # 🤖 Advanced Multi-Agent AI System
    ### Powered by Groq + LangGraph

    **Features:**
    - ✅ **Groq SQL** - Perfect SQL generation
    - ✅ **CSV Upload** - Upload and analyze instantly
    - ✅ **Smart Analysis** - Groq-powered data insights
    - ✅ **Web Search** - Search + summarization
    - ✅ **Visualization** - Auto charts
    - ✅ **Multilingual** - Urdu support
    - ✅ **LangGraph** - State management & routing
    """)

    with gr.Row():
        with gr.Column(scale=2):
            query_input = gr.Textbox(
                label="Your Question",
                placeholder="Query database, upload CSV, search web, create charts...",
                lines=4
            )

            file_upload = gr.File(
                label="📁 Upload CSV/Excel",
                file_types=[".csv", ".xlsx", ".xls"]
            )

            submit_btn = gr.Button("🚀 Submit", variant="primary", size="lg")

        with gr.Column(scale=3):
            output = gr.Markdown(label="Response")
            chart = gr.Image(label="📊 Chart")

    gr.Examples(
        examples=[
            ["Show all customers"],
            ["مجھے سب customers دکھائیں"],
            ["Create table products with id, name, price"],
            ["Search latest AI developments"],
            ["Analyze uploaded data"],
            ["Create chart of revenue by country"],
        ],
        inputs=query_input
    )

    submit_btn.click(
        process_query,
        inputs=[query_input, file_upload],
        outputs=[output, chart]
    )

print("✅ Interface ready!\n")

# ============================================
# LAUNCH
# ============================================

print("""
╔════════════════════════════════════════════════════════════════╗
║         🚀 SYSTEM READY!                                       ║
╚════════════════════════════════════════════════════════════════╝

✨ Key Features:
   ✅ Groq SQL Model (Best!)
   ✅ CSV/Excel Upload Support
   ✅ Smart Data Analysis
   ✅ Web Search + Summarization
   ✅ Auto Visualization
   ✅ LangGraph Workflow
   ✅ Multilingual Support

📱 Launching...
""")

demo.launch(share=True, debug=False)