import os
import streamlit as st
import tempfile
import warnings
warnings.filterwarnings("ignore")
import chromadb
from agno.agent import Agent
from agno.models.google import Gemini
from agno.models.xai import xAI
from agno.embedder.google import GeminiEmbedder
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.vectordb.chroma import ChromaDb
from agno.vectordb.chroma.chromadb import ChromaDb
from agno.document.chunking.document import DocumentChunking
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

# Initialize Streamlit
st.set_page_config(page_title="AI Legal Team Agents", page_icon="⚖️", layout="wide")

# Page title and description
st.markdown("<h1 style='text-align: center; color: #3e8e41;'>👨‍⚖️ AI Legal Team Agents</h1>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; font-size: 18px; color: #4B0082;'>
        Upload your legal document and let the <b>AI LegalAdvisor</b>, <b>AI ContractsAnalyst</b>, 
        <b>AI LegalStrategist</b>, and <b>AI Team Lead</b> do the work for you. You can also ask 
        questions in between for enhanced collaboration and insights.
    </div>
""", unsafe_allow_html=True)


# Session state init
if "vector_db" not in st.session_state:
    st.session_state.vector_db = ChromaDb(
        collection="law", path="tmp/chromadb", persistent_client=True, embedder= GeminiEmbedder()
    )
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None


from dotenv import load_dotenv
import os
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Sidebar
with st.sidebar:
    st.header("Configuration")

    chunk_size_in = st.sidebar.number_input("Chunk Size", min_value=1, max_value=5000, value=500)
    overlap_in = st.sidebar.number_input("Overlap", min_value=1, max_value=1000, value=100)

    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader("Upload a Legal Document (PDF)", type=["pdf"])

    if uploaded_file:
        if uploaded_file.name not in st.session_state.processed_files:
            with st.spinner("Processing document..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_path = temp_file.name

                    st.session_state.knowledge_base = PDFKnowledgeBase(
                        path=temp_path,
                        vector_db=st.session_state.vector_db,
                        reader=PDFReader(),
                        chunking_strategy=DocumentChunking(chunk_size=chunk_size_in, overlap=overlap_in)
                    )
                    st.session_state.knowledge_base.load(recreate=True, upsert=True)
                    st.session_state.processed_files.add(uploaded_file.name)
                    st.session_state.last_uploaded_file = uploaded_file.name

                    st.success("✅ Document processed and stored in knowledge base!")
                except Exception as e:
                    st.error(f"Error processing document: {e}")
    elif st.session_state.last_uploaded_file:
        st.info(f"Reusing previous file: **{st.session_state.last_uploaded_file}**")

# Agents
if st.session_state.knowledge_base:
    legal_researcher = Agent(
        name="LegalAdvisor",
        model=Gemini(id="gemini-2.0-flash-exp"),
        knowledge=st.session_state.knowledge_base,
        search_knowledge=True,
        description="Legal Researcher AI - Finds and cites relevant legal cases, regulations, and precedents using all data in the knowledge base.",
        instructions=[
            "Extract all available data from the knowledge base and search for legal cases, regulations, and citations.",
            "If needed, use DuckDuckGo for additional legal references.",
            "Always provide source references in your answers."
        ],
        tools=[DuckDuckGoTools()],
        show_tool_calls=True,
        markdown=True
    )

    contract_analyst = Agent(
        name="ContractAnalyst",
        model=Gemini(id="gemini-2.0-flash-exp"),
        knowledge=st.session_state.knowledge_base,
        search_knowledge=True,
        description="Contract Analyst AI - Reviews contracts and identifies key clauses, risks, and obligations using the full document data.",
        instructions=[
            "Extract all available data from the knowledge base and analyze the contract for key clauses, obligations, and potential ambiguities.",
            "Reference specific sections of the contract where possible."
            "Don't give any extra excuses or inaccurate responses as it is for legal works"
        ],
        show_tool_calls=True,
        markdown=True
    )

    legal_strategist = Agent(
        name="LegalStrategist",
        model=Gemini(id="gemini-2.0-flash-exp"),
        knowledge=st.session_state.knowledge_base,
        search_knowledge=True,
        description="Legal Strategist AI - Provides comprehensive risk assessment and strategic recommendations based on all the available data from the contract.",
        instructions=[
            "Using all data from the knowledge base, assess the contract for legal risks and opportunities.",
            "Provide actionable recommendations and ensure compliance with applicable laws."
        ],
        show_tool_calls=True,
        markdown=True
    )

    team_lead = Agent(
        name="teamlead",
        model=Gemini(id="gemini-2.0-flash-exp"),
        description="Team Lead AI - Integrates responses from the Legal Researcher, Contract Analyst, and Legal Strategist into a comprehensive report.",
        instructions=[
            "Combine and summarize all insights provided by the Legal Researcher, Contract Analyst, and Legal Strategist. "
            "Ensure the final report includes references to all relevant sections from the document."
            "Don't give any extra excuses or inaccurate responses as it is for legal works"
        ],
        show_tool_calls=True,
        markdown=True
    )

    # Cached function for performance
    @st.cache_data(show_spinner="🔎 Running analysis...")
    def get_team_response(query):
        research_response = legal_researcher.run(query)
        contract_response = contract_analyst.run(query)
        strategy_response = legal_strategist.run(query)

        final_response = team_lead.run(
            f"Summarize and integrate the following insights gathered using the full contract data:\n\n"
            f"Legal Researcher:\n{research_response}\n\n"
            f"Contract Analyst:\n{contract_response}\n\n"
            f"Legal Strategist:\n{strategy_response}\n\n"
            "Provide a structured legal analysis report that includes key terms, obligations, risks, and recommendations, with references to the document."
        )
        return research_response, contract_response, strategy_response, final_response

# UI – Analysis options
if st.session_state.knowledge_base:
    st.header("🔍 Select Analysis Type")
    analysis_type = st.selectbox(
        "Choose Analysis Type:",
        ["Contract Review", "Legal Research", "Risk Assessment", "Compliance Check", "Custom Query"]
    )

    query = None
    if analysis_type == "Custom Query":
        query = st.text_area("Enter your custom legal question:")
    else:
        predefined_queries = {
            "Contract Review": (
                "Analyze this document, contract, or agreement using all available data from the knowledge base. "
                "Identify key terms, obligations, and risks in detail."
            ),
            "Legal Research": (
                "Using all available data from the knowledge base, find relevant legal cases and precedents related to this document, contract, or agreement. "
                "Provide detailed references and sources."
            ),
            "Risk Assessment": (
                "Extract all data from the knowledge base and identify potential legal risks in this document, contract, or agreement. "
                "Detail specific risk areas and reference sections of the text."
            ),
            "Compliance Check": (
                "Evaluate this document, contract, or agreement for compliance with legal regulations using all available data from the knowledge base. "
                "Highlight any areas of concern and suggest corrective actions."
            )
        }
        query = predefined_queries[analysis_type]

    if st.button("Analyze"):
        if not query:
            st.warning("Please enter a query.")
        else:
            with st.spinner("Analyzing..."):
                r1, r2, r3, team_response = get_team_response(query)

                tabs = st.tabs(["📑 Team Analysis", "🔎 LegalAdvisor", "📂 ContractAnalyst", "⚖️ LegalStrategist", "📌 Key Points", "📋 Recommendations"])

                with tabs[0]:
                    st.subheader("🧠 Combined Legal Analysis")
                    st.markdown(team_response.content or "No response generated.")
                    st.download_button("📥 Download Full Report", team_response.content or "", file_name="legal_analysis.txt")

                with tabs[1]:
                    st.subheader("🔎 LegalAdvisor's Input")
                    st.markdown(r1.content or "No response.")

                with tabs[2]:
                    st.subheader("📂 ContractAnalyst's Input")
                    st.markdown(r2.content or "No response.")

                with tabs[3]:
                    st.subheader("⚖️ LegalStrategist's Input")
                    st.markdown(r3.content or "No response.")

                with tabs[4]:
                    st.subheader("📌 Key Points Summary")
                    summary = team_lead.run(f"Summarize the key legal points from this analysis:\n{team_response.content}")
                    st.markdown(summary.content or "No summary generated.")

                with tabs[5]:
                    st.subheader("📋 Recommendations")
                    recs = team_lead.run(f"Provide specific legal recommendations based on this analysis:\n{team_response.content}")
                    st.markdown(recs.content or "No recommendations generated.")
