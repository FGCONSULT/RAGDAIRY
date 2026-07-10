"""
RAGDAIRY — Vocalkisan Dairy AI Document Assistant
Preloaded RAG application for viksitdairy2047.in
Powered by Groq API
"""

import os
import uuid
import streamlit as st
from llama_index.core import Settings, VectorStoreIndex, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# --------------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------------
st.set_page_config(page_title="Vocalkisan Dairy AI Assistant", page_icon="🐄", layout="wide")
st.title("🐄 Vocalkisan Dairy — Document Q&A (Preloaded RAG)")
st.caption(
    "Ask questions about breed improvement, nutrition, bulking impacts, "
    "economic/financial analysis, and risk models. Answers are strictly grounded in project data."
)

# --------------------------------------------------------------------------------
# Embedded Project Data (Breed Improvement, Nutrition, Financials & Risk Models)
# --------------------------------------------------------------------------------
NDSP_DATA = """
INDIA: National Dairy Support Project
Economic and Financial Analysis

1. Project Overview & Components:
The project includes two main components:
- Component A - Productivity Enhancement: will increase milk productivity through improved animal breeding, nutrition, and delivery of Artificial Insemination (AI) services. Accounts for 57% of total project costs, shared between animal breed improvement and AI services (36%) and animal nutrition (21%).
- Component B - Milk Collection and Bulking: will improve access to markets by investing in village level milk collection and bulking facilities and formation of producer companies and dairy cooperative societies. Accounts for 37% of total project costs.

2. Project Interventions and Zones:
The project covers 14 states grouped into two broad zones:
- Zone 1 (Convergence Zone): Includes about 20,000 villages in 5 states (Andhra Pradesh, Gujarat - Saurashtra and Kutch regions, Punjab, Rajasthan, and Uttar Pradesh) where producer companies will be promoted. These areas receive ALL project interventions: bull production, semen production, AI, Ration Balancing Program (RBP), and milk pooling/bulking. All project benefits are counted in this zone.
- Zone 2: Includes about 20,000 villages with existing cooperatives. AI services are provided through existing non-project systems (government/NGOs). The project provides other interventions: expanding milk pooling/bulking through dairy cooperative societies (DCS) and RBP. Only benefits of RBP and milk collection/bulking are counted.
- Zone 3: HGM Semen doses from the project are used here (95% are used outside Zone 1) but delivered by non-project AI centers. Not considered for benefit quantification.

3. Breed Improvement and AI Services Impacts (Component A - 36% of costs):
- High Genetic Merit (HGM) bulls will supply about 3.85 million semen doses annually, increasing milk productivity through better genetics.
- 3,000 trained Mobile AI Technicians (MAITs) will deliver AI services, improving conception rate from 35% (Without Project) to 45% (With Project), reducing the inter-calving period by 14 days.
- (a) Genetics Impact: Without Project (WOP), genetic milk productivity increases by 0.2% annually. With Project (WP), productivity increases by 5% for daughters milking in 5th year (cross bred) or 6th year (indigenous/buffalo), followed by an annual compounded rate of 1%. Cross bred daughters from indigenous cows achieve a 220% jump in year 6, then 1% compounded annually. Benefits count from year 7.
- Average milk yield increase: 12% in year 7, 16% in year 8, and 26% in year 20. Average annual growth is 1.5% WP vs 0.2% WOP.
- By year 7, 36,170 HGM bovines produced, stabilizing at 3.5 million annually from year 15. Incremental milk production reaches 24,440 tons in year 7 and 1.99 million tons in year 20.
- Financial Gross Margin: Average is Rs 3,065 per metric ton. Genetic improvement improves annual gross margin to Rs 75 million in year 7 and Rs 6,100 million in year 15.
- (b) AI Services/Inter-calving Impact: MAITs scale from 45 to 136 inseminations/month by year 6. 1.56 million bovines inseminated annually, reducing inter-calving by 14 days. Average yield is 5.6 kg/day. Incremental milk per lactation is 78 kg. Annual incremental milk reaches 0.12 million tons from year 8. Financial benefits are Rs 373 million annually from year 8 onwards (Zone 1 only).

4. Bulking and Nutrition Impacts (Ration Balancing Program - RBP, 21% of costs):
- RBP feed innovation delivered through 40,319 trained Local Resource Persons (LRPs) across 40,319 villages covering 2.7 million bovines annually. LRPs shared equally between Zone 1 and Zone 2.
- (a) Increased Milk Productivity: Average yield is 4.7 kg/day. RBP increases yield by 10%. Incremental milk production reaches 0.46 million tons in year 11 (financial benefits of Rs 1,417 million).
- (b) Reduced Feed/Production Cost: Average cost is Rs 16.70/kg. Feed is 70% of cost. RBP reduces feed cost by 5% conservatively. Annual financial benefits reach Rs 2,691 million in year 10.
- (c) Methane Emissions: Reduces methane by 10% for 2.7 million bovines. Methane reduction equals 0.16 tons of Certified Emission Reduction (CER) per animal, valued at Rs 122/animal/year. Reaches Rs 331 million by year 10.
- RBP Summary: Total annual incremental benefit is Rs 4,439 million (61% from cost reduction, 32% from increased yield, 7% from methane reduction). Shared between Zone 1 (53%) and Zone 2 (47%).

5. Component B - Milk Collection and Bulking (37% of costs):
- Benefits 1.3 million producers in 32,194 villages. Zone 1: 0.6 million producers in 20,297 villages (Producer Companies). Zone 2: 0.7 million producers in 11,897 villages (Dairy Cooperative Societies).
- Targets handling 5.3 million kg/day by year 6, with Bulk Milk Chilling (BMC) capacity of 1.7 million kg/day operating at 80% capacity (handling 0.5 million tons annually).
- (a) Increased Producer Price: Organised village collection increases producer prices by Rs 1 per kg. Reaches annual incremental benefits of Rs 2,048 million by handling 2 million tons of milk by year 6.
- (b) Reduced Transaction Costs: BMC cooling facility saves Rs 0.92 per kg in transport, operations, handling, and processing. Generates Rs 460 million annually from year 6.
- Summary Component B: Incremental annual financial gains of Rs 2,508 million (82% from price increase, 18% from transaction savings). Shared between Zone 1 (44%) and Zone 2 (56%).

6. Financial Analysis Results:
- Total project cost (with contingencies): Rs 20.4 billion.
- Undiscounted annual financial net benefits: Rs 13.1 billion (50% Breed Improvement, 31% Nutrition, 19% Bulking/Procurement).
- Financial Net Present Value (NPV): Rs 14.9 billion at 2011 prices over 20-year life.
- Financial Rate of Return (FRR):
  * Breed Improvement: 18.0%
  * Productivity Enhancement (Breed + Nutrition): 24.9%
  * Village Level Procurement/Bulking: 20.2%
  * Overall Project (including management): 22.1%

7. Economic Analysis Results:
- Economic project cost: Rs 18.4 billion (after netting taxes, subsidies, and transfers).
- Undiscounted annual net economic benefits: Rs 13.4 billion (88% Productivity Enhancement, 12% Milk Collection/Bulking).
- Economic Net Present Value (NPV): Rs 16.2 billion.
- Economic Rate of Return (ERR):
  * Breed Improvement: 20.5%
  * Productivity Enhancement (Breed + Nutrition): 29.0%
  * Village Level Procurement/Bulking: 14.3%
  * Overall Project: 23.5%

8. Sensitivity and Risk Analysis:
- Base Level ERR: 23.5%, NPV: Rs 16.2 billion.
- Sustainability Limitations (benefits limited to 75%):
  * Milk productivity limited to 75%: ERR drops to 22.1%, NPV drops to 13.5 billion.
  * Milk collection/bulking limited to 75%: ERR drops to 22.4%, NPV drops to 14.6 billion.
  * Nutrition management limited to 75%: ERR drops to 21.2%, NPV drops to 12.5 billion.
- Delayed Breed Improvement: 2-year delay drops ERR to 18.3%, NPV to 7.3 billion (55% reduction).
- Adverse Scenario (Costs at 125% & Benefits at 75%): ERR drops to 14.9%, NPV to 4.2 billion (74% reduction).
- Joint Risk Model Simulation: Costs allowed to increase up to 30%, benefits allowed to decrease up to 50%. ERR varies from 11.8% to 20.4% (coefficient of variation of 8%). Expected ERR is 16.2%, Expected NPV is Rs 5,761 million. Risk model predicts a 0.84 probability of ERR exceeding 15% and 0.2% probability of a negative NPV outcome.

9. Impact of Convergence & Productivity Growth:
- Average herd size: 2 livestock per household. 1.75 million households benefit.
- 39% of beneficiaries are in Zone 1 where all interventions converge. Annual financial benefits in Zone 1 are Rs 10 billion (Rs 14,585 per household). Zone 2 benefits are Rs 3.4 billion (Rs 3,250 per household).
- Annual Growth in Milk Productivity over 20-year project life:
  * Without Project (WOP) total growth: 2.20% (Genetic Improvement: 0.20%, AI/Nutrition: 2.00%)
  * With Project (WP) incremental growth: 2.11% (Genetic Improvement: 1.30%, AI Service: 0.26%, Nutrition Management: 0.55%)
  * Total Growth With Project: 4.31% (almost double). Composition of incremental 2.11% growth: Breed improvement (62%), AI services (12%), Nutrition (26%).

10. Reference Tables:
Table 1: NDSP Financial and Economic Summary (Rs Billion)
- Breed Improvement: Financial NPV 3.8, FRR 18.0% | Economic NPV 5.3, ERR 20.5%
- Breed + Nutrition: Financial NPV 12.6, FRR 24.9% | Economic NPV 16.5, ERR 29.0%
- Village Procurement: Financial NPV 3.4, FRR 20.2% | Economic NPV 0.8, ERR 14.3%
- Overall Project: Financial NPV 14.9, FRR 22.1% | Economic NPV 16.2, ERR 23.5%

Table 2: Sensitivity Summary
- Base Level: NPV 16.2 B, ERR 23.5%
- Milk productivity 75%: NPV 13.5 B, ERR 22.1%
- Milk bulking 75%: NPV 14.6 B, ERR 22.4%
- Nutrition 75%: NPV 12.5 B, ERR 21.2%
- Delayed genetics by 2 years: NPV 7.3 B, ERR 18.3%
- Costs 125% & Benefits 75%: NPV 4.2 B, ERR 14.9%

Table T-3: Risk Analysis Summary
- Expected Value: NPV 5,761 M, ERR 16.2%
- Standard Deviation: NPV 1,787 M, ERR 1.3%
- Minimum: NPV (230) M, ERR 11.8%
- Maximum: NPV 11,694 M, ERR 20.4%
- CV: NPV 0.31, ERR 0.08
- Probability of negative outcome: NPV 0.2%, ERR 0.0%

Table A-1: Impacts on Milk Productivity (kg/day/animal) for Zone 1
- Base Level: WOP 3.99 | PY-6 3.99 | PY-7 4.00 | PY-8 4.00 | PY-20 4.10
- Genetic Improvement: WOP 0.00 | PY-6 0.00 | PY-7 4.49 | PY-8 4.63 | PY-20 5.17
- AI Service: WOP 0.00 | PY-6 0.00 | PY-7 0.21 | PY-8 0.21 | PY-20 0.21
- Nutrition Management: WOP 0.00 | PY-6 0.41 | PY-7 0.45 | PY-8 0.45 | PY-20 0.45
- Total: WOP 3.99 | PY-6 4.62 | PY-7 5.16 | PY-8 5.30 | PY-20 5.84

Table A-2: Incremental Milk Production by Sources (Million tonnes)
- PY-1: Total 0.00
- PY-6: Genetics 0.00 | AI 0.10 | Nutrition 0.33 | Total 0.43
- PY-7: Genetics 0.02 | AI 0.12 | Nutrition 0.40 | Total 0.55
- PY-8: Genetics 0.08 | AI 0.12 | Nutrition 0.45 | Total 0.66
- PY-15: Genetics 1.61 | AI 0.12 | Nutrition 0.46 | Total 2.19
- PY-20: Genetics 1.99 | AI 0.12 | Nutrition 0.46 | Total 2.57

Table A-3: Reduced Inter-calving period due to AI in Zone 1
- Inter-calving period decreases by 14 days. Average yield is 78 kg per animal.
- PY-6: 1.28M animals calved | 0.10M tonnes | Rs 306.6M
- PY-8: 1.56M animals calved | 0.12M tonnes | Rs 372.8M
- PY-20: 1.56M animals calved | 0.12M tonnes | Rs 372.8M

Table A-4: Nutrition Management (RBP) Benefits
- Reduced Production cost: Rs 2,691 Million (61% share)
- Increased Milk yield: Rs 1,417 Million (32% share)
- Reduced Methane emission: Rs 331 Million (7% share)
- Total: Rs 4,439 Million (100% share)
"""

# --------------------------------------------------------------------------------
# Session state initialization
# --------------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --------------------------------------------------------------------------------
# Sidebar management
# --------------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown(
        "This assistant is preloaded with the comprehensive National Dairy Support Project (NDSP) data "
        "covering breed improvement, nutrition, bulking, economic/financial analyses, and risk structures."
    )
    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.chat_history = []
        st.rerun()

# --------------------------------------------------------------------------------
# Embeddings and Caching Setup
# --------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embed_model():
    return HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

Settings.embed_model = get_embed_model()
Settings.node_parser = SentenceSplitter(chunk_size=800, chunk_overlap=100)

@st.cache_resource(show_spinner=False)
def get_vector_index():
    # Construct an in-memory document structure from preloaded data string
    documents = [
        Document(
            text=NDSP_DATA,
            metadata={"file_name": "NDSP_Economic_Financial_Analysis_Model.pdf"}
        )
    ]
    chroma_client = chromadb.EphemeralClient()
    collection = chroma_client.get_or_create_collection(f"dairy_static_{st.session_state.session_id}")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    return VectorStoreIndex.from_documents(documents, vector_store=vector_store)

# --------------------------------------------------------------------------------
# LLM Binding — Groq Configuration
# --------------------------------------------------------------------------------
def configure_llm():
    from llama_index.llms.groq import Groq
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
    if not api_key:
        st.error(
            "No GROQ_API_KEY found. Add it under Streamlit Cloud → "
            "App settings → Secrets or set it as an environment variable."
        )
        st.stop()
    Settings.llm = Groq(model="llama-3.3-70b-versatile", api_key=api_key)

# --------------------------------------------------------------------------------
# Chat interface
# --------------------------------------------------------------------------------
GUARDRAIL_PROMPT = (
    "You are a dairy-sector planning assistant. Answer ONLY using the provided "
    "document context. If the answer is not clearly contained in the context, "
    "reply exactly: \"I don't have this information in the uploaded documents.\" "
    "Do not guess or use outside knowledge. Keep answers concise and, where "
    "relevant, mention which document section or table the information came from."
)

for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(content)

question = st.chat_input("Ask about breed improvement, nutrition, bulking impacts, or financial analysis...")

if question:
    configure_llm()
    st.session_state.chat_history.append(("user", question))
    
    with st.chat_message("user"):
        st.write(question)
        
    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant database sections..."):
            index = get_vector_index()
            query_engine = index.as_query_engine(
                similarity_top_k=4,
                system_prompt=GUARDRAIL_PROMPT,
            )
            try:
                response = query_engine.query(question)
                answer_text = str(response)
                st.write(answer_text)
                
                # Render source reference metadata
                sources = getattr(response, "source_nodes", [])
                if sources:
                    with st.expander("📎 Sources used for this answer"):
                        for i, node in enumerate(sources, start=1):
                            fname = node.node.metadata.get("file_name", "unknown file")
                            score = getattr(node, "score", None)
                            score_txt = f" (relevance: {score:.2f})" if score is not None else ""
                            st.markdown(f"**{i}. {fname}**{score_txt}")
                            st.caption(node.node.get_text()[:300] + "...")
            except Exception as e:
                answer_text = f"⚠️ Something went wrong: {e}"
                st.error(answer_text)
                
    st.session_state.chat_history.append(("assistant", answer_text))

# --------------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------------
st.divider()
st.caption(
    "Vocalkisan Dairy AI — Production RAG Engine. Data is parsed and contextualized within your browser session."
)
