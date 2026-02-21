"""
Streamlit Demo: Cost-Adaptive Hallucination Detection in Financial RAG
Run: streamlit run app.py
"""

import streamlit as st
import time
import json
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Financial RAG Hallucination Detection",
    page_icon="🔍",
    layout="wide"
)

# Cache model loading
@st.cache_resource
def load_rag():
    """Load RAG system"""
    from build_rag import FinancialRAG
    rag = FinancialRAG()
    rag.load('models/financial_rag')
    return rag

@st.cache_resource
def load_detectors():
    """Load all detectors"""
    from src.detection import create_detector
    return {
        'Token Overlap': create_detector('token_overlap'),
        'Semantic Similarity': create_detector('semantic_similarity'),
        'BERT NLI': create_detector('bert_nli'),
        'LLM Judge (Mock)': create_detector('llm_judge', backend='mock')
    }

def main():
    st.title("Cost-Adaptive Hallucination Detection")
    st.markdown("**Independent Study Project** | Financial RAG with RL-based Detector Selection")
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.markdown("""
    This system detects hallucinations in financial RAG responses 
    using 4 different methods with varying cost/accuracy tradeoffs.
    
    **Research Question:**  
    Can an RL agent learn when to use expensive vs cheap detection methods?
    """)
    
    st.sidebar.header("Detection Methods")
    st.sidebar.markdown("""
    | Method | Cost | Latency |
    |--------|------|---------|
    | Token Overlap | 0.05 | ~3ms |
    | Semantic Sim | 0.10 | ~40ms |
    | BERT NLI | 0.30 | ~50ms |
    | LLM Judge | 1.00 | ~500ms |
    """)
    
    # Load components
    with st.spinner("Loading RAG system and detectors..."):
        rag = load_rag()
        detectors = load_detectors()
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "RAG Query", 
        "Hallucination Detection", 
        "Method Comparison",
        "Dataset Info"
    ])
    
    # Tab 1: RAG Query
    with tab1:
        st.header("Financial Document Retrieval")
        st.markdown("Query SEC 10-K filings from Apple, Microsoft, Tesla, Amazon, and Google")
        
        query = st.text_input(
            "Enter your question:",
            placeholder="e.g., What was Apple's iPhone revenue in 2025?"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            k = st.selectbox("Results:", [1, 2, 3, 5], index=1)
        
        if query:
            results = rag.retrieve(query, k=k)
            
            if results:
                for i, r in enumerate(results):
                    with st.expander(f"Result {i+1}: {r['company']} ({r['filing_type']}) - Score: {r['score']:.3f}", expanded=(i==0)):
                        st.markdown(f"**Filing Date:** {r['filing_date']}")
                        st.markdown(f"**Relevance Score:** {r['score']:.4f}")
                        st.markdown("**Content:**")
                        st.text(r['text'][:500] + "..." if len(r['text']) > 500 else r['text'])
            else:
                st.warning("No results found.")
    
    # Tab 2: Hallucination Detection
    with tab2:
        st.header("Hallucination Detection Demo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input")
            
            question = st.text_area(
                "Question:",
                value="What was Apple's total revenue in fiscal year 2025?",
                height=80
            )
            
            response = st.text_area(
                "Response to check:",
                value="Apple's total revenue for fiscal year 2025 was $485.3 billion, representing a 22% increase.",
                height=100
            )
            
            detector_choice = st.selectbox(
                "Select detector:",
                list(detectors.keys())
            )
        
        with col2:
            st.subheader("Retrieved Context")
            
            if question:
                results = rag.retrieve(question, k=3)
                context = [r['text'] for r in results]
                
                for i, r in enumerate(results[:2]):
                    st.markdown(f"**Source {i+1}:** {r['company']} 10-K")
                    st.text(r['text'][:200] + "...")
        
        if st.button("Detect Hallucination", type="primary"):
            if question and response:
                detector = detectors[detector_choice]
                
                start = time.time()
                result = detector.detect(question, response, context)
                elapsed = (time.time() - start) * 1000
                
                st.markdown("---")
                
                if result.is_hallucinated:
                    st.error(f"**HALLUCINATION DETECTED** (Confidence: {result.confidence:.1%})")
                else:
                    st.success(f"**GROUNDED** (Confidence: {result.confidence:.1%})")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Confidence", f"{result.confidence:.1%}")
                col2.metric("Latency", f"{result.latency_ms:.1f}ms")
                col3.metric("Method Cost", f"{detector.cost}")
                
                if result.explanation:
                    st.info(f"**Explanation:** {result.explanation}")
    
    # Tab 3: Method Comparison
    with tab3:
        st.header("Compare All Detection Methods")
        
        st.markdown("Test all 4 methods on the same input to see cost-accuracy tradeoffs.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            test_question = st.text_area(
                "Question:",
                value="What was Apple's iPhone revenue in 2025?",
                height=80,
                key="compare_q"
            )
        
        with col2:
            grounded_resp = st.text_area(
                "Grounded Response:",
                value="Apple generated $209.6 billion in iPhone revenue for fiscal year 2025.",
                height=80
            )
            
            halluc_resp = st.text_area(
                "Hallucinated Response:",
                value="Apple's iPhone revenue reached $245.8 billion in 2025.",
                height=80
            )
        
        if st.button("Run All Detectors", type="primary"):
            results_data = rag.retrieve(test_question, k=3)
            context = [r['text'] for r in results_data]
            
            st.markdown("---")
            st.subheader("Results")
            
            # Create results table
            results_table = []
            
            for name, detector in detectors.items():
                r1 = detector.detect(test_question, grounded_resp, context)
                r2 = detector.detect(test_question, halluc_resp, context)
                
                results_table.append({
                    "Method": name,
                    "Grounded": "Halluc" if r1.is_hallucinated else "OK",
                    "G Correct": "Yes" if not r1.is_hallucinated else "No",
                    "Hallucinated": "Halluc" if r2.is_hallucinated else "OK", 
                    "H Correct": "Yes" if r2.is_hallucinated else "No",
                    "Latency (ms)": f"{r1.latency_ms:.1f}",
                    "Cost": detector.cost
                })
            
            st.dataframe(results_table, use_container_width=True)
            
            # Summary
            correct_count = sum(1 for r in results_table if r["G Correct"] == "Yes" and r["H Correct"] == "Yes")
            st.metric("Methods with Both Correct", f"{correct_count}/4")
    
    # Tab 4: Dataset Info
    with tab4:
        st.header("Dataset Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("RAG Index")
            st.metric("Document Chunks", "975")
            st.metric("Embedding Dimension", "384")
            st.metric("Model", "all-MiniLM-L6-v2")
            
            st.markdown("**Companies:**")
            st.markdown("- Apple\n- Amazon\n- Google\n- Microsoft\n- Tesla")
        
        with col2:
            st.subheader("Test Dataset")
            
            # Load test dataset stats
            dataset_path = Path("data/processed/hallucination_test_dataset.json")
            if dataset_path.exists():
                with open(dataset_path) as f:
                    dataset = json.load(f)
                
                st.metric("Test Examples", len(dataset['examples']))
                st.metric("Total Predictions", len(dataset['examples']) * 2)
                
                # Count by type
                types = {}
                for ex in dataset['examples']:
                    t = ex.get('hallucination_type', 'unknown')
                    types[t] = types.get(t, 0) + 1
                
                st.markdown("**Hallucination Types:**")
                for t, count in sorted(types.items(), key=lambda x: -x[1]):
                    st.markdown(f"- {t.replace('_', ' ').title()}: {count}")
            else:
                st.warning("Test dataset not found. Run create_test_dataset.py")
        
        st.markdown("---")
        st.subheader("Research Problem")
        st.markdown("""
        **Goal:** Train an RL agent to select the optimal hallucination detection method 
        based on query characteristics.
        
        **Insight:** Cheap methods (Token Overlap) work well for numerical hallucinations,
        but complex cases need expensive methods (LLM Judge).
        
        **Approach:** Use PPO to learn a policy that minimizes:
        ```
        Loss = -accuracy_reward + lambda * cost_penalty
        ```
        """)


if __name__ == "__main__":
    main()
