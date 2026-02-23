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

    layout="wide",
    initial_sidebar_state="expanded"
)

# Example questions from PHANTOM dataset
EXAMPLE_QUESTIONS = {
    "Revenue Query": "What was the company's total revenue?",
    "Growth Metrics": "What was the revenue growth rate?",
    "Profitability": "What was the operating income?",
    "Market Performance": "What were the key financial highlights?",
    "Business Segments": "What are the main business segments?"
}

# Will be loaded from PHANTOM dataset dynamically
EXAMPLE_PAIRS = {}

# Cache model loading
@st.cache_resource
def load_rag():
    """Load RAG system with error handling"""
    try:
        from build_rag import FinancialRAG
        rag = FinancialRAG()
        rag.load('models/financial_rag')
        return rag, None
    except Exception as e:
        return None, str(e)

@st.cache_resource
def load_phantom_examples():
    """Load examples from PHANTOM dataset with original context"""
    try:
        from datasets import load_dataset
        import pandas as pd
        
        print("Loading PHANTOM dataset examples...")
        phantom = load_dataset('seyled/Phantom_Hallucination_Detection',
                               data_files='PhantomDataset/Phantom_10k_seed.csv')
        df = phantom['train'].to_pandas()
        
        # PHANTOM has natural pairs: same query/context, different answers with different labels
        # Find queries with both hallucinated and grounded answers
        query_counts = df.groupby('query').size()
        paired_queries = query_counts[query_counts == 2].index[:5]  # Get first 5 paired queries
        
        examples = {}
        for i, query in enumerate(paired_queries):
            pair_rows = df[df['query'] == query]
            
            grounded_row = pair_rows[pair_rows['ground_truth_label'] == 'not hallucination'].iloc[0]
            hallucinated_row = pair_rows[pair_rows['ground_truth_label'] == 'hallucination'].iloc[0]
            
            example_name = f"Pair {i+1}"
            examples[example_name] = {
                "question": query,
                "context": [grounded_row['context']],  # Both rows have same context
                "grounded": grounded_row['answer'],
                "hallucinated": hallucinated_row['answer']
            }
        
        return examples, None
    except Exception as e:
        # Fallback examples if PHANTOM fails to load
        return {
            "Fallback": {
                "question": "What was the company's revenue?",
                "context": ["The company reported total revenue of $100 million for fiscal year 2025."],
                "grounded": "The company's revenue was $100 million.",
                "hallucinated": "The company's revenue was $500 billion."
            }
        }, str(e)

@st.cache_resource
def load_detectors():
    """Load all detectors with error handling"""
    from src.detection import create_detector
    detectors = {}
    errors = {}
    
    # Load each detector individually with error handling
    detector_configs = [
        ('Token Overlap', 'token_overlap', {}),
        ('Semantic Similarity', 'semantic_similarity', {}),
        ('BERT NLI', 'bert_nli', {}),
        ('LLM Judge (Qwen2.5-7B)', 'llm_judge', {'backend': 'ollama', 'model': 'qwen2.5:7b'}),
    ]
    
    for display_name, detector_name, kwargs in detector_configs:
        try:
            detectors[display_name] = create_detector(detector_name, **kwargs)
        except Exception as e:
            errors[display_name] = str(e)
    
    return detectors, errors


def main():
    st.title("Financial RAG Hallucination Detection")
    st.markdown("*Adaptive strategy selection for detecting hallucinations in financial document Q&A*")
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This system detects hallucinations in financial RAG responses 
        using multiple detection methods with different cost/accuracy tradeoffs.
        """)
        
        st.header("Detection Methods")
        st.markdown("""
        | Method | Cost | Speed |
        |--------|------|-------|
        | Token Overlap | Low | Very Fast |
        | Semantic Sim | Low | Fast |
        | BERT NLI | Medium | Slow |
        | LLM Judge | High | Very Slow |
        """)
        
        st.header("Research Goal")
        st.info("Train an RL agent to select the optimal detector based on query characteristics")
        
        st.header("Quick Start")
        st.markdown("""
        1. Go to **Quick Test** tab
        2. Click an example or enter your own
        3. Click **Run Detection**
        4. See which detector works best!
        """)
    
    # Load components with error handling
    with st.spinner("Loading RAG system, detectors, and PHANTOM examples..."):
        rag, rag_error = load_rag()
        detectors, detector_errors = load_detectors()
        example_pairs, phantom_error = load_phantom_examples()
    
    # Show errors if any
    if rag_error:
        st.error(f"Failed to load RAG system: {rag_error}")
        st.info("Make sure you've run: `python build_rag.py` to create the index")
        st.stop()
    
    if phantom_error:
        st.warning(f"Could not load PHANTOM examples: {phantom_error}")
        st.info("Using fallback examples. Install datasets: `pip install datasets`")
    
    if detector_errors:
        with st.expander("Some detectors failed to load (click to expand)", expanded=False):
            for name, error in detector_errors.items():
                st.warning(f"**{name}:** {error}")
                if "ollama" in error.lower():
                    st.info(f"To use {name}, install Ollama and run: `ollama pull qwen2.5:7b`")
    
    if not detectors:
        st.error("No detectors available. Please check your installation.")
        st.stop()
    
    st.success(f"Loaded {len(detectors)} detector(s) successfully")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["Quick Test", "RAG Query", "Compare All"])
    
    # ==================== TAB 1: Quick Test ====================
    with tab1:
        st.header("Quick Hallucination Detection Test")
        st.markdown("*Test a response against retrieved financial documents*")
        
        # Example buttons
        st.markdown("**Try an example:**")
        example_cols = st.columns(min(len(example_pairs), 5))  # Max 5 columns
        example_key = None
        for i, name in enumerate(list(example_pairs.keys())[:5]):  # Show first 5 examples
            if example_cols[i].button(f"Example {i+1}", key=f"ex_{i}", use_container_width=True):
                example_key = name
        
        # Initialize session state on first load
        if 'q_input' not in st.session_state:
            st.session_state.q_input = ""
            st.session_state.g_input = ""
            st.session_state.h_input = ""
            st.session_state.phantom_context = None
        
        # Update session state only when example button clicked
        if example_key:
            ex = example_pairs[example_key]
            st.session_state.q_input = ex['question']
            st.session_state.g_input = ex['grounded']
            st.session_state.h_input = ex['hallucinated']
            st.session_state.phantom_context = ex.get('context', None)
        
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Input")
            
            question = st.text_area(
                "Question:",
                height=80,
                placeholder="e.g., What was Apple's total revenue in 2025?",
                key="q_input"
            )
            
            grounded = st.text_area(
                "Grounded Response (should pass):",
                height=90,
                placeholder="A response that accurately reflects the source documents",
                key="g_input"
            )
            
            hallucinated = st.text_area(
                "Hallucinated Response (should fail):",
                height=90,
                placeholder="A response with fabricated or incorrect information",
                key="h_input"
            )
            
            detector_choice = st.selectbox(
                "Select detector:",
                list(detectors.keys()),
                help="Choose which method to use for detection"
            )
        
        with col2:
            st.subheader("Context")
            
            # Use PHANTOM context if available, otherwise use RAG
            if st.session_state.get('phantom_context'):
                context = st.session_state.phantom_context
                st.info("Using PHANTOM's original context")
                with st.expander("PHANTOM Context", expanded=True):
                    st.text(context[0][:500] + "..." if len(context[0]) > 500 else context[0])
            elif question:
                with st.spinner("Retrieving documents from SEC filings..."):
                    results = rag.retrieve(question, k=3)
                    context = [r['text'] for r in results]
                
                st.success(f"Found {len(results)} relevant chunks")
                
                for i, r in enumerate(results[:2]):
                    with st.expander(f"{r['company']} 10-K ({r['filing_date']}) - Score: {r['score']:.3f}", expanded=(i==0)):
                        st.text(r['text'][:400] + "..." if len(r['text']) > 400 else r['text'])
            else:
                st.info("Enter a question or select an example")
        
        st.markdown("---")
        
        # Detection button
        if st.button("Run Detection", type="primary", use_container_width=True):
            if not question:
                st.error("Please enter a question")
            elif not (grounded or hallucinated):
                st.error("Please enter at least one response to test")
            else:
                detector = detectors[detector_choice]
                
                st.subheader("Results")
                col1, col2 = st.columns(2)
                
                # Test grounded response
                if grounded:
                    with col1:
                        with st.spinner("Testing grounded response..."):
                            result = detector.detect(question, grounded, context)
                        
                        st.markdown("**Grounded Response**")
                        if result.is_hallucinated:
                            st.error(f"INCORRECTLY flagged as hallucination")
                            st.caption(f"Confidence: {result.confidence:.1%}")
                        else:
                            st.success(f"CORRECTLY identified as grounded")
                            st.caption(f"Confidence: {result.confidence:.1%}")
                        
                        st.metric("Latency", f"{result.latency_ms:.1f}ms")
                        
                        if result.explanation:
                            with st.expander("Explanation"):
                                st.write(result.explanation)
                
                # Test hallucinated response
                if hallucinated:
                    with col2:
                        with st.spinner("Testing hallucinated response..."):
                            result = detector.detect(question, hallucinated, context)
                        
                        st.markdown("**Hallucinated Response**")
                        if result.is_hallucinated:
                            st.success(f"CORRECTLY detected hallucination")
                            st.caption(f"Confidence: {result.confidence:.1%}")
                        else:
                            st.error(f"MISSED the hallucination")
                            st.caption(f"Confidence: {result.confidence:.1%}")
                        
                        st.metric("Latency", f"{result.latency_ms:.1f}ms")
                        
                        if result.explanation:
                            with st.expander("Explanation"):
                                st.write(result.explanation)
    
    # ==================== TAB 2: RAG Query ====================
    with tab2:
        st.header("Financial Document Search")
        st.markdown("*Query SEC 10-K filings from Apple, Microsoft, Tesla, Amazon, and Google*")
        
        # Quick examples
        st.markdown("**Quick examples:**")
        ex_cols = st.columns(len(EXAMPLE_QUESTIONS))
        selected_q = None
        for i, (name, q) in enumerate(EXAMPLE_QUESTIONS.items()):
            if ex_cols[i].button(name, key=f"rag_ex_{i}", use_container_width=True):
                selected_q = q
        
        st.markdown("---")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input(
                "Enter your question:",
                value=selected_q if selected_q else "",
                placeholder="e.g., What was Apple's iPhone revenue in 2025?"
            )
        with col2:
            k = st.selectbox("Results:", [1, 2, 3, 5, 10], index=2)
        
        if query:
            with st.spinner("Searching..."):
                results = rag.retrieve(query, k=k)
            
            if results:
                st.success(f"Found {len(results)} relevant document chunks")
                
                for i, r in enumerate(results):
                    with st.expander(
                        f"{i+1}. {r['company']} {r['filing_type']} ({r['filing_date']}) - Relevance: {r['score']:.4f}",
                        expanded=(i==0)
                    ):
                        st.markdown(f"**Company:** {r['company']}")
                        st.markdown(f"**Filing:** {r['filing_type']} - {r['filing_date']}")
                        st.markdown(f"**Relevance Score:** {r['score']:.4f}")
                        st.markdown("**Content:**")
                        st.text_area("", r['text'], height=200, key=f"content_{i}", label_visibility="collapsed")
            else:
                st.warning("No results found")
        else:
            st.info("Enter a question or click an example")
    
    # ==================== TAB 3: Compare All ====================
    with tab3:
        st.header("Compare All Detection Methods")
        st.markdown("*Test all available detectors side-by-side*")
        
        # Example selector
        st.markdown("**Load an example:**")
        comp_cols = st.columns(min(len(example_pairs), 5))  # Max 5 columns
        selected_example = None
        for i, name in enumerate(list(example_pairs.keys())[:5]):  # First 5 examples
            if comp_cols[i].button(f"Example {i+1}", key=f"comp_ex_{i}", use_container_width=True):
                selected_example = name
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if selected_example:
                ex = example_pairs[selected_example]
                test_q = st.text_area("Question:", value=ex['question'], height=80)
                grounded_r = st.text_area("Grounded:", value=ex['grounded'], height=90)
            else:
                test_q = st.text_area("Question:", value="What was Apple's total revenue in fiscal year 2025?", height=80)
                grounded_r = st.text_area("Grounded:", value="Apple's total net sales for fiscal year 2025 were $416.2 billion.", height=90)
        
        with col2:
            if selected_example:
                ex = example_pairs[selected_example]
                halluc_r = st.text_area("Hallucinated:", value=ex['hallucinated'], height=90)
            else:
                halluc_r = st.text_area("Hallucinated:", value="Apple's total revenue for fiscal year 2025 was $485.3 billion.", height=90)
            
            st.info(f"**Available detectors:** {len(detectors)}")
        
        if st.button("Run All Detectors", type="primary", use_container_width=True):
            if not test_q:
                st.error("Please enter a question")
            elif not (grounded_r or halluc_r):
                st.error("Please enter at least one response")
            else:
                # Use PHANTOM context if example selected, otherwise RAG
                if selected_example and 'context' in example_pairs[selected_example]:
                    ctx = example_pairs[selected_example]['context']
                    st.info("Using PHANTOM's original context")
                else:
                    with st.spinner("Retrieving context from SEC filings..."):
                        ctx_results = rag.retrieve(test_q, k=3)
                        ctx = [r['text'] for r in ctx_results]
                    st.info(f"Retrieved {len(ctx)} chunks from SEC filings")
                
                st.markdown("---")
                st.subheader("Comparison Results")
                
                results_table = []
                progress = st.progress(0)
                
                for idx, (name, det) in enumerate(detectors.items()):
                    with st.spinner(f"Testing {name}..."):
                        r_g = det.detect(test_q, grounded_r, ctx) if grounded_r else None
                        r_h = det.detect(test_q, halluc_r, ctx) if halluc_r else None
                        
                        g_res = "N/A"
                        g_check = ""
                        if r_g:
                            g_res = "Flagged" if r_g.is_hallucinated else "OK"
                            g_check = "FAIL" if r_g.is_hallucinated else "PASS"
                        
                        h_res = "N/A"
                        h_check = ""
                        if r_h:
                            h_res = "Flagged" if r_h.is_hallucinated else "Missed"
                            h_check = "PASS" if r_h.is_hallucinated else "FAIL"
                        
                        avg_lat = 0
                        if r_g and r_h:
                            avg_lat = (r_g.latency_ms + r_h.latency_ms) / 2
                        elif r_g:
                            avg_lat = r_g.latency_ms
                        elif r_h:
                            avg_lat = r_h.latency_ms
                        
                        results_table.append({
                            "Method": name,
                            "Grounded": g_res,
                            "G-Check": g_check,
                            "Hallucinated": h_res,
                            "H-Check": h_check,
                            "Latency": f"{avg_lat:.1f}ms",
                            "Cost": f"{det.cost:.2f}"
                        })
                    
                    progress.progress((idx + 1) / len(detectors))
                
                progress.empty()
                
                st.dataframe(results_table, use_container_width=True, hide_index=True)
                
                # Summary
                col1, col2, col3 = st.columns(3)
                
                g_ok = sum(1 for r in results_table if r["G-Check"] == "PASS")
                h_ok = sum(1 for r in results_table if r["H-Check"] == "PASS")
                both_ok = sum(1 for r in results_table if r["G-Check"] == "PASS" and r["H-Check"] == "PASS")
                
                col1.metric("Grounded Correct", f"{g_ok}/{len(detectors)}")
                col2.metric("Hallucination Correct", f"{h_ok}/{len(detectors)}")
                col3.metric("Both Correct", f"{both_ok}/{len(detectors)}")
                
                # Insights
                st.markdown("---")
                if both_ok == len(detectors):
                    st.success("All detectors performed perfectly!")
                elif both_ok > 0:
                    st.info(f"{both_ok}/{len(detectors)} detectors got both correct")
                else:
                    st.warning("This is a challenging case - no detector got both correct")


if __name__ == "__main__":
    main()
