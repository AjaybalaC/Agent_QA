import streamlit as st
from src.document_processor import DocumentProcessor
from src.gemini_agent import GeminiAgent

def main():
    st.set_page_config(page_title="Document QA Debugger", page_icon="🔍")
    st.title("🕵 Document QA Agent - Debugging Mode")

    if "processor" not in st.session_state:
        st.session_state.processor = DocumentProcessor()
    if "gemini_agent" not in st.session_state:
        st.session_state.gemini_agent = GeminiAgent()

    processor = st.session_state.processor
    agent = st.session_state.gemini_agent
    if not hasattr(agent, 'model') or agent.model is None:
        st.error("Gemini Agent failed to initialize. Check API key and configuration in the sidebar.")
        return

    # Button to list Qdrant Cloud contents
    if st.button("List Qdrant Cloud Contents"):
        agent.list_qdrant_points()

    uploaded_file = st.file_uploader(
        "Upload Document",
        type=['pdf', 'csv'],
        help="Supports PDF and CSV files for analysis"
    )

    query = st.text_area("📝 Enter your specific document query:", height=100)

    if uploaded_file is not None and query:
        try:
            if uploaded_file.type == 'application/pdf':
                document_data = processor.read_pdf(uploaded_file)
            elif uploaded_file.type == 'text/csv':
                document_data = processor.read_csv(uploaded_file)
            else:
                st.warning("Unsupported file type")
                return

            if not document_data:
                st.warning("Could not extract data from the document")
                return

            # if isinstance(document_data, dict):
            #     st.write(f"Debug - Text Length: {len(document_data['text'])} characters")
            #     st.text_area("Debug - Full Document Text", document_data['text'], height=200)
            #     st.text_area("Debug - JSON Data", document_data['json'], height=200)
            # else:
            #     st.write(f"Debug - Document Length: {len(document_data)} characters")
            #     st.text_area("Debug - Full Document Text", document_data, height=200)

            if st.button("🚀 Analyze Document", type="primary"):
                with st.spinner('Analyzing document...'):
                    if "result" in st.session_state:
                        del st.session_state.result
                    response = agent.analyze_document(document_data, query, processor)
                    st.session_state.result = response
                    st.subheader("📄 Analysis Result")
                    st.markdown(st.session_state.result)

        except Exception as e:
            st.error(f"Document Processing Error: {e}")