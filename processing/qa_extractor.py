from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.settings import OPENAI_API_KEY

def extract_questions_answers(docs, chunk_size=5000):
    llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=200 
    )
    
    qa_prompt = """You are an advanced AI specialized in deep text comprehension, logical inference, and structured 
    knowledge extraction. Your task is to meticulously analyze the provided content and generate an extensive list of 
    well-formed Q&A pairs that cover every possible dimension of understanding. Extract every possible fact, concept, 
    principle, definition, mechanism, process, analogy, and implication from the text. Explore hidden patterns, logical 
    implications, and nuanced interpretations. Consider counterfactuals and comparative analysis. Expand upon every 
    extracted fact by generating nested follow-up questions that explore different angles. Introduce hypothetical 
    scenarios to trigger deeper reasoning. Generate reverse questions. Integrate real-world examples, case studies, 
    analogies, and historical perspectives into answers. Ensure that each question is direct and unambiguous while the 
    answer remains concise, factual, and well-structured. Avoid redundancy. Now, extract Q&A pairs from the following 
    text, covering every conceivable angle, depth, and logical interpretation. text:\n\n{text}\n\nFormat: Q: <question> A: <answer>"""
    
    structured_data = []
    for doc in docs:
        text_chunks = text_splitter.split_text(doc.page_content)
        
        for chunk in text_chunks:
            response = llm.predict(qa_prompt.format(text=chunk))
            qa_pairs = response.split("\n\n")
            
            for qa in qa_pairs:
                if "Q:" in qa and "A:" in qa:
                    parts = qa.split("A:")
                    question = parts[0].replace("Q:", "").strip()
                    answer = parts[1].strip()
                    structured_data.append({"question": question, "answer": answer})
    
    return structured_data
