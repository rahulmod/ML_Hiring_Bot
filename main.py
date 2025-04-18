import gradio as gr
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from typing import List, Dict
import os
import tempfile

# Initialize embeddings
embeddings = HuggingFaceEmbeddings()

# Initialize separate vector stores for resumes and culture docs
resume_store = Chroma(
    collection_name="resumes",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

culture_store = Chroma(
    collection_name="culture_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# Initialize LLM
llm = ChatGroq(
    api_key=os.environ["GROQ_API_KEY"],
    model_name="deepseek-r1-distill-llama-70b",
    temperature = 0,seed = 42
)

def process_candidate_submission(resume_file, job_description: str) -> str:
    # Load and process resume
    if resume_file.name.endswith('.pdf'):
        loader = PyPDFLoader(resume_file.name)
    else:
        loader = UnstructuredFileLoader(resume_file.name)
    
    resume_doc = loader.load()[0]
    
    # Create proper prompt template
    prompt_template = PromptTemplate(
        input_variables=["resume_text", "job_description"],
        template="""
        Given the following resume and job description, create a professional cold email to the candidate:
        
        Resume:
        {resume_text}
        
        Job Description:
        {job_description}
        
        Generate a concise, compelling cold email to the candidate that highlights the candidate's relevant skills and experience, how they align with the job requirements and company. Include a strong call-to-action.
        Ensure the email is well-structured, error-free, and tailored to the specific candidate and job description. Do not include any text apart from the email content.
        """
    )
    
    chain = LLMChain(
        llm=llm,
        prompt=prompt_template
    )
    
    response = chain.run({
        "resume_text": resume_doc.page_content,
        "job_description": job_description
    })
    
    return response

def store_culture_docs(culture_files: List[tempfile._TemporaryFileWrapper]) -> str:
    """Store company culture documentation in the vector store"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    
    all_docs = []
    for file in culture_files:
        if file.name.endswith('.pdf'):
            loader = PyPDFLoader(file.name)
        else:
            loader = UnstructuredFileLoader(file.name)
        docs = loader.load()
        splits = text_splitter.split_documents(docs)
        all_docs.extend(splits)
    
    culture_store.add_documents(all_docs)
    return f"Successfully stored {len(all_docs)} culture document chunks"

def store_resumes(resume_files: List[tempfile._TemporaryFileWrapper]) -> str:
    """Store resumes in the vector store with proper metadata"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    
    all_docs = []
    for file in resume_files:
        if file.name.endswith('.pdf'):
            loader = PyPDFLoader(file.name)
        else:
            loader = UnstructuredFileLoader(file.name)
        docs = loader.load()
        
        # Extract filename without extension as resume ID
        resume_id = os.path.splitext(os.path.basename(file.name))[0]
        
        # Add metadata to each chunk
        splits = text_splitter.split_documents(docs)
        for split in splits:
            split.metadata["resume_id"] = resume_id
            split.metadata["source"] = "resume"
        
        all_docs.extend(splits)
    
    resume_store.add_documents(all_docs)
    return f"Successfully stored {len(resume_files)} resumes"

def analyze_candidates(job_description: str) -> str:
    # First extract required skills from job description
    skills_prompt = PromptTemplate(
        input_variables=["job_description"],
        template="""
        Extract the key technical skills and requirements from this job description:
        
        {job_description}
        
        Return the skills as a comma-separated list.
        """
    )
    
    skills_chain = LLMChain(
        llm=llm,
        prompt=skills_prompt
    )
    
    skills = skills_chain.run({"job_description": job_description})
    
    # Get relevant culture documents based on job description
    relevant_culture_docs = culture_store.similarity_search(
        job_description,  # Using job description to find relevant culture aspects
        k=3  # Adjust based on how many culture chunks you want to consider
    )
    culture_context = "\n".join([doc.page_content for doc in relevant_culture_docs])
    
    # First analyze what cultural aspects we're looking for based on the role
    culture_requirements_prompt = PromptTemplate(
        input_variables=["job_description", "culture_docs"],
        template="""
        Based on this job description and our company culture documents, identify the key cultural attributes we should look for in candidates:

        Job Description:
        {job_description}

        Relevant Company Culture Context:
        {culture_docs}

        List the top 3-5 cultural attributes that would make someone successful in this role at our company:
        """
    )
    
    culture_req_chain = LLMChain(
        llm=llm,
        prompt=culture_requirements_prompt
    )
    
    cultural_requirements = culture_req_chain.run({
        "job_description": job_description,
        "culture_docs": culture_context
    })
    
    # Query resumes
    results = resume_store.similarity_search(
        job_description,
        k=10
    )
    
    # Group resume chunks by resume_id
    resume_groups = {}
    for doc in results:
        resume_id = doc.metadata.get("resume_id")
        if resume_id not in resume_groups:
            resume_groups[resume_id] = []
        resume_groups[resume_id].append(doc.page_content)
    
    # For each resume, compare against culture docs
    consolidated_analyses = []  # Initialize empty list for all analyses
    for resume_id, chunks in resume_groups.items():
        resume_text = "\n".join(chunks)
        
        # Compare this specific resume against culture docs
        culture_analysis_prompt = PromptTemplate(
            input_variables=["resume", "cultural_requirements"],
            template="""
            Analyze this candidate's potential culture fit based on their resume and company culture, both attached below.
            
            Resume:
            {resume}
            
            Key Cultural Requirements for this Role:
            {cultural_requirements}
            
            For this candidate, provide:
            1. Concise cultural fit assessment with pros and cons. Only provide a summary.
            2. Overall culture fit score (0-100%)
            3. Recommendation on cultural fit (e.g. "Strong fit", "Moderate fit", "Not a fit")
            4. Brief explanation of your recommendation.

            Review your response, ensure it is concise and within 200 words.
            """
        )
        
        culture_chain = LLMChain(
            llm=llm,
            prompt=culture_analysis_prompt
        )
        
        try:
            culture_fit = culture_chain.run({
                "resume": resume_text,
                "cultural_requirements": cultural_requirements
            })
            
            # Now analyze technical skills match
            skills_analysis_prompt = PromptTemplate(
                input_variables=["resume", "required_skills", "job_description"],
                template="""
                Analyze this candidate's technical skills match for the position, using the resume, required skills and job description provided below.
                
                Resume:
                {resume}
                
                Required Skills:
                {required_skills}
                
                Job Description:
                {job_description}
                
                For this candidate, provide:
                1. Concise skills match assessment with pros and cons. Only provide a summary.
                2. Overall skill match score (0-100%)
                3. Recommendation on skill fit (e.g. "Strong fit", "Moderate fit", "Not a fit")
                4. Brief explanation of your recommendation

                Review your response, ensure it is concise and within 200 words.
                """
            )
            
            skills_chain = LLMChain(
                llm=llm,
                prompt=skills_analysis_prompt
            )
            
            skills_fit = skills_chain.run({
                "resume": resume_text,
                "required_skills": skills,
                "job_description": job_description
            })

            # Create final recommendation
            final_recommendation_prompt = PromptTemplate(
                input_variables=["skills_analysis", "culture_analysis", "job_description"],
                template="""
                Provide a final hiring recommendation, using the job description, technical skills analysis, and culture fit analysis provided below.

                Job Description:
                {job_description}

                Technical Skills Analysis:
                {skills_analysis}

                Culture Fit Analysis:
                {culture_analysis}

                Provide your recommendation in the following format:

                FINAL HIRING RECOMMENDATION:
                Decision: [PROCEED / DO NOT PROCEED]

                Rationale:
                [Concise explanation of the recommendation]

                Review your final recommendation, be cut throat and make a data driven decision. For senior technical roles, give more importance to skills over culture fit.
                Ensure this response is concise and within 200 words.
                """
            )

            recommendation_chain = LLMChain(
                llm=llm,
                prompt=final_recommendation_prompt
            )

            final_recommendation = recommendation_chain.run({
                "skills_analysis": skills_fit,
                "culture_analysis": culture_fit,
                "job_description": job_description
            })

            # Append the analysis for this candidate to the consolidated analyses
            consolidated_analyses.append(f"""
            === Candidate Analysis (Resume ID: {resume_id}) ===
            
            CULTURE FIT ANALYSIS:
            {culture_fit}
            
            TECHNICAL SKILLS ANALYSIS:
            {skills_fit}

            HIRING RECOMMENDATION:
            {final_recommendation}
            
            ----------------------------------------
            """)
            
        except Exception as e:
            # If there's an error analyzing this candidate, add error message but continue with others
            consolidated_analyses.append(f"""
            === Candidate Analysis (Resume ID: {resume_id}) ===
            Error analyzing candidate: {str(e)}
            ----------------------------------------
            """)
            continue
    
    # Return all analyses joined together
    return "\n".join(consolidated_analyses)




def clear_databases():
    """Clear both resume and culture document databases"""
    global resume_store, culture_store
    
    status_messages = []
    
    # Clear resume store
    try:
        results = resume_store.get()
        if results and results['ids']:
            num_docs = len(results['ids'])
            resume_store._collection.delete(
                ids=results['ids']
            )
            status_messages.append(f"Cleared {num_docs} documents from resume database")
        else:
            status_messages.append("Resume database was already empty")
    except Exception as e:
        status_messages.append(f"Error clearing resume store: {e}")
        
    # Clear culture store
    try:
        results = culture_store.get()
        if results and results['ids']:
            num_docs = len(results['ids'])
            culture_store._collection.delete(
                ids=results['ids']
            )
            status_messages.append(f"Cleared {num_docs} documents from culture database")
        else:
            status_messages.append("Culture database was already empty")
    except Exception as e:
        status_messages.append(f"Error clearing culture store: {e}")
    
    return "\n".join(status_messages)




def create_interface():
    with gr.Blocks(theme='freddyaboulton/test-blue') as app:
        gr.Markdown("# AI Recruiter Assistant")
        
        with gr.Tabs():

            # Recruiter View
            with gr.Tab("Candidate Assessment"):
                gr.Markdown("Clear existing culture documents and resumes from storage. Use this every time you are uploading new company documentation or do not want to select from the existing pool of resumes.")
                clear_btn = gr.Button("Clear All Databases")
                clear_status = gr.Textbox(label="Clear Status")
                gr.Markdown("Use this feature to upload company culture documents. These documents will be used to assess the cultural fit of candidates.")
                with gr.Row():
                    culture_docs_upload = gr.File(
                        label="Upload Company Culture Documents",
                        file_count="multiple"
                    )
                    store_culture_btn = gr.Button("Store Culture Docs")
                    culture_status = gr.Textbox(label="Status")
                gr.Markdown("Use this feature to upload resumes in bulk. These resumes will be used to assess the technical skills and culture fit of candidates.")
                with gr.Row():
                    resume_bulk_upload = gr.File(
                        label="Upload Resumes",
                        file_count="multiple"
                    )
                    store_resumes_btn = gr.Button("Store Resumes")
                    resume_status = gr.Textbox(label="Status")
                
                with gr.Row():
                    job_desc_recruiter = gr.Textbox(
                        label="Paste Job Description",
                        lines=20
                    )
                with gr.Row():
                    analyze_btn = gr.Button("Analyze Candidates")
                with gr.Row():
                    analysis_output = gr.Textbox(
                        label="Analysis Results",
                        lines=30
                    )
                
                store_culture_btn.click(
                    store_culture_docs,
                    inputs=culture_docs_upload,
                    outputs=culture_status
                )
                
                store_resumes_btn.click(
                    store_resumes,
                    inputs=resume_bulk_upload,
                    outputs=resume_status
                )
                
                analyze_btn.click(
                    analyze_candidates,
                    inputs=job_desc_recruiter,
                    outputs=analysis_output
                )

                clear_btn.click(
                    clear_databases,
                    inputs=[],
                    outputs=clear_status
                )
             # Candidate View
            with gr.Tab("Cold Email Generator"):
                with gr.Row():
                    resume_upload = gr.File(label="Upload Resume")
                    job_desc_input = gr.Textbox(
                        label="Paste Job Description",
                        lines=10
                    )
                generate_btn = gr.Button("Generate Cold Email")
                email_output = gr.Textbox(
                    label="Generated Cold Email",
                    lines=10
                )
                
                generate_btn.click(
                    process_candidate_submission,
                    inputs=[resume_upload, job_desc_input],
                    outputs=email_output
                )
               
    return app

if __name__ == "__main__":
    app = create_interface()
    app.launch()
