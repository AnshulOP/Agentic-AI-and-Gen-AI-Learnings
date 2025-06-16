from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
import faiss
import re
from dotenv import load_dotenv

from langchain_tavily import TavilySearch
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough, RunnableLambda

from rag_pipeline import RAG
from schemas import AgentState, TopicClassification, ResponseValidator

class WorkflowNodes:

    def __init__(self):
        load_dotenv()
        rag = RAG() 
        self.retriever = rag.get_retriever() 
        self.llm = ChatGroq(model="mistral-saba-24b")

    def supervisor(self, state: AgentState):

        parser = PydanticOutputParser(pydantic_object = TopicClassification)
        question = state["messages"][0]

        print("=========================================== [ Input Received by Supervisor ] ===========================================")
        print(f"User Question : {question}")
        
        template = """
        You are an intelligent classifier. Your task is to classify the following user query into one of the following categories:
        - Data Structures
        - General Query
        - Web Search

        Guidelines:
        - Choose "Data Structures" if the query is about linked lists, trees, stacks, queues, graphs, or any other data structure topic.
        - Choose "General Query" if the query is not about data structures but can be answered using general knowledge.
        - Choose "Web Search" if the query is about real-time data, current events, or requires up-to-date information not available in your current knowledge.
        
        User Query: {question}
        Format Instruction: {format_instructions}
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["question"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        chain = prompt | self.llm | parser
        
        response = chain.invoke({"question": question})
        print(f"Classified Topic : {response.topic}")
        print("=========================================== [ Supervision Complete ] ===========================================\n")
        return {
            "messages": [response.topic], 
            "original_question": question, 
            "retry_count": state.get("retry_count", 0)
        }
    
    def rag_call(self, state : AgentState):

        print("=========================================== [ RAG Call Execution ] ===========================================")

        question = state.get("original_question", state["messages"][0])
        retry_count = state.get("retry_count", 0)
        validation_feedback = state.get("validation_feedback", "")

        def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
        
        if retry_count > 0:
            print(f"RAG Retry {retry_count}...........")

            template = """
            You are a helpful and structured AI assistant. Your task is to answer the user's question **using only the information provided in the context**.

            ⚠️ NOTE: This is a retry attempt. Your previous response was rejected due to the following issue:
            FEEDBACK: {feedback}

            You must now revise your response to address the above feedback **and strictly follow the required format**
            
            Context: {context}
            Question: {question}
            """

            prompt = PromptTemplate(
                template=template,
                input_variables=['context', 'question', 'feedback']
            )

            rag_chain = (
            RunnableParallel({
                "context": RunnableLambda(lambda x: x["question"]) | self.retriever | RunnableLambda(format_docs),
                "question": RunnableLambda(lambda x: x["question"]),
                "feedback": RunnableLambda(lambda x: x["feedback"]),
            }) 
            | prompt 
            | self.llm
            | StrOutputParser()
)

            response = rag_chain.invoke({
                "question": question,
                "feedback": validation_feedback
            })

            print(f"RAG Output after Retry {retry_count} : {response}")
            print("=========================================== [ RAG Call Execution Completed ] ===========================================\n")
 
        else:
            print("First RAG Try...........")

            template = """
            You are a helpful AI assistant.
            Answer ONLY from the provided transcript context.
            If the context is insufficient, just say you don't know.

            Context: {context}
            Question: {question}
            """
            
            prompt = PromptTemplate(
                template=template,
                input_variables=['context', 'question']
            )
            
            rag_chain = (
                {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )
            
            response = rag_chain.invoke(question)
            print(f"RAG Output after First Try : {response}")
            print("=========================================== [ RAG Call Execution Completed ] ===========================================\n")

        return {
            'messages': [response],
            'last_node': 'RAG',
            'retry_count': retry_count
        }
    
    def llm_call(self, state: AgentState):

        print("=========================================== [ LLM Call Execution ] ===========================================")

        question = state.get("original_question", state["messages"][0])
        retry_count = state.get("retry_count", 0)
        validation_feedback = state.get("validation_feedback", "")

        if retry_count > 0:

            print(f"LLM Retry {retry_count}...........")
            template = """
            You are a helpful and structured AI assistant. Your task is to answer the user's question **using your general knowledge**.

            NOTE: This is a retry attempt. Your previous response was rejected due to the following issue:
            FEEDBACK: {feedback}

            You must now revise your response to address the above feedback **and strictly follow the required format** outlined below.

            REQUIRED RESPONSE FORMAT:

            **Definition/Overview:**  
            [Begin with a clear and concise definition or high-level overview of the concept.]

            **Key Points:**  
            1. [First key point — explain with relevant detail.]  
            2. [Second key point — explain with relevant detail.]  
            3. [Third key point — explain with relevant detail.]

            **Example/Details:**  
            [Provide one or more specific examples, explanations, or implementations to support the key points.]

            **Conclusion:**  
            [Wrap up the response with a brief and meaningful summary.]
            
            Question: {question}
            """

            prompt = PromptTemplate(
                template=template,
                input_variables=['question', 'feedback']
            )

            # Fixed: Create a proper chain that handles both question and feedback
            llm_chain = (
                {
                    "question": lambda x: question,
                    "feedback": lambda x: validation_feedback
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )

            response = llm_chain.invoke({})

            print(f"LLM Output after Retry {retry_count} : {response}")
            print("=========================================== [ LLM Call Execution Completed ] ===========================================\n")
        else:
            print("First LLM Try...........")
            prompt = PromptTemplate(
                template="""
                You are a knowledgeable assistant with access to real-world facts (up to your knowledge cutoff).
                Answer the following user question accurately and concisely based on your existing knowledge.

                User Question: {question}
                """,
                input_variables=["question"],
            )
            
            llm_chain = prompt | self.llm | StrOutputParser()

            response = llm_chain.invoke({"question": question})

            print(f"LLM Output after Retry {retry_count} : {response}")
            print("=========================================== [ LLM CAll Execution Completed ] ===========================================\n")
        
        return {
            "messages": [response],
            "last_node": "LLM",
            "retry_count": retry_count
        }
    
    def web_call(self, state : AgentState):
    
        print("=========================================== [ Web Crawler Execution ] ===========================================")

        question = state.get("original_question", state["messages"][0])

        def clean_text(text):
            # Remove URLs
            text = re.sub(r'http[s]?://\S+', '', text)
            text = re.sub(r'www\.\S+', '', text)

            # Remove markdown links [text](link) → text
            text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

            # Remove unwanted characters (retain basic punctuation)
            text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)

            # Replace multiple dots (...) with a single dot
            text = re.sub(r'\.{3,}', '.', text)

            # Normalize whitespace and newlines
            text = re.sub(r'\n\s*\n', '\n\n', text)
            text = re.sub(r'\s+', ' ', text)            
            text = text.strip()                         

            return text

        search = TavilySearch(max_results = 5, include_answer=True)

        results = search.invoke(question)
        print("Scraping content from web.............")
        combined_content = "\n\n".join(clean_text(res['content']) for res in results['results'])

        # Convert to PromptTemplate
        consolidation_prompt = PromptTemplate(
            template="""
            Based on the following search results, provide a comprehensive, consolidated answer to: {question}
            
            Search Results:
            {search_results}

            Provide the final consolidated content only.
            """,
            input_variables=["question", "search_results"]
        )

        # Create chain and invoke
        chain = consolidation_prompt | self.llm | StrOutputParser()
        
        print("Passing content to LLM for consolidated response...........")
        response = chain.invoke({
            "question": question,
            "search_results": combined_content
        })
        
        print(f"Final Output for Web Query : {response}")
        print("=========================================== [ Web Crawler Execution Completed ] ===========================================\n")
        return {"messages" : [response]}
    
    def validator(self, state: AgentState):
    
        print("=========================================== [ Output Validation Started ] ===========================================")
        
        response = state["messages"][-1]
        retry_count = state.get("retry_count", 0)
        valid_parser = PydanticOutputParser(pydantic_object = ResponseValidator)
        
        validation_template = """
        You are a VERY STRICT response validator. Your task is to check if the response EXACTLY follows the required format structure.

        REQUIRED FORMAT STRUCTURE (ALL SECTIONS MUST BE PRESENT):
        The response MUST contain ALL of these sections with their exact headers:

        1. **Definition/Overview:** (must start with this exact header)
        2. **Key Points:** (must have this header followed by numbered or bulleted list)
        3. **Example/Details:** (must have this header with concrete examples)
        4. **Conclusion:** (must have this header with a summary)

        VALIDATION CHECKLIST:
        Does it have "**Definition/Overview:**" header?
        Does it have "**Key Points:**" header with numbered/bulleted points?
        Does it have "**Example/Details:**" header with examples?
        Does it have "**Conclusion:**" header with summary?
        Is it at least 100 words long?
        
        IMPORTANT: If ANY of these sections are missing or incorrectly formatted, mark as INVALID.

        RESPONSE TO VALIDATE: 
        {response}

        Be extremely strict. The response must have ALL four sections with their exact headers. If even one section is missing, it's invalid.

        Format Instructions: {format_instructions}
        """
        
        prompt = PromptTemplate(
            template=validation_template,
            input_variables=["response"],
            partial_variables={"format_instructions": valid_parser.get_format_instructions()}
        )
        
        validation_chain = prompt | self.llm | valid_parser
        
        validation_result = validation_chain.invoke({
            "response": response
        })

        
        if validation_result.is_valid:
            print(f"Validation Result : Passed")
            print(f"Validation Feedback : {validation_result.feedback}")
            print("=========================================== [ Output Validation Completed ] ===========================================\n")
            return {
                "messages": [response],
                "original_question": state.get("original_question"),
                "validation_status": "passed",
                "validation_feedback": validation_result.feedback
            }
        else:
            if retry_count >= 2:  # Prevent infinite loops
                print("Maximum Retires Reached - Ending")
                print("=========================================== [ Output Validation Completed ] ===========================================\n")
                return {
                    "messages": [response],
                    "original_question": state.get("original_question"),
                    "validation_status": "max_retries",
                    "validation_feedback": f"Final attempt failed: {validation_result.feedback}"
                }
            else:
                print(f"Validation Result : Failed")
                print(f"Validation Feedback : {validation_result.feedback}")
                print("Preparing for Retry...........")
                print(f"Retry Count : {retry_count + 1}")
                print("=========================================== [ Output Validation Completed ] ===========================================\n")
                return {
                    "messages": state["messages"],
                    "original_question": state.get("original_question"),
                    "retry_count": retry_count + 1,
                    "validation_status": "retry",
                    "validation_feedback": validation_result.feedback
                }
    
    def final_output(self, state : AgentState):
        print("=========================================== [ Generating Final Output ] ===========================================")
        result = state["messages"][-1]
        print(f"Question : {state["original_question"]}\n")
        print(f"Final Response : {result}")
        print("=========================================== [ Final Output Generated ] ===========================================\n")
        return {"mesaages" : [result]}
