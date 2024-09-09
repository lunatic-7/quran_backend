import os
from dotenv import load_dotenv
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from pydantic import BaseModel, ValidationError
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables from .env
load_dotenv()

# Persistent directory for vectorstore
persistent_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db", "quran_complete")

# Create embeddings using OpenAI
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load Chroma DB for vectorstore
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Create a retriever for querying the vectorstore
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Create LLM (language model) with OpenAI
llm = ChatOpenAI(model="gpt-4o")

# Contextualize question prompt
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

# Create a prompt template for contextualizing questions
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a history-aware retriever
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# Answer question prompt
qa_system_prompt = (
    "You are an assistant for question-answering tasks related to the Quran only. "
    "Use the following pieces of retrieved context to answer the question. "
    "If the retrived context and query are in different language, match both languages first"
    "If you don't know the answer, say that you don't know. Use three sentences max."
    "\n\n"
    "{context}"
)

# Create a prompt template for answering questions
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a chain to combine documents for question answering
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create a retrieval chain that combines the history-aware retriever and the question-answering chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


# Define the request schema
class ChatRequest(BaseModel):
    query: str
    chat_history: list

@api_view(['POST'])
def quran_api_view(request):
    try:
        # Validate the request data using Pydantic
        chat_request = ChatRequest(**request.data)
        
        # Process the user's query through the retrieval chain
        result = rag_chain.invoke({"input": chat_request.query, "chat_history": chat_request.chat_history})

        # Return the result
        return Response({"answer": result['answer']}, status=status.HTTP_200_OK)

    except ValidationError as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
