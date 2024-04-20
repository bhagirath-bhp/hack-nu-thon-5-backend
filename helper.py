from langchain_google_genai import GoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import os
from dotenv import load_dotenv
from langchain import OpenAI
import openai
from langchain_community.vectorstores import Milvus
from langchain.chains import RetrievalQA 
from langchain.memory import ConversationBufferWindowMemory
from langchain import PromptTemplate
from langchain.memory import PostgresChatMessageHistory
    
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
# from dataCreation import get_docs
load_dotenv()

# initialize LLM
def initialize_llm():
    global llm
    llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv("GOOGLE_API_KEY") , temperature=0.6)




# embeddings
def initialize_embeddings():
    try :
        global embeddings
        embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=os.getenv("HUGGINGFACE_API_KEY"), model_name="sentence-transformers/all-MiniLM-l6-v2"
        )
    except Exception as e :
        print("probelm accured while creating embeddings \n{}".format(e))


# vector database
def load_vectorDB(collection_name):
    global vector_db
    vector_db = Milvus(
    embeddings,
    collection_name=collection_name,
    connection_args={
    "uri":os.getenv("ZILLIZ_CLOUD_URI") ,
    "user":os.getenv("ZILLIZ_CLOUD_USERNAME") ,
    "password":os.getenv("ZILLIZ_CLOUD_PASSWORD") ,
    # "token": ZILLIZ_CLOUD_API_KEY,  # API key, for serverless clusters which can be used as replacements for user and password
    "secure": True,
    },
)
    

def get_prompt(query):
    #  get tags
    model = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)
    tags = model(
        [
            SystemMessage(content="you are a tag generator. you have to assign tags to user query which can be used to suggest products to the customer .do not type any other things only provide tags. tags could be any color , size , gender ,cloth fabric in [Cotton polyester Silk Blend Acrylic Wool Jean Polyester Faux Georgette Silk Jacquard] or design catagory in [Casual Sports Regular Birthday Formal Clubwear Partywear Funky Sweater casual Aesthetic Sport traditional]. if you do not find any relavant tags then return None."),
            HumanMessage(content=query),
        ]
    )
    print("tags" , tags.content )
    # similarity search based on tags
    simmcontext = ""
    if tags.content.strip() not in ["None" , "none" , ""] : 
        simtag = vector_db.similarity_search(query=tags.content , k=3)
        simmcontext = simtag[0].page_content + " --- " + simtag[1].page_content + " --- " + simtag[2].page_content    
        print("sim context " , simmcontext )

    return """ you are a salse person. try to give human like response according to user query and given context.
    Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question. try to give full product names in response :
    ------
    <ctx> """ + simmcontext + """
    {context}
    </ctx>
    ------
    <hs>
    {history}
    </hs>
    ------
    {question}
    Answer:
    """
def ask_ai(companyName , query):
    # load vector db
    load_vectorDB(companyName)
    template = get_prompt(query)
    prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=template,
    )
    # history
    history = PostgresChatMessageHistory(
        connection_string=os.getenv("POSTGRES_CONNECTION_STRING"),
        session_id=companyName,)
    user_memory = ConversationBufferWindowMemory(k=3 , memory_key="history" , chat_memory= history, input_key="question")
    posgres_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=vector_db.as_retriever(),
        # verbose=True,
        chain_type_kwargs={
            # "verbose": True,
            "prompt": prompt,
            "memory": user_memory
        }
    )
    # print("prompt -------------------------- " , vector_db.similarity_search(query=query , k=10))
    res = posgres_qa(query)
    return res["result"]