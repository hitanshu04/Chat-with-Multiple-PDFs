import streamlit as st
from PyPDF2 import PdfReader
## There are some important libraries that need to import with respect to Langchain so what all libraries are those which we will import
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
## As  soon as we read the pdf we need to convert them into vectors so GoogleGenAI or baiscally the Gemini pro also provudes u the embeddings techniques r that we will write the next command

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
## We are going to use only the above 2 genai functionalities.
## Now we will import 4 dierent libraries.

from langchain.vectorstores import FAISS  ##FAISS is for vector embeddings om langchain.vecorstores
from langchain_google_genai import ChatGoogleGenerativeAI  
from langchain.chains.question_answering import load_qa_chain  ## "load_qa_chain" helps us to do the chat 

from langchain.prompts import PromptTemplate ##and any kinds of prompts also I want to define so forr that also I am importing this langchain.prompts  
from dotenv import load_dotenv

load_dotenv()  ##By this you will be able to see the environmnet variable.
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) #since, we need to configure our API key from google gemini so we will write "os.getenv("GOOGLE_API_KEY")"
# In the above step I am configuring the api key with respect to whatever google api key we have loaded in this ".env" le

#now in left side of demo we should be able to read the pdf and then whatever data is there inside the pdwe should be able to give it, so r that we will create a function "def get_pdf_text(pdf_docs)"

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:  ##Read all the pdf pages in this pdf_docs
        pdf_reader=PdfReader(pdf) ## Will read this with help of pdf reader(it will be responsible for reading specific pdfS)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text
## The above function says that we read the pdf we go through each and every pages and we extract the text
## As soon as we will get text we will create another function and we will divide this text into smaller chunks

##If we follow these steps it will be generic for all the pdf files whatever application we develop in genai 

def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000) ##U have the entire text and i want to divide them into smaller chunks othis size 10000 words or tokens and they can be overlap of 1000 so that I get a detailed summary of every question that I ask
    chunks=text_splitter.split_text(text)
    return chunks


## I read pdf firstly, then I divided it into chunks, now I will take this text chunks and will convert into vectors.

def get_vector_store(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model= "models/embedding-001")  ## In googlegenerativeaiembeddings, in documentation, inside models folder there's a model "e-001", this embedding technique I am going to use for embedding,in langchain also u hv different embeddings in OpenAI also u hv different embeddings but right now i want to use an embedding which is completely free for everyone to use but since google gen ai is already providing u so many features so why to go with others and pay money.
    vector_store=FAISS.from_texts(text_chunks,embedding=embeddings)  ## Take all the text_chunks and embedd according to this embeddings I have intialized. So this way the vector store is getting created.
    ##Now this vector store can be saved in a database it can also be saved in a local environment
    vector_store.save_local("faiss_index") ## Now "faiss_index" over here shows u that I am going to save entire information in local so "faiss_index" will be the folder that will be created and inside this folder I will be able to see my vectors in some format which is not readable, so u willl also see a pickle file so that whenever I ask a question to those vectprs I will be able to get the information. 

## I read pdf firstly, then I divided it into chunks, now converted these text chunks into vectors.

def get_conversational_chain():
    prompt_template="""
    Answer the question as detailed as possible from the provided context,make sure to provide all the details, if the answer is not in context, make sure to provide all the details, if the answer is not in provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model=ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)
    ## Obviously I have created Prompt so I am going to use the Prompt template available in Langchain , and so i will write "template=prompt_template" and we know what are the inpuut variables it will be in the rm olist and there r only 2 one is context and the 2nd one is specically "question"

    prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"]) ## these r my info w.r.t prompt template and assigned to prompt
    chain=load_qa_chain(model,chain_type="stuff",prompt=prompt) ## Now once the above step is done I will go ahead and create my chain so it will be "load_qa_chain" and inside that I will use "model" whatever model that I have actually specifically defined and I m goping to use the chain type as "stuff" bcz I also need to do internal tech summarization so this chain type "stuff" documentation chain will actually help us to do that and I will go ahead and give my prompt.
    return chain #return this chain

        
## 1st we will load the gemini pro model , we r going to create a template and then we r going to get the chain.
    
## Now fInally with respect to user input like as soon as I probably define or write in the text something should happen.

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
 ##If I give any info in the text input, ki tell me that topic of that pdf already that pdf is converted into vectors so it's already stored in the faiss index so that's why we r loading faiss index from local 
    docs = new_db.similarity_search(user_question) ## and then trying to do the similarity search based on user question it will do similarity search on all the faiss vectors thsat hv been created 

    chain = get_conversational_chain() #As soon as we do the similrity search we call the "get_conversational_chain" function by this we will get the chain back
    ## and then by following method we get the response
    response = chain(
        {"input_documents":docs,"question":user_question}
        , return_only_outputs=True) 
    
    print(response)
    st.write("Reply: ", response["output_text"])  ##display the response here

    ## so the user one is basically related to what's happening in text box
## creating streamlit app
def main():
    st.set_page_config("Chat with Multiple PDFs")
    st.header("Chat with Multiple PDF using Gemini 💁")

    user_question=st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question) # As sson as i write question in prompt and press enter it should execute this 
    ## But there also will be a side bar where i need to uplad the pdf and convert into vector so that's the reason for doing the following
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner ("Processing...."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
if __name__ == "__main__":
    main()

## 3 functions get pdf text, get text chunk , get vector storre it's only for making sure that your faiss index is created. 
## that's the reason it's given in sidebar there we upload teh file and process it