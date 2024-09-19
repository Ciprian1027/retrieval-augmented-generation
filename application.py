### This code was made to work with OpenAI API, but it can be modified to work with any open source model that can be used for text generation.
### Simply follow the instructions in the comments to use the alternative code. All the necessary libraries are already imported in the code,
### or commented out in case you need to use them.

### Make sure to insert documents in the "data" folder in the root directory of the project. The documents should be in PDF format.
### If the folder is not present, create one, name it "data" and insert the documents there.

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
# from langchain.huggingface import HuggingFaceEmbeddings
# from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import shutil
import openai
import os
import nltk


nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path)
nltk.download('maxent_ne_chunker', download_dir=nltk_data_path)
nltk.download('words', download_dir=nltk_data_path)

### Insert your OpenAI API key in the .env file, if you do not have one, comment this line and use the commented code below.
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

CHROMA_PATH = "chroma"
DATA_PATH = "data"

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf")
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=750,
        length_function=len,
        add_start_index=True
    )
    all_splits = text_splitter.split_documents(documents)
    return all_splits

def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


    ### Use the commented code below as an alternative to OpenAI.

    # client = chromadb.Client()
    # collection = client.create_collection(name="documents")

    # model = SentenceTransformer('all-mpnet-base-v2')

    # for i, chunk in enumerate(chunks):
    #     embedding = model.encode(chunk.page_content)
    #     collection.add(
    #         documents=[chunk.page_content],
    #         metadatas=[chunk.metadata],
    #         ids=[str(i)],
    #         embeddings=[embedding]
    #     )

    embedding_function = OpenAIEmbeddings()

    db = Chroma.from_documents(
        chunks, embedding_function, persist_directory=CHROMA_PATH)

def generate_data_store():
    documents = load_documents()
    all_splits = split_text(documents)
    save_to_chroma(all_splits)


PROMPT_TEMPLATE = """
            Vă rog să răspundeți la întrebările utilizatorilor folosind contextul furnizat:
            
            {context}

            ---

            Raspundeti la urmatoarea intrebare pe baza contextului de mai sus: {question}
            """


def main():
    generate_data_store()

    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    while True:
        query_text = input("Enter your query (or type 'exit' to quit): ")
        
        if query_text.lower() == "exit":
            print("Exiting the program.")
            break
        
        
        results = db.similarity_search_with_relevance_scores(query_text, k=10) # Get the top k most relevant documents. Increase k for more/better results.
        if len(results) == 0 or results[0][1] < 0.7: # If the most relevant document has a relevance score below 0.7, it means the query is not relevant enough.
                                                     # You can change this threshold to a higher or lower value depending on your needs.   
            print("Unable to find matching results")
            continue

        ### Use the commented code below as an alternative to OpenAI.

        # client = chromadb.Client(Settings(persist_directory=CHROMA_PATH))
        # collection = client.get_collection(name="documents")

        # model = SentenceTransformer('all-mpnet-base-v2')
        # query_embedding = model.encode(query_text).tolist()
        # results = collection.query(query_embeddings=[query_embedding], n_results=3)

        # if len(results['documents']) == 0:
        #     print("Unable to find matching results")
        #     return


        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        print(prompt)

        model = ChatOpenAI(model="gpt-4o")
        response = model.predict(prompt)

        sources = [doc.metadata.get("source", None) for doc, _score in results]
        formatted_response = f"Response: {response}\n\nSources: {sources}"
        print(formatted_response)

if __name__ == "__main__":
    main()