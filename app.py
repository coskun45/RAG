# Bibliotheken
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from sentence_transformers import SentenceTransformer
from langchain.chains import LLMChain
from langchain_community.document_loaders import PyPDFLoader
import os




# Load Knowledge Base
loader3 = PyPDFLoader("C:/Users/ecoskun/Desktop/BachelorarbeitAbgabe/bachelorarbeit_eyuep_coskun_3594788/KnowledgeBase/verbraucher_Information.pdf") # local path für Knowledge Base
documents_verbraucher_information = loader3.load_and_split()
knowledgeBase=documents_verbraucher_information

#chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
texts_1024 = text_splitter.split_documents(knowledgeBase)

# Embeddings
model_deutsche_telekom = HuggingFaceEmbeddings(model_name="deutsche-telekom/gbert-large-paraphrase-cosine")

# Speicher Embeddings in db
db_name = "chroma_db_mistral"
db_path = os.path.join(db_name)
if os.path.exists(db_path):
    db=Chroma(persist_directory="chroma_db_mistral", embedding_function=model_deutsche_telekom)# wenn db schon erstellt wurde,load db
else:
    db = Chroma.from_documents(texts_1024, model_deutsche_telekom, persist_directory="chroma_db_mistral")# erstellt vektor datenbank

# Load Mistral-LLM
model_path = "C:/Users/ecoskun/AppData/Local/nomic.ai/GPT4All/em_german_mistral_v01.Q4_0.gguf"#local path für LLM
callbacks = [StreamingStdOutCallbackHandler()]
llm= GPT4All(model=model_path, callbacks=callbacks, verbose=True,n_threads=16, temp=0.5)

# Prompt Template
from langchain.prompts import PromptTemplate
prompt_template='''
Du bist ein hilfreicher Assistent. Für die folgende Aufgabe stehen dir zwischen den tags BEGININPUT und ENDINPUT Kontext zur Verfügung. Die eigentliche Frage ist zwischen BEGININSTRUCTION und ENDINCSTRUCTION zu finden. Beantworten Sie die folgende Frage nur basierend auf dem angegebenen context auf Deutsch. Sollten diese keine Antwort enthalten, antworte, dass auf Basis der gegebenen Informationen keine Antwort möglich ist! 
USER: 
BEGININPUT{context}ENDINPUT
BEGININSTRUCTION {question} ENDINSTRUCTION 
ASSISTANT:
'''
PROMPT=PromptTemplate(
    template=prompt_template,input_variables=["context","question"]
)

# Erstellt Chain
chain_type_kwargs={"prompt":PROMPT}
qa_RAG_chain =RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
    return_source_documents=True,
    verbose=False,
    chain_type_kwargs=chain_type_kwargs
)


def main():

    print("Hallo! Ich bin ein ein Kfz-Versicherungsassistent. Du kannst 'exit' eingeben, um dich abzumelden.")
    while True:
        user_input = input("Du: ")
        if user_input.lower() == 'exit':
            print("Kfz-Versicherungsassistent wird gesschlossen.")
            break
        print("Kfz-Versicherungsassistent: ", end="") 
        qa_RAG_chain.invoke(user_input)["result"]
        print()


if __name__ == "__main__":
    main()