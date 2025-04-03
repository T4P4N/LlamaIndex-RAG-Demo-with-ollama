from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import os

Settings.embed_model = OllamaEmbedding(
    model_name="llama3.1",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)
Settings.llm = Ollama(model="llama3.1", request_timeout=60.0)


PERSIST_DIR = "./storage"

if not os.path.exists(PERSIST_DIR):
    documents = SimpleDirectoryReader("./data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)


question = "What is the secret to become a 10X developer?"
print("Q: ",question)

query_engine = index.as_query_engine()

answer = query_engine.query(question)
print("A:",answer)
