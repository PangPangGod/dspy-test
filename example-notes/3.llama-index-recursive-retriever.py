from llama_index.readers.file import FlatReader
from llama_index.core.node_parser import UnstructuredElementNodeParser
from pathlib import Path

reader = FlatReader()
data = reader.load_data(Path(""))

node_parser = UnstructuredElementNodeParser()

raw_nodes = node_parser.get_nodes_from_documents(data)
base_nodes, node_mappings = node_parser.get_base_nodes_and_mappings(raw_nodes)


from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import VectorStoreIndex

vector_index = VectorStoreIndex(base_nodes)
vector_retriever = vector_index.as_retriever(similarity_top_k=3)
vector_query_engine = vector_index.as_query_engine(similarity_top_k=3)

recursive_retriever = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever},
    node_dict=node_mappings,
)
query_engine = RetrieverQueryEngine.from_args(recursive_retriever)

response = query_engine.query("")
print(response)