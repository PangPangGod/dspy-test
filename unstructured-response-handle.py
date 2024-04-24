from documenthandle import PDFHandler
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


import time
start_time = time.time()

document_handler = PDFHandler(filename="example/multi-column.pdf")
response = document_handler.load_cache()

if not response:
    response = document_handler.process_document()
    document_handler.save_to_cache(data=response)

print(type(response)) # 'unstructured_client.models.operations.partition.PartitionResponse'

print("Execution Complete.")
end_time = time.time()
print(f"Elapsed time: {end_time - start_time} seconds")

### 여기서부터 document handle해서 metadata handling하는것까지
### 'unstructured_client.models.operations.partition.PartitionResponse' -> response type

elements = response.elements

### make langchain Document object
docs = [
    Document(page_content=element['text'], 
            metadata={
            'type': element['type'],
            'element_id': element['element_id'],
            **element['metadata'],
        }) for element in elements
]

for doc in docs:
    print(doc.metadata)

###################################### vectorstore handling
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
# # test
# vectorstore = Chroma(
#     collection_name="summaries",
#     embedding_function=embeddings,
# )

## id_key = element_id

