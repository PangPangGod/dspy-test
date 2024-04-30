from documenthandle import PDFHandler
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_groq import ChatGroq

from langchain_community.utils.math import (
    cosine_similarity,
)

from typing import Any, List
import time
import os

start_time = time.time()

file_name = "example/sample_file.pdf"
document_handler = PDFHandler(filename=file_name)

response = document_handler.load_cache()

file_name_seperated = os.path.splitext(os.path.basename(file_name))[0]
### result table summary cache handle
context_result_file_path = f"pkl/{file_name_seperated}_context.pkl"
langchain_result_file_path = f"pkl/{file_name_seperated}_langchain_summary_results.pkl"
dspy_result_file_path = f"pkl/{file_name_seperated}_dspy_summary_results.pkl"
groq_result_file_path = f"pkl/{file_name_seperated}_groq_summary_results.pkl"
###

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

# for doc in docs:
#     print(doc.metadata)

##################################### vectorstore handling
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
# test
vectorstore = Chroma(
    collection_name="summaries",
    embedding_function=embedding_model,
)

# id_key = element_id

#####################################
## setup multivector retriever with asyncronous langchain element(edit)
# first, get document matadata and confirm its Table


### response == categorized_element *response docs

### categorize element
table_elements = []
text_elements = []

for doc in docs:
    if doc.metadata['type'] == "Table":
        table_elements.append(doc)
    else :
        text_elements.append(doc)

## test
# print(len(table_elements))
# print(len(text_elements))

###


######### dspyconfig

import dspy
import re

llm = dspy.OpenAI(model='gpt-3.5-turbo')
dspy.settings.configure(lm=llm)

# 모델 Signature 정의
class BasicTableSummarize(dspy.Signature):
    """Summarize the given text. The text includes the contents of a table. Save this in a detailed information within."""
    text = dspy.InputField(desc="full text to summarize")
    summary = dspy.OutputField(desc="summarized text")

##########
from typing import Dict, Literal, List
import numpy as np
from numpy import percentile, subtract, std

# 유사도 검사를 위한 타입 정의
BreakpointThresholdType = Literal["percentile", "standard_deviation", "interquartile"]
BREAKPOINT_DEFAULTS: Dict[BreakpointThresholdType, float] = {
    "percentile": 95,
    "standard_deviation": 3,
    "interquartile": 1.5,
}

def _calculate_similarity_through_chunks(
        text_element, embedded_table: List[float], breakpoint_threshold_type: BreakpointThresholdType = "percentile", breakpoint_threshold: float = None):
    """ Table Element 전체와 Text Element를 chunk 한 문장들과의 cosine 유사도를 구해서 threshold 이상이면 연관성이 존재한다고 판단,
    더해준 다음 SummaryAgent에 전달 후 MultiVectorRetriever에 이용. 추가적으로 통계적 분석을 통해 문장 필터링 가능."""

    # 옵션값 설정
    if breakpoint_threshold is None:
        breakpoint_threshold = BREAKPOINT_DEFAULTS[breakpoint_threshold_type]

    # 입력 텍스트 분리 및 임베딩
    splitted_text_chunk = text_element.page_content.split("\n\n")
    embedded_text_chunk = embedding_model.embed_documents(texts=[text for text in splitted_text_chunk])

    similarities = []
    result_context_to_insert = ""

    # 유사도 계산
    for embedded_text in embedded_text_chunk:
        similarity = cosine_similarity([embedded_text], embedded_table)[0][0]
        similarities.append(similarity)

    # 유사도 데이터를 array로 변환
    similarity_array = np.array(similarities)
    # 분석 기준값 계산
    if breakpoint_threshold_type == "percentile":
        breakpoint_value = percentile(similarity_array, breakpoint_threshold)
    elif breakpoint_threshold_type == "standard_deviation":
        mean_value = np.mean(similarity_array)
        breakpoint_value = mean_value + std(similarity_array) * breakpoint_threshold
    elif breakpoint_threshold_type == "interquartile":
        iqr = subtract(*percentile(similarity_array, [75, 25]))
        breakpoint_value = percentile(similarity_array, 75) + iqr * breakpoint_threshold

    print(f"Breakpoint Value: {breakpoint_value}")

    # 유사도가 분석 기준값 이상인 문장만 결과에 추가
    for index, similarity in enumerate(similarities):
        if similarity > breakpoint_value:
            result_context_to_insert += splitted_text_chunk[index] + "\n"

    return result_context_to_insert



#### 이 부분을 gather 해서 coroutine하도록 하는게 나을지도?
#### cache화 해서 둘 다 테스트 해보는게 정답일듯 (dspy <-> langchain ainvoke)
def _summarize_table_chain(table_context, prev_context="", after_context=""):
    prompt_text = """You are an assistant tasked with summarizing tables and text. \
            Based on contexts below, please summarize the table context using the relevant context." \
            
            Previous context: {prev_context} \
            Table chunk: {table_context} \
            Following context: {after_context}"""
    prompt = ChatPromptTemplate.from_template(prompt_text)

    model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", max_tokens=512)
    chain = ({"table_context": lambda x: x, "prev_context": lambda x: x, "after_context": lambda x: x}
            | prompt
            | model
            | StrOutputParser())

    return chain.invoke({"element": table_context, "prev_context": prev_context, "after_context": after_context})
##################################################################

def _summarize_table_chain_with_dspy(table_context, prev_context="", after_context=""):        
    all_context = prev_context+"\n"+table_context+"\n"+after_context

    summarizer = dspy.ChainOfThought(BasicTableSummarize)
    result = summarizer(text=all_context)
    return result.summary

###################################################################

##################################################################
# Add groq example(24-04-30)
def _summarize_table_chain_with_groq(table_context, prev_context="", after_context=""):
    prompt_text = """You are an assistant tasked with summarizing tables and text. \
            Based on contexts below, please summarize the table context using the relevant context." \
            
            Previous context: {prev_context} \
            Table chunk: {table_context} \
            Following context: {after_context}"""
    prompt = ChatPromptTemplate.from_template(prompt_text)

    model = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
    chain = ({"table_context": lambda x: x, "prev_context": lambda x: x, "after_context": lambda x: x}
            | prompt
            | model
            | StrOutputParser())

    return chain.invoke({"element": table_context, "prev_context": prev_context, "after_context": after_context})
###################################################################


results = []
results_dspy = []
results_groq = []
contexts = []

## docs var name to more specific ones
for index, element in enumerate(docs):
    if element.metadata['type'] == "Table":
        embedded_table_text = embedding_model.embed_documents(texts=[element.page_content])
        context_before_table = "" 
        context_after_table = ""

        if index > 0 and docs[index-1].metadata['type'] == "CompositeElement":
            prev_element = docs[index-1]
            context_before_table += _calculate_similarity_through_chunks(prev_element, embedded_table_text)

        if index < len(docs)-1 and docs[index+1].metadata['type'] == "CompositeElement":
            after_element = docs[index+1]
            context_after_table += _calculate_similarity_through_chunks(after_element, embedded_table_text)

        full_context = context_before_table+"\n"+element.page_content+"\n"+context_after_table

        task = _summarize_table_chain(prev_context=context_before_table, after_context=context_after_table, table_context=element.page_content)
        task_dspy = _summarize_table_chain_with_dspy(prev_context=context_before_table, after_context=context_after_table, table_context=element.page_content)
        task_groq = _summarize_table_chain_with_groq(prev_context=context_before_table, after_context=context_after_table, table_context=element.page_content)

        results.append(task)
        results_dspy.append(task_dspy)
        results_groq.append(task_groq)
        contexts.append(full_context)

    # elif element.type == "text":
    #     task = self._summarize_text_chain(element.text)
    #     text_coroutine_tasks.append(task)

import pickle
# print(results)

with(open(context_result_file_path, 'wb')) as f:
    pickle.dump(contexts, f)
with(open(langchain_result_file_path, 'wb')) as f:
    pickle.dump(results, f)
with(open(dspy_result_file_path, 'wb')) as f:
    pickle.dump(results_dspy, f)
with(open(groq_result_file_path, 'wb')) as f:
    pickle.dump(results_groq, f)

print("end discussion")