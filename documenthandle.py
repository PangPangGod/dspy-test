import os
import pickle
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
import datetime

from typing import Dict, Literal, List
import numpy as np
from numpy import percentile, subtract, std

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.utils.math import (
    cosine_similarity,
)

import asyncio


### Configs

BreakpointThresholdType = Literal["percentile", "standard_deviation", "interquartile"]
BREAKPOINT_DEFAULTS: Dict[BreakpointThresholdType, float] = {
    "percentile": 95,
    "standard_deviation": 3,
    "interquartile": 1.5,
}

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# test
vectorstore = Chroma(
    collection_name="summaries",
    embedding_function=embedding_model,
)



class BaseDocument(BaseModel, ABC):
    file_path: str
    cache_dir: str = "pkl"

    def setup_directories(self):
        """Set up cache and log directories when the object is used for the first time."""
        base_dir = os.path.join(self.cache_dir, os.path.splitext(os.path.basename(self.file_path))[0])
        os.makedirs(base_dir, exist_ok=True)
        log_path = os.path.join(base_dir, 'log.txt')
        if not os.path.exists(log_path):
            with open(log_path, 'w') as log_file:
                log_file.write("Log created\n")

    def get_path(self, footer: str) -> str:
        """
        Generate and return the full directory path for a specified file type within the cache directory.
        
        Args:
        footer (str): The filename to append to the path, such as 'data.pkl' or 'log.txt'.
        
        Returns:
        str: The full path to the specified file.
        """
        base_dir = os.path.join(self.cache_dir, os.path.splitext(os.path.basename(self.file_path))[0])
        return os.path.join(base_dir, footer)

    def save_to_cache(self, data, footer: str):
        """Save data to the cache file."""
        cache_path = self.get_path(footer)
        with open(cache_path, 'wb') as file:
            pickle.dump(data, file)
        print(f"Data saved to cache {cache_path}")

    def load_cache(self, footer: str):
        """Load data from the cache file if it exists."""
        cache_path = self.get_path(footer)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as file:
                return pickle.load(file)
        return None

    def log_message(self, message: str):
        """Write a message to the log file, including the current timestamp."""
        log_path = self.get_path('log.txt')
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{current_time}: {message}\n"
        with open(log_path, 'a') as log_file:
            log_file.write(log_entry)



class APIPDFHandler(BaseDocument):
    def _get_api_client(self):
        client = UnstructuredClient(
        server_url="http://localhost:8000",
        api_key_auth="", #no need to authorize this parameter cause you don't use SASS api key.
        )
        """ Get client from local only for docker env ........ 
            Using
            docker run -p 8000:8000 -d --rm --name unstructured-api -e UNSTRUCTURED_PARALLEL_MODE_THREADS=3 downloads.unstructured.io/unstructured-io/unstructured-api:latest --port 8000 --host 0.0.0.0
        """
        return client

    def handle_document(self):
        self.setup_directories() ## directory setup
        response = self.load_cache(footer='response.pkl')
        if response:
            cache_path = self.get_path('response.pkl')
            self.log_message(f"Loading cached data from {cache_path}")
        else:
            self.log_message("No cached data found. Processing document.")
            response = self.process_document()
        
        return response

    def process_document(self):
        self.log_message("Starting document processing.")
        client = self._get_api_client()

        with open(self.file_path, "rb") as f:
            files = shared.Files(
                content=f.read(),
                file_name=self.file_path,
            )
            self.log_message("File has been read successfully.")

        """ Unstructured Parameter Handler here. """
        req = shared.PartitionParameters(
            files=files,
            chunking_strategy="by_title",
            strategy='hi_res',
            split_pdf_page=True,
            coordinates=True, ## this is just example. but if you want split_pdf_page, recommand to use hi_res strategy.
        )

        try:
            response = client.general.partition(req)
            self.log_message(f"Handled {len(response.elements)} elements successfully.")
            self.save_to_cache(data=response, footer="response.pkl")
            return response
        except Exception as e:
            self.log_message(f"Exception occurred: {e}")
            return None
    
    @classmethod
    def convert_response_to_langchain_document(self, response):
        """ Convert unstructured object into langchain document. """
        elements = response.elements

        docs = [
            Document(page_content=element['text'], 
                    metadata={
                    'type': element['type'],
                    'element_id': element['element_id'],
                    **element['metadata'],
                }) for element in elements
        ]

        return docs

##################################################################
class DocumentSummaryHandler(BaseDocument):
    model: str

    def _calculate_similarity_through_chunks(
            self, text_element, embedded_table: List[float], breakpoint_threshold_type: BreakpointThresholdType = "percentile", breakpoint_threshold: float = None):
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

        self.log_message(f"Threshold for filtering: {breakpoint_value}")

        # 유사도가 분석 기준값 이상인 문장만 결과에 추가
        for index, similarity in enumerate(similarities):
            if similarity > breakpoint_value:
                result_context_to_insert += splitted_text_chunk[index] + "\n"

        return result_context_to_insert
    
    async def _summarize_text_chain(self, element):
        prompt_text = """You are an assistant tasked with summarizing tables and text. \ 
            Give a concise summary of the table or text. Table or text chunk: {element}"""
        prompt = ChatPromptTemplate.from_template(prompt_text)

        model = ChatOpenAI(temperature=0, model=self.model, max_tokens=512)
        chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

        # 'astream' 대신 'ainvoke'를 사용
        return await chain.ainvoke(element)
    
    async def _summarize_table_chain(self, table_context, prev_context="", after_context=""):
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

        return chain.ainvoke({"element": table_context, "prev_context": prev_context, "after_context": after_context})
    
    
    async def _async_process_document(self):
        categorized_elements = self.process_document()
        self.table_elements = [e.text for e in categorized_elements if e.type == "Table"]
        self.text_elements = [e.text for e in categorized_elements if e.type == "CompositeElement"]

        # Create coroutine objects for table, text elements
        table_coroutine_tasks = []
        text_coroutine_tasks = []

        for index, element in enumerate(categorized_elements):
            if element.type == "Table":
                embedded_table = self.embedding_model.embed_documents(texts=[element.text])
                context_before_table = "" 
                context_after_table = ""

                if index > 0 and categorized_elements[index-1].type == "text":
                    prev_element = categorized_elements[index-1]
                    context_before_table += self._calculate_similarity_through_chunks(prev_element, embedded_table)

                if index < len(categorized_elements)-1 and categorized_elements[index+1].type == "text":
                    after_element = categorized_elements[index+1]
                    context_after_table += self._calculate_similarity_through_chunks(after_element, embedded_table)

                task = self._summarize_table_chain(prev_context=context_before_table, after_context=context_after_table, table_context=element.text)
                table_coroutine_tasks.append(task)

            elif element.type == "text":
                task = self._summarize_text_chain(element.text)
                text_coroutine_tasks.append(task)

        table_summaries = await asyncio.gather(*table_coroutine_tasks)
        text_summaries = await asyncio.gather(*text_coroutine_tasks)

        return table_summaries, text_summaries
    
    async def execute(self):
        self.setup_directories()
        table_summaries = self.load_cache("table_summary.pkl")
        text_summaries = self.load_cache("text_summary.pkl")

        if not table_summaries or not text_summaries:
            self.log_message("No cached summaries found. Processing document.")
            table_summaries, text_summaries = await self._async_process_document()
            self.save_to_cache(table_summaries, "table_summary.pkl")
            self.save_to_cache(text_summaries, "text_summary.pkl")
        else:
            self.log_message("Loaded summaries from cache.")

        return table_summaries, text_summaries

    async def run_summary_processing(self):
        """Method to run the summary processing, cache results, and print summaries."""
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        table_summaries, text_summaries = await self.execute()
        print("Table Summaries:", table_summaries)
        print("Text Summaries:", text_summaries)



import asyncio
import time

async def main():
    # Initialize the document handler with the specific PDF file path
    document_handler = APIPDFHandler(file_path="example/sample_file.pdf")
    
    # Handle the document: this will either load from cache or process the document using the API
    response = document_handler.handle_document()
    
    # Assuming that the response contains elements that can be converted into langchain documents
    documents = APIPDFHandler.convert_response_to_langchain_document(response)

    # Initialize the summary handler with the same file path (or you could specify where the summaries should be cached)
    summary_handler = DocumentSummaryHandler(file_path="example/sample_file.pdf", model="gpt-3.5-turbo")

    # Iterate over documents to perform summarization
    table_summaries = []
    text_summaries = []
    for doc in documents:
        if doc.metadata['type'] == 'table':
            summary = await summary_handler._summarize_table_chain(table_context=doc.page_content)
            table_summaries.append(summary)
        else:
            summary = await summary_handler._summarize_text_chain(doc.page_content)
            text_summaries.append(summary)

    # Save summaries to cache
    summary_handler.save_to_cache(table_summaries, "table_summary.pkl")
    summary_handler.save_to_cache(text_summaries, "text_summary.pkl")

    # Optionally, print summaries
    print("Table Summaries:", table_summaries)
    print("Text Summaries:", text_summaries)

    # Execution time logging
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time} seconds")

if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(main())

