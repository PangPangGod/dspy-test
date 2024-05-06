from typing import Any, List
from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf
import asyncio
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.embeddings import Embeddings
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings
from abc import ABC, abstractmethod

###
import os
import pickle
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from unstructured_client.models.operations.partition import PartitionResponse #Response type check

from typing import Dict, Literal, List
import numpy as np
from numpy import percentile, subtract, std

""" 
TODO:
- ABS CLASS Base 만들어서 나중에 사용 용이하게 만들 것
- 만들어놓은 documenthandler class 통합
- Docker 환경 어떻게 넣을 것인지(-> Image 설정 생각)
- pheonix로 추적 가능하게 할 것(log 남겨서 이용 가능)

- 추후에 cloud 이용한다면 storage랑 연결해서 만들 수 있도록 해야 함(현재 cache -> pkl 폴더로 옮기도록 해놓을 것)
- 다형성 측면에서 하나의 클래스가 하나의 역할만 하도록 (예시: Document class는 document 처리만 하도록 & 문서 요약 class는 response 받아서 문서 요약만 하도록 & 저장하고 retrieve architecture build하는 class)
"""

### Configs

BreakpointThresholdType = Literal["percentile", "standard_deviation", "interquartile"]
BREAKPOINT_DEFAULTS: Dict[BreakpointThresholdType, float] = {
    "percentile": 95,
    "standard_deviation": 3,
    "interquartile": 1.5,
}


class BaseDocument(BaseModel, ABC):
    """ Get document and cached for performance issuse -> default directory : ./pkl """
    filename: str
    cache_dir: str = "pkl"

    @property
    @abstractmethod
    def cache_path(self) -> str:
        """Dynamically generate and return the full directory path for the cache file."""
        pass

    def load_cache(self):
        """Load data from the cache file if it exists."""
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as file:
                print(f"Loading cached data from {self.cache_path}")
                return pickle.load(file)
        return None

    def save_to_cache(self, data):
        """Save data to a specified directory, creating the directory if it does not exist."""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        with open(self.cache_path, 'wb') as file:
            pickle.dump(data, file)
        print(f"Data saved to cache {self.cache_path}")

    @abstractmethod
    def process_document(self):
        pass

class APIPDFHandler(BaseDocument):
    def cache_path(self) -> str:
        base_filename = os.path.splitext(os.path.basename(self.filename))[0] + '_api_response.pkl'
        return os.path.join(self.cache_dir, base_filename)
    
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

    def process_document(self) -> PartitionResponse:
        client = self._get_api_client()

        with open(self.filename, "rb") as f:
            files = shared.Files(
                content=f.read(),
                file_name=self.filename,
            )

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
            print("Handled results :", len(response.elements))
            return response # PartitionResponse

        except Exception as e:
            print("Exception :", e)

class Element(BaseModel):
    type: str
    text: Any

class DocumentProcessor:
    def __init__(self, file_path:str, embedding_model:Embeddings, model="gpt-3.5-turbo-0125"):
        self.file_path = file_path
        self.embedding_model = embedding_model
        self.model = model
        self.text = []
        self.table = []

    def process_document(self):
        handler = APIPDFHandler(filename=self.file_path)
        response = handler.process_document()
        print(response)
        return self._categorize_elements(response.elements)

    def _categorize_elements(self, elements):
        categorized_elements = []
        for element in elements:
            if element['type'] == 'Table':
                categorized_elements.append(Element(type="table", text=element["text"]))
            elif element['type'] == 'CompositeElement':
                categorized_elements.append(Element(type="text", text=element["text"]))

        return categorized_elements
    
    def _calculate_similarity_through_chunks(
            self, text_element, embedded_table: List[float], breakpoint_threshold_type: BreakpointThresholdType = "percentile", breakpoint_threshold: float = None):
        """ Table Element 전체와 Text Element를 chunk 한 문장들과의 cosine 유사도를 구해서 threshold 이상이면 연관성이 존재한다고 판단,
        더해준 다음 SummaryAgent에 전달 후 MultiVectorRetriever에 이용. 추가적으로 통계적 분석을 통해 문장 필터링 가능."""

        # 옵션값 설정
        if breakpoint_threshold is None:
            breakpoint_threshold = BREAKPOINT_DEFAULTS[breakpoint_threshold_type]

        # 입력 텍스트 분리 및 임베딩
        splitted_text_chunk = text_element.text.split("\n")
        embedded_text_chunk = self.embedding_model.embed_documents(texts=[text for text in splitted_text_chunk])
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

        # self.log_message(f"Threshold for filtering: {breakpoint_value}")

        # 유사도가 분석 기준값 이상인 문장만 결과에 추가
        for index, similarity in enumerate(similarities):
            if similarity > breakpoint_value:
                result_context_to_insert += splitted_text_chunk[index] + "\n"

        return result_context_to_insert

    async def _summarize_chain(self, context_key, context, element, model=ChatOpenAI, max_tokens=512, temperature=0):
        prompt_text = f"You are an assistant tasked with summarizing tables and text.\n\
                    Based on contexts below, please summarize the {context_key} using the relevant context.\n\
                    \n\
                    Previous context: {context['prev_context']} \
                    {context_key}: {element} \
                    Following context: {context['after_context']}"
        prompt = ChatPromptTemplate.from_template(prompt_text)

        chain = (
            {context_key: lambda x: x, "prev_context": lambda x: context["prev_context"], "after_context": lambda x: context["after_context"]}
            | prompt
            | model(temperature=temperature, model=self.model, max_tokens=max_tokens)
            | StrOutputParser()
        )

        return await chain.ainvoke({context_key: element, "prev_context": context["prev_context"], "after_context": context["after_context"]})

    async def _summarize_text_chain(self, element):
        """ Wrapper function _summarize_chain """
        return await self._summarize_chain("text", {"prev_context": "", "after_context": ""}, element)

    async def _summarize_table_chain(self, table_context, prev_context="", after_context=""):
        """ Wrapper function _summarize_chain """
        return await self._summarize_chain("table", {"prev_context": prev_context, "after_context": after_context}, table_context)

    async def _async_process_document(self):
        categorized_elements = self.process_document()
        self.table_elements = [e.text for e in categorized_elements if e.type == "table"]
        self.text_elements = [e.text for e in categorized_elements if e.type == "text"]

        table_tasks = [
            self._summarize_table_chain(**self._get_context(element, categorized_elements), table_context=element.text)
            for element in categorized_elements if element.type == "table"
        ]

        text_tasks = [
            self._summarize_text_chain(element.text)
            for element in categorized_elements if element.type == "text"
        ]

        table_summaries = await asyncio.gather(*table_tasks)
        text_summaries = await asyncio.gather(*text_tasks)

        return table_summaries, text_summaries

    def _get_context(self, element, categorized_elements):
        context = {"prev_context": "", "after_context": ""}
        index = categorized_elements.index(element)

        if index > 0 and categorized_elements[index-1].type == "text":
            context["prev_context"] = self._calculate_similarity_through_chunks(
                categorized_elements[index-1], self.embedding_model.embed_documents(texts=[element.text])
            )

        if index < len(categorized_elements)-1 and categorized_elements[index+1].type == "text":
            context["after_context"] = self._calculate_similarity_through_chunks(
                categorized_elements[index+1], self.embedding_model.embed_documents(texts=[element.text])
            )

        return context
    
    async def execute(self):
        # 비동기적 문서 처리
        table_summaries, text_summaries = await self._async_process_document()

        print("="*80)
        # 결과 출력
        print("Table Summaries : ")
        for summary in table_summaries:
            print(summary)
        print("\nText Summaries : ")
        for summary in text_summaries:
            print(summary)
        print("="*80, "Table summary, Text summary Execute with ainvoke complete.")

        return self.table_elements, table_summaries, self.text_elements, text_summaries

    @classmethod
    async def run(cls, file_path:str, embedding_model:Embeddings):
        processor = cls(file_path=file_path, embedding_model=embedding_model)
        return await processor.execute()

if __name__ == "__main__":
    tables, summary_tables, texts, summary_texts = asyncio.run(DocumentProcessor.run(file_path="example/sample_file.pdf", embedding_model=OpenAIEmbeddings(model="text-embedding-3-small")))

    print(summary_tables)
    print("=====================")
    print(summary_texts)
