import os
import pickle
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field
import os
import pickle

class BaseDocument(BaseModel, ABC):
    filename: str
    cache_dir: str = "pkl"

    @property
    def cache_path(self) -> str:
        """Dynamically generate and return the full directory path for the cache file."""
        base_filename = os.path.splitext(os.path.basename(self.filename))[0] + '.pkl'
        return os.path.join(self.cache_dir, base_filename)

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

class PDFHandler(BaseDocument):
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

    def process_document(self):
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
            return response

        except Exception as e:
            print("Exception :", e)

    
if __name__ == "__main__":
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


