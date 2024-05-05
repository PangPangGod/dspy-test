import os
import pickle
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
import datetime

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

    @abstractmethod
    def process_document(self):
        """Process the document. This method must be implemented by subclasses."""
        pass

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
        
    def return_langchain_document(self):
        """ TODO : GET response object and parse it to the langchain Document object(for Retrieval usage.) and returns it."""
        pass


if __name__ == "__main__":
    import time
    start_time = time.time()
    document_handler = APIPDFHandler(file_path="example/multi-column.pdf")
    response = document_handler.handle_document()

    print(type(response)) # 'unstructured_client.models.operations.partition.PartitionResponse'
    print("Execution Complete.")

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time} seconds")


