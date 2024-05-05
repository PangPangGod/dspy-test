ARG NB_USER=notebook-user
ARG NB_UID=1000
ARG PIP_VERSION

### FIRST STAGE
## SET DEV ENV

FROM python:3.10.14-slim AS python-base

ENV HOME=/home/${NB_USER}
ENV PYTHONPATH="${PYTHONPATH}:${HOME}"
ENV PATH="${HOME}/.local/bin:${PATH}"

WORKDIR ${HOME}

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

### SECOND STAGE

## Multi-stage build. integration with unstructured-io
## https://github.com/Unstructured-IO/unstructured/blob/main/Dockerfile

FROM quay.io/unstructured-io/base-images:rocky9.2-9@sha256:73d8492452f086144d4b92b7931aa04719f085c74d16cae81e8826ef873729c9 AS unstructured-base

WORKDIR ${HOME}

RUN groupadd --gid ${NB_UID} ${NB_USER} && \
    useradd --uid ${NB_UID} --gid ${NB_UID} --create-home --shell /bin/bash ${NB_USER}

COPY --from=python-base ${HOME} ${HOME}

RUN python3 -m pip install pip==${PIP_VERSION} && \
    python3 -m nltk.downloader punkt averaged_perceptron_tagger

RUN python3 -c "from unstructured.partition.model_init import initialize; initialize()"

USER ${NB_USER}

CMD ["/bin/bash"]
