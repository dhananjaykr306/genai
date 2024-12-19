{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "source": [
        "pip install chromadb"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "xcCBIpRgHxoJ",
        "outputId": "8d0962a2-652c-47ef-d8e8-23e6bd00c17c",
        "collapsed": true
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting chromadb\n",
            "  Downloading chromadb-0.5.23-py3-none-any.whl.metadata (6.8 kB)\n",
            "Collecting build>=1.0.3 (from chromadb)\n",
            "  Downloading build-1.2.2.post1-py3-none-any.whl.metadata (6.5 kB)\n",
            "Requirement already satisfied: pydantic>=1.9 in /usr/local/lib/python3.10/dist-packages (from chromadb) (2.10.3)\n",
            "Collecting chroma-hnswlib==0.7.6 (from chromadb)\n",
            "  Downloading chroma_hnswlib-0.7.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (252 bytes)\n",
            "Collecting fastapi>=0.95.2 (from chromadb)\n",
            "  Downloading fastapi-0.115.6-py3-none-any.whl.metadata (27 kB)\n",
            "Collecting uvicorn>=0.18.3 (from uvicorn[standard]>=0.18.3->chromadb)\n",
            "  Downloading uvicorn-0.34.0-py3-none-any.whl.metadata (6.5 kB)\n",
            "Requirement already satisfied: numpy>=1.22.5 in /usr/local/lib/python3.10/dist-packages (from chromadb) (1.26.4)\n",
            "Collecting posthog>=2.4.0 (from chromadb)\n",
            "  Downloading posthog-3.7.4-py2.py3-none-any.whl.metadata (2.0 kB)\n",
            "Requirement already satisfied: typing_extensions>=4.5.0 in /usr/local/lib/python3.10/dist-packages (from chromadb) (4.12.2)\n",
            "Collecting onnxruntime>=1.14.1 (from chromadb)\n",
            "  Downloading onnxruntime-1.20.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (4.5 kB)\n",
            "Requirement already satisfied: opentelemetry-api>=1.2.0 in /usr/local/lib/python3.10/dist-packages (from chromadb) (1.29.0)\n",
            "Collecting opentelemetry-exporter-otlp-proto-grpc>=1.2.0 (from chromadb)\n",
            "  Downloading opentelemetry_exporter_otlp_proto_grpc-1.29.0-py3-none-any.whl.metadata (2.2 kB)\n",
            "Collecting opentelemetry-instrumentation-fastapi>=0.41b0 (from chromadb)\n",
            "  Downloading opentelemetry_instrumentation_fastapi-0.50b0-py3-none-any.whl.metadata (2.1 kB)\n",
            "Requirement already satisfied: opentelemetry-sdk>=1.2.0 in /usr/local/lib/python3.10/dist-packages (from chromadb) (1.29.0)\n",
            "Collecting tokenizers<=0.20.3,>=0.13.2 (from chromadb)\n",
            "  Downloading tokenizers-0.20.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.7 kB)\n",
            "Collecting pypika>=0.48.9 (from chromadb)\n",
            "  Downloading PyPika-0.48.9.tar.gz (67 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m67.3/67.3 kB\u001b[0m \u001b[31m4.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25h  Installing build dependencies ... \u001b[?25l\u001b[?25hdone\n",
            "  Getting requirements to build wheel ... \u001b[?25l\u001b[?25hdone\n",
            "  Preparing metadata (pyproject.toml) ... \u001b[?25l\u001b[?25hdone\n",
            "Requirement already satisfied: tqdm>=4.65.0 in /usr/local/lib/python3.10/dist-packages (from chromadb) (4.67.1)\n",
            "Collecting overrides>=7.3.1 (from chromadb)\n",
            "  Downloading overrides-7.7.0-py3-none-any.whl.metadata (5.8 kB)\n",
            "Requirement already satisfied: importlib-resources in /usr/local/lib/python3.10/dist-packages (from chromadb) (6.4.5)\n",
            "Requirement already satisfied: grpcio>=1.58.0 in /usr/local/lib/python3.10/dist-packages (from chromadb) (1.68.1)\n",
            "Collecting bcrypt>=4.0.1 (from chromadb)\n",
            "  Downloading bcrypt-4.2.1-cp39-abi3-manylinux_2_28_x86_64.whl.metadata (9.8 kB)\n",
            "Requirement already satisfied: typer>=0.9.0 in /usr/local/lib/python3.10/dist-packages (from chromadb) (0.15.1)\n",
            "Collecting kubernetes>=28.1.0 (from chromadb)\n",
            "  Downloading kubernetes-31.0.0-py2.py3-none-any.whl.metadata (1.5 kB)\n",
            "Requirement already satisfied: tenacity>=8.2.3 in /usr/local/lib/python3.10/dist-packages (from chromadb) (9.0.0)\n",
            "Requirement already satisfied: PyYAML>=6.0.0 in /usr/local/lib/python3.10/dist-packages (from chromadb) (6.0.2)\n",
            "Collecting mmh3>=4.0.1 (from chromadb)\n",
            "  Downloading mmh3-5.0.1-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (14 kB)\n",
            "Requirement already satisfied: orjson>=3.9.12 in /usr/local/lib/python3.10/dist-packages (from chromadb) (3.10.12)\n",
            "Requirement already satisfied: httpx>=0.27.0 in /usr/local/lib/python3.10/dist-packages (from chromadb) (0.28.1)\n",
            "Requirement already satisfied: rich>=10.11.0 in /usr/local/lib/python3.10/dist-packages (from chromadb) (13.9.4)\n",
            "Requirement already satisfied: packaging>=19.1 in /usr/local/lib/python3.10/dist-packages (from build>=1.0.3->chromadb) (24.2)\n",
            "Collecting pyproject_hooks (from build>=1.0.3->chromadb)\n",
            "  Downloading pyproject_hooks-1.2.0-py3-none-any.whl.metadata (1.3 kB)\n",
            "Requirement already satisfied: tomli>=1.1.0 in /usr/local/lib/python3.10/dist-packages (from build>=1.0.3->chromadb) (2.2.1)\n",
            "Collecting starlette<0.42.0,>=0.40.0 (from fastapi>=0.95.2->chromadb)\n",
            "  Downloading starlette-0.41.3-py3-none-any.whl.metadata (6.0 kB)\n",
            "Requirement already satisfied: anyio in /usr/local/lib/python3.10/dist-packages (from httpx>=0.27.0->chromadb) (3.7.1)\n",
            "Requirement already satisfied: certifi in /usr/local/lib/python3.10/dist-packages (from httpx>=0.27.0->chromadb) (2024.12.14)\n",
            "Requirement already satisfied: httpcore==1.* in /usr/local/lib/python3.10/dist-packages (from httpx>=0.27.0->chromadb) (1.0.7)\n",
            "Requirement already satisfied: idna in /usr/local/lib/python3.10/dist-packages (from httpx>=0.27.0->chromadb) (3.10)\n",
            "Requirement already satisfied: h11<0.15,>=0.13 in /usr/local/lib/python3.10/dist-packages (from httpcore==1.*->httpx>=0.27.0->chromadb) (0.14.0)\n",
            "Requirement already satisfied: six>=1.9.0 in /usr/local/lib/python3.10/dist-packages (from kubernetes>=28.1.0->chromadb) (1.17.0)\n",
            "Requirement already satisfied: python-dateutil>=2.5.3 in /usr/local/lib/python3.10/dist-packages (from kubernetes>=28.1.0->chromadb) (2.8.2)\n",
            "Requirement already satisfied: google-auth>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from kubernetes>=28.1.0->chromadb) (2.27.0)\n",
            "Requirement already satisfied: websocket-client!=0.40.0,!=0.41.*,!=0.42.*,>=0.32.0 in /usr/local/lib/python3.10/dist-packages (from kubernetes>=28.1.0->chromadb) (1.8.0)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from kubernetes>=28.1.0->chromadb) (2.32.3)\n",
            "Requirement already satisfied: requests-oauthlib in /usr/local/lib/python3.10/dist-packages (from kubernetes>=28.1.0->chromadb) (1.3.1)\n",
            "Requirement already satisfied: oauthlib>=3.2.2 in /usr/local/lib/python3.10/dist-packages (from kubernetes>=28.1.0->chromadb) (3.2.2)\n",
            "Requirement already satisfied: urllib3>=1.24.2 in /usr/local/lib/python3.10/dist-packages (from kubernetes>=28.1.0->chromadb) (2.2.3)\n",
            "Collecting durationpy>=0.7 (from kubernetes>=28.1.0->chromadb)\n",
            "  Downloading durationpy-0.9-py3-none-any.whl.metadata (338 bytes)\n",
            "Collecting coloredlogs (from onnxruntime>=1.14.1->chromadb)\n",
            "  Downloading coloredlogs-15.0.1-py2.py3-none-any.whl.metadata (12 kB)\n",
            "Requirement already satisfied: flatbuffers in /usr/local/lib/python3.10/dist-packages (from onnxruntime>=1.14.1->chromadb) (24.3.25)\n",
            "Requirement already satisfied: protobuf in /usr/local/lib/python3.10/dist-packages (from onnxruntime>=1.14.1->chromadb) (4.25.5)\n",
            "Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from onnxruntime>=1.14.1->chromadb) (1.13.1)\n",
            "Requirement already satisfied: deprecated>=1.2.6 in /usr/local/lib/python3.10/dist-packages (from opentelemetry-api>=1.2.0->chromadb) (1.2.15)\n",
            "Requirement already satisfied: importlib-metadata<=8.5.0,>=6.0 in /usr/local/lib/python3.10/dist-packages (from opentelemetry-api>=1.2.0->chromadb) (8.5.0)\n",
            "Requirement already satisfied: googleapis-common-protos~=1.52 in /usr/local/lib/python3.10/dist-packages (from opentelemetry-exporter-otlp-proto-grpc>=1.2.0->chromadb) (1.66.0)\n",
            "Collecting opentelemetry-exporter-otlp-proto-common==1.29.0 (from opentelemetry-exporter-otlp-proto-grpc>=1.2.0->chromadb)\n",
            "  Downloading opentelemetry_exporter_otlp_proto_common-1.29.0-py3-none-any.whl.metadata (1.8 kB)\n",
            "Collecting opentelemetry-proto==1.29.0 (from opentelemetry-exporter-otlp-proto-grpc>=1.2.0->chromadb)\n",
            "  Downloading opentelemetry_proto-1.29.0-py3-none-any.whl.metadata (2.3 kB)\n",
            "Collecting protobuf (from onnxruntime>=1.14.1->chromadb)\n",
            "  Downloading protobuf-5.29.2-cp38-abi3-manylinux2014_x86_64.whl.metadata (592 bytes)\n",
            "Collecting opentelemetry-instrumentation-asgi==0.50b0 (from opentelemetry-instrumentation-fastapi>=0.41b0->chromadb)\n",
            "  Downloading opentelemetry_instrumentation_asgi-0.50b0-py3-none-any.whl.metadata (1.9 kB)\n",
            "Collecting opentelemetry-instrumentation==0.50b0 (from opentelemetry-instrumentation-fastapi>=0.41b0->chromadb)\n",
            "  Downloading opentelemetry_instrumentation-0.50b0-py3-none-any.whl.metadata (6.1 kB)\n",
            "Requirement already satisfied: opentelemetry-semantic-conventions==0.50b0 in /usr/local/lib/python3.10/dist-packages (from opentelemetry-instrumentation-fastapi>=0.41b0->chromadb) (0.50b0)\n",
            "Collecting opentelemetry-util-http==0.50b0 (from opentelemetry-instrumentation-fastapi>=0.41b0->chromadb)\n",
            "  Downloading opentelemetry_util_http-0.50b0-py3-none-any.whl.metadata (2.5 kB)\n",
            "Requirement already satisfied: wrapt<2.0.0,>=1.0.0 in /usr/local/lib/python3.10/dist-packages (from opentelemetry-instrumentation==0.50b0->opentelemetry-instrumentation-fastapi>=0.41b0->chromadb) (1.17.0)\n",
            "Collecting asgiref~=3.0 (from opentelemetry-instrumentation-asgi==0.50b0->opentelemetry-instrumentation-fastapi>=0.41b0->chromadb)\n",
            "  Downloading asgiref-3.8.1-py3-none-any.whl.metadata (9.3 kB)\n",
            "Collecting monotonic>=1.5 (from posthog>=2.4.0->chromadb)\n",
            "  Downloading monotonic-1.6-py2.py3-none-any.whl.metadata (1.5 kB)\n",
            "Collecting backoff>=1.10.0 (from posthog>=2.4.0->chromadb)\n",
            "  Downloading backoff-2.2.1-py3-none-any.whl.metadata (14 kB)\n",
            "Requirement already satisfied: annotated-types>=0.6.0 in /usr/local/lib/python3.10/dist-packages (from pydantic>=1.9->chromadb) (0.7.0)\n",
            "Requirement already satisfied: pydantic-core==2.27.1 in /usr/local/lib/python3.10/dist-packages (from pydantic>=1.9->chromadb) (2.27.1)\n",
            "Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich>=10.11.0->chromadb) (3.0.0)\n",
            "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from rich>=10.11.0->chromadb) (2.18.0)\n",
            "Requirement already satisfied: huggingface-hub<1.0,>=0.16.4 in /usr/local/lib/python3.10/dist-packages (from tokenizers<=0.20.3,>=0.13.2->chromadb) (0.27.0)\n",
            "Requirement already satisfied: click>=8.0.0 in /usr/local/lib/python3.10/dist-packages (from typer>=0.9.0->chromadb) (8.1.7)\n",
            "Requirement already satisfied: shellingham>=1.3.0 in /usr/local/lib/python3.10/dist-packages (from typer>=0.9.0->chromadb) (1.5.4)\n",
            "Collecting httptools>=0.6.3 (from uvicorn[standard]>=0.18.3->chromadb)\n",
            "  Downloading httptools-0.6.4-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.6 kB)\n",
            "Collecting python-dotenv>=0.13 (from uvicorn[standard]>=0.18.3->chromadb)\n",
            "  Downloading python_dotenv-1.0.1-py3-none-any.whl.metadata (23 kB)\n",
            "Collecting uvloop!=0.15.0,!=0.15.1,>=0.14.0 (from uvicorn[standard]>=0.18.3->chromadb)\n",
            "  Downloading uvloop-0.21.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.9 kB)\n",
            "Collecting watchfiles>=0.13 (from uvicorn[standard]>=0.18.3->chromadb)\n",
            "  Downloading watchfiles-1.0.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.9 kB)\n",
            "Requirement already satisfied: websockets>=10.4 in /usr/local/lib/python3.10/dist-packages (from uvicorn[standard]>=0.18.3->chromadb) (14.1)\n",
            "Requirement already satisfied: cachetools<6.0,>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from google-auth>=1.0.1->kubernetes>=28.1.0->chromadb) (5.5.0)\n",
            "Requirement already satisfied: pyasn1-modules>=0.2.1 in /usr/local/lib/python3.10/dist-packages (from google-auth>=1.0.1->kubernetes>=28.1.0->chromadb) (0.4.1)\n",
            "Requirement already satisfied: rsa<5,>=3.1.4 in /usr/local/lib/python3.10/dist-packages (from google-auth>=1.0.1->kubernetes>=28.1.0->chromadb) (4.9)\n",
            "Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.16.4->tokenizers<=0.20.3,>=0.13.2->chromadb) (3.16.1)\n",
            "Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.16.4->tokenizers<=0.20.3,>=0.13.2->chromadb) (2024.10.0)\n",
            "Requirement already satisfied: zipp>=3.20 in /usr/local/lib/python3.10/dist-packages (from importlib-metadata<=8.5.0,>=6.0->opentelemetry-api>=1.2.0->chromadb) (3.21.0)\n",
            "Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich>=10.11.0->chromadb) (0.1.2)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->kubernetes>=28.1.0->chromadb) (3.4.0)\n",
            "Requirement already satisfied: sniffio>=1.1 in /usr/local/lib/python3.10/dist-packages (from anyio->httpx>=0.27.0->chromadb) (1.3.1)\n",
            "Requirement already satisfied: exceptiongroup in /usr/local/lib/python3.10/dist-packages (from anyio->httpx>=0.27.0->chromadb) (1.2.2)\n",
            "Collecting humanfriendly>=9.1 (from coloredlogs->onnxruntime>=1.14.1->chromadb)\n",
            "  Downloading humanfriendly-10.0-py2.py3-none-any.whl.metadata (9.2 kB)\n",
            "Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.10/dist-packages (from sympy->onnxruntime>=1.14.1->chromadb) (1.3.0)\n",
            "Requirement already satisfied: pyasn1<0.7.0,>=0.4.6 in /usr/local/lib/python3.10/dist-packages (from pyasn1-modules>=0.2.1->google-auth>=1.0.1->kubernetes>=28.1.0->chromadb) (0.6.1)\n",
            "Downloading chromadb-0.5.23-py3-none-any.whl (628 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m628.3/628.3 kB\u001b[0m \u001b[31m19.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading chroma_hnswlib-0.7.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.4 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m2.4/2.4 MB\u001b[0m \u001b[31m59.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading bcrypt-4.2.1-cp39-abi3-manylinux_2_28_x86_64.whl (278 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m278.6/278.6 kB\u001b[0m \u001b[31m19.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading build-1.2.2.post1-py3-none-any.whl (22 kB)\n",
            "Downloading fastapi-0.115.6-py3-none-any.whl (94 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m94.8/94.8 kB\u001b[0m \u001b[31m7.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading kubernetes-31.0.0-py2.py3-none-any.whl (1.9 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m1.9/1.9 MB\u001b[0m \u001b[31m54.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading mmh3-5.0.1-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (93 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m93.2/93.2 kB\u001b[0m \u001b[31m7.6 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading onnxruntime-1.20.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (13.3 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m13.3/13.3 MB\u001b[0m \u001b[31m89.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading opentelemetry_exporter_otlp_proto_grpc-1.29.0-py3-none-any.whl (18 kB)\n",
            "Downloading opentelemetry_exporter_otlp_proto_common-1.29.0-py3-none-any.whl (18 kB)\n",
            "Downloading opentelemetry_proto-1.29.0-py3-none-any.whl (55 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m55.8/55.8 kB\u001b[0m \u001b[31m4.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading opentelemetry_instrumentation_fastapi-0.50b0-py3-none-any.whl (12 kB)\n",
            "Downloading opentelemetry_instrumentation-0.50b0-py3-none-any.whl (30 kB)\n",
            "Downloading opentelemetry_instrumentation_asgi-0.50b0-py3-none-any.whl (16 kB)\n",
            "Downloading opentelemetry_util_http-0.50b0-py3-none-any.whl (6.9 kB)\n",
            "Downloading overrides-7.7.0-py3-none-any.whl (17 kB)\n",
            "Downloading posthog-3.7.4-py2.py3-none-any.whl (54 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m54.8/54.8 kB\u001b[0m \u001b[31m4.4 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading tokenizers-0.20.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.0 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m3.0/3.0 MB\u001b[0m \u001b[31m52.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading uvicorn-0.34.0-py3-none-any.whl (62 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m62.3/62.3 kB\u001b[0m \u001b[31m5.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading backoff-2.2.1-py3-none-any.whl (15 kB)\n",
            "Downloading durationpy-0.9-py3-none-any.whl (3.5 kB)\n",
            "Downloading httptools-0.6.4-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (442 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m442.1/442.1 kB\u001b[0m \u001b[31m28.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading monotonic-1.6-py2.py3-none-any.whl (8.2 kB)\n",
            "Downloading protobuf-5.29.2-cp38-abi3-manylinux2014_x86_64.whl (319 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m319.7/319.7 kB\u001b[0m \u001b[31m20.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading python_dotenv-1.0.1-py3-none-any.whl (19 kB)\n",
            "Downloading starlette-0.41.3-py3-none-any.whl (73 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m73.2/73.2 kB\u001b[0m \u001b[31m6.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading uvloop-0.21.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.8 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m3.8/3.8 MB\u001b[0m \u001b[31m58.9 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading watchfiles-1.0.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (443 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m443.8/443.8 kB\u001b[0m \u001b[31m28.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading coloredlogs-15.0.1-py2.py3-none-any.whl (46 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m46.0/46.0 kB\u001b[0m \u001b[31m3.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading pyproject_hooks-1.2.0-py3-none-any.whl (10 kB)\n",
            "Downloading asgiref-3.8.1-py3-none-any.whl (23 kB)\n",
            "Downloading humanfriendly-10.0-py2.py3-none-any.whl (86 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m86.8/86.8 kB\u001b[0m \u001b[31m6.8 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hBuilding wheels for collected packages: pypika\n",
            "  Building wheel for pypika (pyproject.toml) ... \u001b[?25l\u001b[?25hdone\n",
            "  Created wheel for pypika: filename=PyPika-0.48.9-py2.py3-none-any.whl size=53725 sha256=82139c104e3af3853060094edd7c04cf3a3439c0ae10a719e7a0547e062a7028\n",
            "  Stored in directory: /root/.cache/pip/wheels/e1/26/51/d0bffb3d2fd82256676d7ad3003faea3bd6dddc9577af665f4\n",
            "Successfully built pypika\n",
            "Installing collected packages: pypika, monotonic, durationpy, uvloop, uvicorn, python-dotenv, pyproject_hooks, protobuf, overrides, opentelemetry-util-http, mmh3, humanfriendly, httptools, chroma-hnswlib, bcrypt, backoff, asgiref, watchfiles, starlette, posthog, opentelemetry-proto, coloredlogs, build, tokenizers, opentelemetry-exporter-otlp-proto-common, onnxruntime, kubernetes, fastapi, opentelemetry-instrumentation, opentelemetry-instrumentation-asgi, opentelemetry-exporter-otlp-proto-grpc, opentelemetry-instrumentation-fastapi, chromadb\n",
            "  Attempting uninstall: protobuf\n",
            "    Found existing installation: protobuf 4.25.5\n",
            "    Uninstalling protobuf-4.25.5:\n",
            "      Successfully uninstalled protobuf-4.25.5\n",
            "  Attempting uninstall: tokenizers\n",
            "    Found existing installation: tokenizers 0.21.0\n",
            "    Uninstalling tokenizers-0.21.0:\n",
            "      Successfully uninstalled tokenizers-0.21.0\n",
            "\u001b[31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.\n",
            "tensorflow 2.17.1 requires protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3, but you have protobuf 5.29.2 which is incompatible.\n",
            "tensorflow-metadata 1.13.1 requires protobuf<5,>=3.20.3, but you have protobuf 5.29.2 which is incompatible.\n",
            "transformers 4.47.0 requires tokenizers<0.22,>=0.21, but you have tokenizers 0.20.3 which is incompatible.\u001b[0m\u001b[31m\n",
            "\u001b[0mSuccessfully installed asgiref-3.8.1 backoff-2.2.1 bcrypt-4.2.1 build-1.2.2.post1 chroma-hnswlib-0.7.6 chromadb-0.5.23 coloredlogs-15.0.1 durationpy-0.9 fastapi-0.115.6 httptools-0.6.4 humanfriendly-10.0 kubernetes-31.0.0 mmh3-5.0.1 monotonic-1.6 onnxruntime-1.20.1 opentelemetry-exporter-otlp-proto-common-1.29.0 opentelemetry-exporter-otlp-proto-grpc-1.29.0 opentelemetry-instrumentation-0.50b0 opentelemetry-instrumentation-asgi-0.50b0 opentelemetry-instrumentation-fastapi-0.50b0 opentelemetry-proto-1.29.0 opentelemetry-util-http-0.50b0 overrides-7.7.0 posthog-3.7.4 protobuf-5.29.2 pypika-0.48.9 pyproject_hooks-1.2.0 python-dotenv-1.0.1 starlette-0.41.3 tokenizers-0.20.3 uvicorn-0.34.0 uvloop-0.21.0 watchfiles-1.0.3\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import chromadb"
      ],
      "metadata": {
        "id": "NJQe7um-H8UE"
      },
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#create"
      ],
      "metadata": {
        "id": "rDhtQVM2HI1A"
      },
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "client = chromadb.Client()\n",
        "collection = client.create_collection(name=\"my_collection1\")"
      ],
      "metadata": {
        "id": "DUTix2qZIGOZ"
      },
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(collection)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "p8IYfwRdIJ9p",
        "outputId": "075487ab-ee66-43c7-d849-18689578a5cb"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collection(name=my_collection1)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# add"
      ],
      "metadata": {
        "id": "BKZ0FNlvHCdo"
      },
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "collection.add(\n",
        "    documents=[\n",
        "        \"This document is about New Delhi\",\n",
        "        \"This document is about New York\",\n",
        "\n",
        "    ],\n",
        "    ids=['id1','id2']\n",
        ")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "PcZBYrSFIMZV",
        "outputId": "a019fb2a-edd9-4393-ac2e-dcb9b462fd3a"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/root/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx.tar.gz: 100%|██████████| 79.3M/79.3M [00:01<00:00, 78.9MiB/s]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# read"
      ],
      "metadata": {
        "id": "PTdhHbOfHAYu"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "collection.get()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "CFQZkcawIOzQ",
        "outputId": "afb314b4-95a0-436d-b3b2-89c4cc69d3ff"
      },
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "{'ids': ['id1', 'id2'],\n",
              " 'embeddings': None,\n",
              " 'documents': ['This document is about New Delhi',\n",
              "  'This document is about New York'],\n",
              " 'uris': None,\n",
              " 'data': None,\n",
              " 'metadatas': [None, None],\n",
              " 'included': [<IncludeEnum.documents: 'documents'>,\n",
              "  <IncludeEnum.metadatas: 'metadatas'>]}"
            ]
          },
          "metadata": {},
          "execution_count": 9
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Update"
      ],
      "metadata": {
        "id": "itJszCG3i-0H"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "collection.upsert(documents=['This document is about Bangalore'],ids=['id3'])"
      ],
      "metadata": {
        "id": "E5c7NXNvIWR6"
      },
      "execution_count": 12,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "collection.get()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "WDn65maIIYmr",
        "outputId": "604e8fd1-fef1-46e6-f604-e0f714d4288c"
      },
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "{'ids': ['id1', 'id2', 'id3'],\n",
              " 'embeddings': None,\n",
              " 'documents': ['This document is about New Delhi',\n",
              "  'This document is about New York',\n",
              "  'This document is about Bangalore'],\n",
              " 'uris': None,\n",
              " 'data': None,\n",
              " 'metadatas': [None, None, None],\n",
              " 'included': [<IncludeEnum.documents: 'documents'>,\n",
              "  <IncludeEnum.metadatas: 'metadatas'>]}"
            ]
          },
          "metadata": {},
          "execution_count": 13
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Retrive using metadatas"
      ],
      "metadata": {
        "id": "24SQDypejGvN"
      },
      "execution_count": 1,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "res=collection.get()\n",
        "res['metadatas']"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "NZL6Hd5TIbJV",
        "outputId": "3dec2c32-efd6-4940-e500-02eaa14f8ef9"
      },
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "[None, None, None]"
            ]
          },
          "metadata": {},
          "execution_count": 14
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "res['uris']"
      ],
      "metadata": {
        "id": "ezQOIYbw2xoe"
      },
      "execution_count": 15,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# update with metadatas"
      ],
      "metadata": {
        "id": "F-_Ez5GRjRek"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "collection.upsert(\n",
        "    documents=['This document is about Mumbai'],\n",
        "    metadatas=[{'city': 'Mumbai', 'topic': 'Travel'}],\n",
        "    ids=['id3']\n",
        ")\n"
      ],
      "metadata": {
        "id": "UCce77XcIdhv"
      },
      "execution_count": 16,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "res=collection.get()\n",
        "res['metadatas']"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "QqycbPVYIgBB",
        "outputId": "15d3a05f-4c5f-4ba4-d13c-5ccb5435b482"
      },
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "[None, None, {'city': 'Mumbai', 'topic': 'Travel'}]"
            ]
          },
          "metadata": {},
          "execution_count": 17
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "res['uris']"
      ],
      "metadata": {
        "id": "gv7I8CKG27OY"
      },
      "execution_count": 18,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "collection.get()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Uv2_0qOmIiq_",
        "outputId": "cc2a9dc5-e1cd-4f4d-9aef-3835740a11c9"
      },
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "{'ids': ['id1', 'id2', 'id3'],\n",
              " 'embeddings': None,\n",
              " 'documents': ['This document is about New Delhi',\n",
              "  'This document is about New York',\n",
              "  'This document is about Mumbai'],\n",
              " 'uris': None,\n",
              " 'data': None,\n",
              " 'metadatas': [None, None, {'city': 'Mumbai', 'topic': 'Travel'}],\n",
              " 'included': [<IncludeEnum.documents: 'documents'>,\n",
              "  <IncludeEnum.metadatas: 'metadatas'>]}"
            ]
          },
          "metadata": {},
          "execution_count": 19
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# update with metadatas"
      ],
      "metadata": {
        "id": "8VqJlYGFHOU_"
      },
      "execution_count": 20,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "collection.upsert(documents=['This document is about New Delhi'],metadatas=[{'city':'Delhi','topic':'weather'}],ids=['id1'])"
      ],
      "metadata": {
        "id": "GOuDWsgsIlA8"
      },
      "execution_count": 21,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "collection.get()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "h8lndvZCKN8p",
        "outputId": "19b8dfc6-cfd8-4b76-c788-f8c35c9aa6a1"
      },
      "execution_count": 22,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "{'ids': ['id1', 'id2', 'id3'],\n",
              " 'embeddings': None,\n",
              " 'documents': ['This document is about New Delhi',\n",
              "  'This document is about New York',\n",
              "  'This document is about Mumbai'],\n",
              " 'uris': None,\n",
              " 'data': None,\n",
              " 'metadatas': [{'city': 'Delhi', 'topic': 'weather'},\n",
              "  None,\n",
              "  {'city': 'Mumbai', 'topic': 'Travel'}],\n",
              " 'included': [<IncludeEnum.documents: 'documents'>,\n",
              "  <IncludeEnum.metadatas: 'metadatas'>]}"
            ]
          },
          "metadata": {},
          "execution_count": 22
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "collection.upsert(documents=['This document is about Bangalore'],metadatas=[{'city':'Bangalore','topic':'traffic'}],ids=['id2'])"
      ],
      "metadata": {
        "id": "Pc8eJHUdKbEo"
      },
      "execution_count": 23,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "collection.get()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "b0VH6M5YKd-R",
        "outputId": "c9964010-0931-4366-d658-db89d2a7f13e"
      },
      "execution_count": 24,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "{'ids': ['id1', 'id2', 'id3'],\n",
              " 'embeddings': None,\n",
              " 'documents': ['This document is about New Delhi',\n",
              "  'This document is about Bangalore',\n",
              "  'This document is about Mumbai'],\n",
              " 'uris': None,\n",
              " 'data': None,\n",
              " 'metadatas': [{'city': 'Delhi', 'topic': 'weather'},\n",
              "  {'city': 'Bangalore', 'topic': 'traffic'},\n",
              "  {'city': 'Mumbai', 'topic': 'Travel'}],\n",
              " 'included': [<IncludeEnum.documents: 'documents'>,\n",
              "  <IncludeEnum.metadatas: 'metadatas'>]}"
            ]
          },
          "metadata": {},
          "execution_count": 24
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## delete using id and metadatas"
      ],
      "metadata": {
        "id": "5AsB1cBLKiLQ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "collection.delete(where={'city':'Delhi'})"
      ],
      "metadata": {
        "id": "RQ29Sm9LKf5w"
      },
      "execution_count": 25,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "collection.get()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "9AhXW6tIKkdz",
        "outputId": "8efc8e72-16a9-41fb-d55c-1f5cf75e3842"
      },
      "execution_count": 26,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "{'ids': ['id2', 'id3'],\n",
              " 'embeddings': None,\n",
              " 'documents': ['This document is about Bangalore',\n",
              "  'This document is about Mumbai'],\n",
              " 'uris': None,\n",
              " 'data': None,\n",
              " 'metadatas': [{'city': 'Bangalore', 'topic': 'traffic'},\n",
              "  {'city': 'Mumbai', 'topic': 'Travel'}],\n",
              " 'included': [<IncludeEnum.documents: 'documents'>,\n",
              "  <IncludeEnum.metadatas: 'metadatas'>]}"
            ]
          },
          "metadata": {},
          "execution_count": 26
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Read or Retrieve in Chromadb"
      ],
      "metadata": {
        "id": "Tg_DUtgIKmTc"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "collection.get(where={'city':'Bangalore'})"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "LRjBk2-u3I_8",
        "outputId": "d77eb806-d668-4ec1-d042-eea3b5d2bf78"
      },
      "execution_count": 27,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "{'ids': ['id2'],\n",
              " 'embeddings': None,\n",
              " 'documents': ['This document is about Bangalore'],\n",
              " 'uris': None,\n",
              " 'data': None,\n",
              " 'metadatas': [{'city': 'Bangalore', 'topic': 'traffic'}],\n",
              " 'included': [<IncludeEnum.documents: 'documents'>,\n",
              "  <IncludeEnum.metadatas: 'metadatas'>]}"
            ]
          },
          "metadata": {},
          "execution_count": 27
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "result = collection.get()\n",
        "print(\"Documents:\", result['documents'])\n",
        "print(\"Metadatas:\", result['metadatas'])\n",
        "print(\"IDs:\", result['ids'])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "-a9XAwS1Ko45",
        "outputId": "ee3dde72-0b35-4d8d-e5c8-4debe04d9be4"
      },
      "execution_count": 28,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Documents: ['This document is about Bangalore', 'This document is about Mumbai']\n",
            "Metadatas: [{'city': 'Bangalore', 'topic': 'traffic'}, {'city': 'Mumbai', 'topic': 'Travel'}]\n",
            "IDs: ['id2', 'id3']\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "result = collection.get(ids=[\"id1\"])\n",
        "print(\"Filtered Documents:\", result['documents'])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8aXpo1FfKrEf",
        "outputId": "d024f9a0-53f7-4e11-8c79-7fe341718bbc"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Filtered Documents: []\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "collection.delete(where={'ids': 'id3'})\n",
        "print(collection.get())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "JDh73RUbKsh0",
        "outputId": "e39db9e5-72db-48ec-f173-f7771eff79ad"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "{'ids': ['id2', 'id3'], 'embeddings': None, 'documents': ['This document is about Bangalore', 'This document is about Mumbai'], 'uris': None, 'data': None, 'metadatas': [{'city': 'Bangalore', 'topic': 'traffic'}, {'city': 'Mumbai', 'topic': 'Travel'}], 'included': [<IncludeEnum.documents: 'documents'>, <IncludeEnum.metadatas: 'metadatas'>]}\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "collection.delete(ids=[ 'id3'])"
      ],
      "metadata": {
        "id": "WtEzrUe2M9Er"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "collection.get()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "xojfF4AbNBtW",
        "outputId": "fc9c05ce-064c-4399-ed5e-aba8aca8d488"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "{'ids': ['id2'],\n",
              " 'embeddings': None,\n",
              " 'documents': ['This document is about Bangalore'],\n",
              " 'uris': None,\n",
              " 'data': None,\n",
              " 'metadatas': [{'city': 'Bangalore', 'topic': 'traffic'}],\n",
              " 'included': [<IncludeEnum.documents: 'documents'>,\n",
              "  <IncludeEnum.metadatas: 'metadatas'>]}"
            ]
          },
          "metadata": {},
          "execution_count": 35
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "5s16ygr4NERE"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}
