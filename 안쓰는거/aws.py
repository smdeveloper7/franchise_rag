import boto3
from langchain_aws import BedrockEmbeddings

def get_bedrock_runtime():
    return boto3.client(
        service_name="bedrock-runtime",
        region_name="us-west-2"
        # aws_access_key_id=aws_config["access_key_id"],
        # aws_secret_access_key=aws_config["secret_access_key"]
    )

def get_bedrock_embeddings(client=None, model_id="amazon.titan-embed-text-v1"):
    """Bedrock 임베딩 클라이언트 생성"""
    if client is None:
        client = get_bedrock_runtime()
    return BedrockEmbeddings(
        client=client,
        model_id=model_id
    )