from dotenv import load_dotenv
import os 
import boto3


def read_s3(s3file_name):
    s3 = boto3.client("s3")

    bucket_name = os.getenv("S3_BUCKET_NAME")

    response = s3.get_object(
        Bucket=bucket_name,
        Key=f"data/{s3file_name}"
    )
    return response["Body"].read()
        
        
def write_s3(local_file, s3file_name):
    s3 = boto3.client("s3")
    
    bucket_name = os.getenv("S3_BUCKET_NAME")
    
    s3.upload_file(local_file, bucket_name, f"output/{s3file_name}")