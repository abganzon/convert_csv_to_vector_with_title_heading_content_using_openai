# This is a sample Python script.
import json
import math

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import openai
import pandas as pd
import numpy as np
import pickle
from transformers import GPT2TokenizerFast
from typing import List, Dict, Tuple
import pandas as pd
import csv
import pinecone
import json
import os
import chardet

openai.api_key = "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"
COMPLETIONS_MODEL = "text-davinci-003"

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    with open('C:/Users/Ferret9 PC/PycharmProjects/segmentTextOpenAI/pinebrooke-output-final.csv', 'rb') as f:
        result = chardet.detect(f.read())

    df = pd.read_csv('C:/Users/Ferret9 PC/PycharmProjects/segmentTextOpenAI/pinebrooke-output-final.csv',
                     encoding=result['encoding'])

    #for converting csv to vector data
    # df = pd.read_csv('https://cdn.openai.com/API/examples/data/olympics_sections_text.csv')
    # df = pd.read_csv('C:/Users/Ferret9 PC/PycharmProjects/segmentTextOpenAI/Pinebrooke_Output_CSV.csv')
    df = df.set_index(["title", "heading"])
    # MODEL_NAME = "curie"
    MODEL_NAME = "ada"

    DOC_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-doc-001"
    QUERY_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-query-001"


    def get_embedding(text: str, model: str) -> List[float]:
        result = openai.Embedding.create(
            model=model,
            input=text)
        return result["data"][0]["embedding"]

    def get_doc_embedding(text: str) -> List[float]:
        return get_embedding(text, DOC_EMBEDDINGS_MODEL)

    def get_query_embedding(text: str) -> List[float]:
        return get_embedding(text, QUERY_EMBEDDINGS_MODEL)

    #working nani
    # def compute_doc_embeddings(df: pd.DataFrame) -> Dict[Tuple[str, str], List[float]]:
    #     return {
    #         idx: get_doc_embedding(r.content.replace("\n", " ")) for idx, r in df.iterrows()
    #     }

    def compute_doc_embeddings(df: pd.DataFrame) -> Dict[Tuple[str, str], List[float]]:
        return {
            (title, heading, content): get_doc_embedding(str(content).replace("\n", " ")) for idx, (title, heading, content) in
            enumerate(df.itertuples(name=None))
        }


    # def compute_doc_embeddings(df: pd.DataFrame, num_cols: int) -> Dict[Tuple[str, str], List[float]]:
    #     return {
    #         idx: get_doc_embedding(r.content.replace("\n", " "))[:num_cols] for idx, r in df.iterrows()
    #     }

    def write_doc_embeddings_to_csv(embeddings, filename):
        data = []
        for title_heading, embedding in embeddings.items():
            title, heading = title_heading
            data.append([title, heading] + embedding)

        df = pd.DataFrame(data, columns=['title', 'heading'] + [f'{i}' for i in range(len(embedding))])
        df.to_csv(filename, index=False)


    # def write_doc_embeddings_to_pinecone(embeddings: Dict[Tuple[str, str], List[float]]):
    #     data = []
    #     pinecone.init(api_key="93994d88-dbe1-4531-8dc7-2d61aba71f9b", environment="us-east1-gcp")
    #     index_name = "embeddings-index-from-python"
    #     pinecone.create_index(name=index_name, dimension=4096, metric="cosine", shards=1)
    #     index = pinecone.Index(index_name)
    #
    #     for i, (title_heading, embedding) in enumerate(embeddings.items()):
    #         title, heading = title_heading
    #         embedding_list = [float(val) for val in embedding]
    #         metadata = {"title": title, "heading": heading}
    #         #(id, vector, metadata)
    #         data.append((str(i), embedding_list, metadata))
    #
    #     success = index.upsert(data)
    #     print(success)
    #     index.flush()
    #     pinecone.deinit()
    #
    #     if all(success):
    #         print("All embeddings were successfully upserted!")
    #     else:
    #         print("Error: Some embeddings were not upserted.")

    #working nani
    # def write_doc_embeddings_to_pinecone(embeddings: Dict[Tuple[str, str], List[float]]):
    #     data = []
    #     for i, (title_heading, embedding) in enumerate(embeddings.items()):
    #         title, heading = title_heading
    #         if isinstance(embedding, list) and all(isinstance(val, float) for val in embedding):
    #             metadata = {
    #                 "title": f"{title}",
    #                 "heading": f"{heading}",
    #             }
    #             vector = {
    #                 "id": f"{i}",
    #                 "metadata": metadata,
    #                 "values": [float(val) for val in embedding],
    #             }
    #             data.append(vector)
    #
    #     index_data = {
    #         "vectors": data,
    #         "namespace": "example_namespace"
    #     }
    #     json_str = json.dumps(index_data)
    #     print(json_str)
    #     pinecone.init(api_key="93994d88-dbe1-4531-8dc7-2d61aba71f9b", environment="us-east1-gcp")
    #     index_name = "embeddings-index-from-python"
    #     # pinecone.create_index(name=index_name, dimension=4096, metric="cosine", shards=0)
    #     index = pinecone.Index(index_name)
    #     index.upsert(json_str)
    #
    #     with open("output.json", 'w') as f:
    #         f.write(json_str)
    #
    #     return json_str

    def write_doc_embeddings_to_pinecone(embeddings: Dict[Tuple[str, str], List[float]]):
        data = []
        for i, ((title, heading, content), embedding) in enumerate(embeddings.items()):
            if isinstance(embedding, list) and all(isinstance(val, float) for val in embedding):
                # print("Title: ",title)
                title_parts = str(title).split("',")
                # print("Title Parts: ",title_parts)
                # Slice the embedding to 2048 elements
                embedding = embedding#[:2048]
                # Split the title by comma
                # title_parts = str(title).split(',')

                # Set the first part of the title as the title metadata
                primary_title = title_parts[0].strip()
                # Set the second part of the title as the second_title metadata
                secondary_title = title_parts[1].strip() if len(title_parts) > 1 else ''
                # metadata = {
                #     "title": f"{primary_title}",
                #     "heading": f"{secondary_title}",
                #     "content": f"{heading}",
                # }
                metadata = {
                    "title": primary_title.replace("('",""),
                    "heading": secondary_title.replace("'","").replace(")",""),
                    "content": f"{heading}",
                }
                vector = {
                    "id": f"{i}",
                    "metadata": metadata,
                    "values": [float(val) for val in embedding],
                }
                data.append(vector)

        index_data = {
            "vectors": data,
            "namespace": "pinebrooke"
        }
        json_str = json.dumps(index_data)
        with open("pinebrooke-output-final-new-6.json", 'w') as f:
            f.write(json_str)

        return json_str


    doc_embeddings = compute_doc_embeddings(df.sample(30))
    #doc_embeddings = compute_doc_embeddings(df.sample(n=30).sort_values('title', ascending=True))
    write_doc_embeddings_to_pinecone(doc_embeddings)

    # print(insert_embeddings)
    # print(json.dumps(write_doc_embeddings_to_pinecone(doc_embeddings)))
    # write_doc_embeddings_to_csv(doc_embeddings, 'doc_embeddings_all.csv')

    # pinecone.init(api_key="93994d88-dbe1-4531-8dc7-2d61aba71f9b", environment="us-east1-gcp")
    # index_name = "embeddings-index-from-python"
    # pinecone.create_index(name=index_name, dimension=4095, metric="cosine", shards=0)
    # index = pinecone.Index(index_name)
    #
    # with open('C:/Users/Ferret9 PC/PycharmProjects/trntyOpenAI/doc_embeddings_limit_one_row_and_index.csv', newline='') as csvfile:
    #     reader = csv.DictReader(csvfile)
    #     for row in reader:
    #         title = row['title']
    #         heading = row['heading']
    #         embeddings = list(map(float, row['embeddings'].split(',')))
    #         index.upsert(ids=title, vectors=embeddings, metadata={'heading': heading})
    #
    # index.flush()
    # pinecone.deinit()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
