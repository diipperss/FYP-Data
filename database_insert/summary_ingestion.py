import os
import yaml
from pathlib import Path
from supabase import create_client, Client
from retry import retry
from tqdm import tqdm #used to display smart progress meters for loops and long-running operations
import logging
from dotenv import load_dotenv
import hashlib
import json

load_dotenv()
#logging setup
logging.basicConfig(
    filename="ingest_chunks.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

#supabase client setup
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

#key variables
CONTENT_ROOT = Path("../data/processed")
BATCH_SIZE = 10
MAX_RETRIES = 3  # for transient failures

#retry decorator for idemppotent isnerts
@retry(tries=MAX_RETRIES, delay=2)
def safe_upsert(table_name,rows):
    try:
        return supabase.table(table_name).upsert(rows, on_conflict="content_hash").execute()
    except Exception as e:
        logging.error(f"Upsert failed for {table_name}: {e}")
        raise

#helper: upsert topic/subtopic and return id
def get_or_create_topic(topic_name):
    try:
        # Upsert (idempotent)
        supabase.table("topics").upsert(
            {"topic_name": topic_name},
            on_conflict=["topic_name"]
        ).execute()

        # Fetch ID
        resp = supabase.table("topics") \
            .select("topic_id") \
            .eq("topic_name", topic_name) \
            .single() \
            .execute()

        return resp.data["topic_id"]

    except Exception as e:
        logging.error(f"Failed to get/create topic '{topic_name}': {e}")
        raise


def get_or_create_subtopic(topic_id, subtopic_name):
    try:
        supabase.table("subtopics").upsert(
            {
                "topic_id": topic_id,
                "subtopic_name": subtopic_name
            },
            on_conflict="topic_id, subtopic_name"
        ).execute()

        resp = supabase.table("subtopics") \
            .select("subtopic_id") \
            .eq("topic_id", topic_id) \
            .eq("subtopic_name", subtopic_name) \
            .single() \
            .execute()

        return resp.data["subtopic_id"]

    except Exception as e:
        logging.error(
            f"Failed to get/create subtopic '{subtopic_name}' "
            f"(topic_id={topic_id}): {e}"
        )
        raise

#helper function for hashjing yaml content
def hash_content(data:dict) -> str:
    normalized = json.dumps(data,sort_keys=True)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

#ingestion
def ingest():
    #get the 3 main topics
    topics = [t for t in CONTENT_ROOT.iterdir() if t.is_dir()]

    #loop for each main topic
    for topic_folder in topics:
        topic_name = topic_folder.name
        topic_id = get_or_create_topic(topic_name)
        logging.info(f"Processing topic: {topic_name} (ID: {topic_id})")

        #repeat for the subtopics in each main topic
        subtopics = [s for s in topic_folder.iterdir() if s.is_dir()]
        for subtopic_folder in subtopics:
            subtopic_name = subtopic_folder.name
            subtopic_id = get_or_create_subtopic(topic_id, subtopic_name)
            logging.info(f"Processing subtopic: {subtopic_name} (ID: {subtopic_id})")

            yaml_files = [f for f in subtopic_folder.iterdir() if f.suffix == ".yaml"]

            #prepare batch
            batch_rows = []

            for yaml_file in tqdm(yaml_files, desc=f"Ingesting {subtopic_name}"):
                try:
                    with open(yaml_file, "r", encoding="utf-8") as f:
                        chunk_data = yaml.safe_load(f)
                        if not isinstance(chunk_data,dict):
                                logging.error(f"Invalid YAML structure in {yaml_file}")
                                continue
                except yaml.YAMLError as e:
                    logging.error(f"Failed to parse{yaml_file}:{e}")
                    continue

                content_hash = hash_content(chunk_data)

                batch_rows.append({
                    "topic_id":topic_id,
                    "subtopic_id":subtopic_id,
                    "summary_content":chunk_data,
                    "content_hash": content_hash,
                    "is_published":True
                })

                #insert batch
                if len(batch_rows)>= BATCH_SIZE:
                    try:
                        safe_upsert("subtopic_summary", batch_rows)
                        logging.info(f"Inserted batch of {len(batch_rows)} chunks")
                    except Exception as e:
                        logging.error(f"Batch insert failed:{e}")
                    
                    #clear batch_rows
                    batch_rows=[]


            if batch_rows:
                try:
                    safe_upsert("subtopic_summary", batch_rows)
                    logging.info(f"Inserted final batch of {len(batch_rows)} chunks")
                except Exception as e:
                    logging.error(f"Final batch insert failed: {e}")
        
    logging.info("Ingestion completed")


#run script
if __name__ == "__main__":
    ingest()
