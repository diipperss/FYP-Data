import os
import yaml
from pathlib import Path
from supabase import create_client, Client
from retry import retry
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

DIFFICULTY_MAP = {
    "beginner": "basic",
    "intermediate": "core",
    "advanced": "advanced",
}

#retry decorator for idemppotent isnerts
@retry(tries=MAX_RETRIES, delay=2)
def safe_upsert(table_name,rows):
    try:
        return supabase.table(table_name).upsert(rows, on_conflict="subtopic_id,difficulty").execute()
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
    normalized = json.dumps(data,sort_keys=True, separators=(",", ":"))
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

            batch_rows = []

            yaml_file = subtopic_folder / "chunks_by_level.yaml"
            if not yaml_file.exists():
                logging.warning(f"No chunks_by_level.yaml in {subtopic_name}")
                continue

            with open(yaml_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if not isinstance(data, dict):
                    logging.error(f"Invalid YAML structure in {yaml_file}")
                    continue

            for yaml_level, db_level in DIFFICULTY_MAP.items():
                if yaml_level not in data:
                    continue

                level_block = data[yaml_level]
                content_hash = hash_content(level_block)

                batch_rows.append({
                    "subtopic_id": subtopic_id,
                    "difficulty": db_level,
                    "title": level_block.get("title"),
                    "summary": level_block.get("summary"),
                    "content_json": level_block,
                    "content_hash": content_hash,
                    "is_published": True,
                })

            if batch_rows:
                safe_upsert("content", batch_rows)
                logging.info(f"Inserted final batch of {len(batch_rows)} chunks")
        
    logging.info("Ingestion completed")


#run script
if __name__ == "__main__":
    ingest()
