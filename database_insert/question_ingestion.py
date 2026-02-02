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
# Logging setup
logging.basicConfig(
    filename="ingest_questions.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Supabase client setup
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Key variables
CONTENT_ROOT = Path("../data/processed")
BATCH_SIZE = 10
MAX_RETRIES = 3  # for transient failures

DIFFICULTY_MAP = {
    "beginner": "basic",
    "intermediate": "core",
    "advanced": "mastery",  # changed advanced → mastery
}

# Retry decorator for idempotent inserts
@retry(tries=MAX_RETRIES, delay=2)
def safe_upsert(table_name, rows):
    try:
        return supabase.table(table_name).upsert(
            rows, on_conflict="question_hash"
        ).execute()
    except Exception as e:
        logging.error(f"Upsert failed for {table_name}: {e}")
        raise

# Helper: upsert topic/subtopic and return ID
def get_or_create_topic(topic_name):
    try:
        supabase.table("topics").upsert(
            {"topic_name": topic_name}, on_conflict=["topic_name"]
        ).execute()

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
            {"topic_id": topic_id, "subtopic_name": subtopic_name},
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
        logging.error(f"Failed to get/create subtopic '{subtopic_name}' (topic_id={topic_id}): {e}")
        raise

# Helper: hash individual question content
def hash_content(data: dict) -> str:
    normalized = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

# Ingestion
def ingest():
    topics = [t for t in CONTENT_ROOT.iterdir() if t.is_dir()]

    for topic_folder in topics:
        topic_name = topic_folder.name
        topic_id = get_or_create_topic(topic_name)
        logging.info(f"Processing topic: {topic_name} (ID: {topic_id})")

        subtopics = [s for s in topic_folder.iterdir() if s.is_dir()]
        for subtopic_folder in subtopics:
            subtopic_name = subtopic_folder.name
            subtopic_id = get_or_create_subtopic(topic_id, subtopic_name)
            logging.info(f"Processing subtopic: {subtopic_name} (ID: {subtopic_id})")

            yaml_file = subtopic_folder / "questions_by_level.yaml"
            if not yaml_file.exists():
                logging.warning(f"No questions_by_level.yaml in {subtopic_name}")
                continue

            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    if not isinstance(data, dict) or "by_level" not in data:
                        logging.error(f"Invalid YAML structure in {yaml_file}")
                        continue
            except yaml.YAMLError as e:
                logging.error(f"Failed to parse {yaml_file}: {e}")
                continue

            batch_rows = []
            levels = data["by_level"]

            for yaml_level, db_level in DIFFICULTY_MAP.items():
                if yaml_level not in levels:
                    continue

                questions = levels[yaml_level].get("questions", [])
                for q in questions:
                    question_type = q.get("type")
                    if not question_type:
                        logging.warning(f"Skipping question with missing type in {yaml_file}")
                        continue

                    content_hash = hash_content(q)

                    batch_rows.append({
                        "subtopic_id": subtopic_id,
                        "difficulty": db_level,
                        "question_type": question_type,
                        "content_json": q,
                        "question_hash": content_hash,
                        "is_published": True,
                    })

                    if len(batch_rows) >= BATCH_SIZE:
                        safe_upsert("questions", batch_rows)
                        batch_rows = []

            if batch_rows:
                safe_upsert("questions", batch_rows)
                logging.info(f"Inserted final batch of {len(batch_rows)} questions for {subtopic_name}")

    logging.info("Questions ingestion completed")

# Run script
if __name__ == "__main__":
    ingest()
