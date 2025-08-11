import re, json, pathlib, backoff, time
from tqdm import tqdm
from transformers import AutoTokenizer
from openai import OpenAI, RateLimitError, APIConnectionError          # :contentReference[oaicite:1]{index=1}
from pydantic import BaseModel
from typing import List, Dict
# import .env  # לטעינת משתני סביבה (כמו OPENAI_API_KEY) :contentReference[oaicite:5]{index=5}
from dotenv import load_dotenv
load_dotenv()  # טוען משתני סביבה מקובץ .env

# ---------- 1.  CONSTANTS ----------
MAX_TOK = 6_000                 # Reduced from 8192 to leave room for system prompt and formatting
MAX_SAMPLES = 300               # Maximum number of data points to create (set to None for unlimited)
SKIP_SAMPLES = 0               # Number of samples to skip from the beginning (NEW)
IN_LOG  = "wordpress-logs.txt"
OUT_DS  = "log_alpaca.jsonl"
MODEL   = "gpt-4o-mini-2024-07-18"     # Structured-Outputs ready  :contentReference[oaicite:2]{index=2}

client  = OpenAI()                       # KEY משתמש ב-ENV

tok = AutoTokenizer.from_pretrained("unsloth/tinyllama-bnb-4bit")

# ---------- 2.  REGEX CLEAN ----------
QS_RE = re.compile(r"\?.*?(GET|POST|HTTP/1\.1)")
UA_RE = re.compile(r'" [^"]+$')          # UA + Referer
def clean(l:str)->str:
    return UA_RE.sub("", QS_RE.sub("", l)).strip()

# ---------- 3.  TOKEN-AWARE CHUNKER ----------
def chunks(path:str, max_tok:int=MAX_TOK):
    buf, n = [], 0
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            ln = clean(ln)
            t  = len(tok(ln).input_ids)
            if t > max_tok:                 # שורה חריגה – דילוג
                continue
            if n+t > max_tok and buf:
                yield "\n".join(buf); buf,n=[],0
            buf.append(ln); n += t
    if buf: yield "\n".join(buf)

# ---------- 4.  JSON SCHEMA (Pydantic) ----------
class CodeStat(BaseModel):
    code:int; count:int|str; description:str
class Summary(BaseModel):
    time_range:str
    status_codes:List[CodeStat]
    traffic_sources:Dict[str,str]|None=None
    resource_types:Dict[str,str]|None=None
    admin_activity:Dict[str,str]|None=None
    observations:Dict[str,str]|None=None
    recommendations:List[str]|None=None
class LogSynopsis(BaseModel):
    summary:Summary
LogSynopsis.model_config={"json_schema_extra":{"required":["summary"]}}
Summary.model_config    ={"json_schema_extra":{"required":["time_range","status_codes"]}}
CodeStat.model_config   ={"json_schema_extra":{"required":["code","count","description"]}}

# ---------- 5.  GPT CALL WITH RETRY ----------
@backoff.on_exception(backoff.expo, (RateLimitError, APIConnectionError), max_tries=5)  # :contentReference[oaicite:3]{index=3}
def gpt_summarise(block:str)->str:
    rsp = client.responses.parse(
        model=MODEL,
        input=[
            {"role":"system","content":(
                "You are NetLogGPT. Respond ONLY with JSON valid for the schema.")},
            {"role":"user","content":f"<LOG>\n{block}\n</LOG>"}
        ],
        text_format=LogSynopsis,        # Structured-Outputs helper :contentReference[oaicite:4]{index=4}
        max_output_tokens=1000,          # מגבלת יציאה
    )
    return rsp.output_parsed.model_dump_json()

# ---------- 6.  BUILD DATASET ----------
output_token_counts = []
input_token_counts = []
samples_created = 0
samples_skipped = 0
samples_processed = 0

print(f"🚀 Starting dataset creation...")
print(f"   📊 Max samples: {MAX_SAMPLES if MAX_SAMPLES else 'unlimited'}")
print(f"   ⏭️  Skip first: {SKIP_SAMPLES} samples")
print(f"   📁 Input file: {IN_LOG}")
print(f"   💾 Output file: {OUT_DS}")
print()

with pathlib.Path(OUT_DS).open("w", encoding="utf-8") as fout:
    for block in tqdm(chunks(IN_LOG), desc="Processing chunks"):
        samples_processed += 1
        
        # Skip the first X samples if specified
        if samples_skipped < SKIP_SAMPLES:
            samples_skipped += 1
            continue
        
        # Check if we've reached the maximum number of samples to create
        if MAX_SAMPLES is not None and samples_created >= MAX_SAMPLES:
            print(f"🛑  Reached maximum limit of {MAX_SAMPLES} samples")
            break
            
        try:
            # Calculate tokens for the input block
            input_tokens = len(tok(block).input_ids)
            input_token_counts.append(input_tokens)
            
            output_json = gpt_summarise(block)
            # Calculate tokens for this output
            output_tokens = len(tok(output_json).input_ids)
            output_token_counts.append(output_tokens)
            samples_created += 1
        except Exception as e:
            print("⚠️  skipped:", e); continue

        fout.write(json.dumps({
            "instruction":"Summarise the following log window and return ONLY valid JSON.",
            "input": block,
            "output": output_json
        }, ensure_ascii=False)+"\n")

# Calculate and display statistics
if output_token_counts and input_token_counts:
    # Output statistics
    avg_output_tokens = sum(output_token_counts) / len(output_token_counts)
    min_output_tokens = min(output_token_counts)
    max_output_tokens = max(output_token_counts)
    
    # Input statistics
    avg_input_tokens = sum(input_token_counts) / len(input_token_counts)
    min_input_tokens = min(input_token_counts)
    max_input_tokens = max(input_token_counts)
    
    total_outputs = len(output_token_counts)
    
    print(f"📊  Token Statistics:")
    print(f"   Samples processed: {samples_processed}")
    print(f"   Samples skipped: {samples_skipped}")
    print(f"   Samples created: {samples_created}/{MAX_SAMPLES if MAX_SAMPLES else 'unlimited'}")
    print(f"   Total samples: {total_outputs}")
    print(f"")
    print(f"   📥 INPUT TOKENS:")
    print(f"      Average: {avg_input_tokens:.2f}")
    print(f"      Min: {min_input_tokens}")
    print(f"      Max: {max_input_tokens}")
    print(f"      Total: {sum(input_token_counts):,}")
    print(f"")
    print(f"   📤 OUTPUT TOKENS:")
    print(f"      Average: {avg_output_tokens:.2f}")
    print(f"      Min: {min_output_tokens}")
    print(f"      Max: {max_output_tokens}")
    print(f"      Total: {sum(output_token_counts):,}")
    print(f"")
    print(f"   📊 COMPRESSION RATIO:")
    print(f"      Average: {avg_input_tokens/avg_output_tokens:.2f}:1")
    print(f"      Total tokens saved: {sum(input_token_counts) - sum(output_token_counts):,}")

print("✅  Dataset saved to", OUT_DS)
