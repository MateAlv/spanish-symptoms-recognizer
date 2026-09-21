import time
import csv
import torch
import torch.nn.functional as F
import os

from transformers import AutoTokenizer, AutoModel

# Embeddings generator
## Paths are anchored to this file so the script runs from any directory.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

## The encoder has to be the one the mapper loads, or it reads no batches.
SEMANTIC_MODEL_NAME = "clinlinker-kb-gp"
LOCAL_TOKENIZER_PATH = os.path.join(CURRENT_DIR, "semantic_model", SEMANTIC_MODEL_NAME)
LOCAL_MODEL_PATH = LOCAL_TOKENIZER_PATH

## Files for token generation
ORIGINAL_TSV_FILE = os.path.join(CURRENT_DIR, "hpo/hp-es.babelon.tsv")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "hpo/hpo_batches", SEMANTIC_MODEL_NAME)

## Columns
ORIGINAL_TSV_FILE_TRANSALTION_COLUMN = "translation_value"
ORIGINAL_TSV_FILE_HPO_CODE = "subject_id"

## Codes per saved .pt file. The mapper globs these, so the split is free.
BATCH_SIZE = 500

## Rows per forward pass. Encoding row by row is many times slower on CPU.
ENCODE_BATCH_SIZE = 64

def generate_tokens_file():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_TOKENIZER_PATH)
    model = AutoModel.from_pretrained(LOCAL_MODEL_PATH).to(device)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    hpo_codes_tokens = {}
    batch_count = 0

    with open(ORIGINAL_TSV_FILE, "r", encoding="utf-8") as original_data_file:
        input_tsv_reader = csv.DictReader(original_data_file, delimiter='\t')

        with torch.no_grad():
            print("Creación de tokens usando torch para guardar en lotes")
            start_time = time.time()
            
            pending_rows = []

            def flush_pending():
                """Encode the buffered rows in one forward pass."""
                if not pending_rows:
                    return

                texts = [text for _, text in pending_rows]
                tokenized = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(device)

                model_output = model(**tokenized)
                cls_embeddings = model_output.last_hidden_state[:, 0, :]
                normalized = F.normalize(cls_embeddings, p=2, dim=1).cpu()

                for (code, text), vector in zip(pending_rows, normalized):
                    hpo_codes_tokens[code] = {"vector": vector, "name": text}

                pending_rows.clear()

            for row in input_tsv_reader:
                clean_text = row[ORIGINAL_TSV_FILE_TRANSALTION_COLUMN].strip()
                if not clean_text:
                    continue

                pending_rows.append((row[ORIGINAL_TSV_FILE_HPO_CODE], clean_text))

                if len(pending_rows) == ENCODE_BATCH_SIZE:
                    flush_pending()

                if len(hpo_codes_tokens) >= BATCH_SIZE:
                    output_file_path = os.path.join(OUTPUT_DIR, f"hpo_batch_{batch_count}.pt")
                    torch.save(hpo_codes_tokens, output_file_path)
                    print(f"Guardado lote {batch_count} con {len(hpo_codes_tokens)} códigos.")

                    # Cleared in place: flush_pending() writes into this same
                    # dict, so rebinding it here would strand later batches.
                    hpo_codes_tokens.clear()
                    batch_count += 1

            flush_pending()

            if hpo_codes_tokens:
                output_file_path = os.path.join(OUTPUT_DIR, f"hpo_batch_{batch_count}.pt")
                torch.save(hpo_codes_tokens, output_file_path)
                print(f"Guardado lote final {batch_count} con {len(hpo_codes_tokens)} códigos.")

            print(f"Duración total: {time.time() - start_time}")

if __name__ == "__main__":
    generate_tokens_file()