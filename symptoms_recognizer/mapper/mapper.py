from pathlib import Path
from scipy.spatial import distance
from transformers import AutoTokenizer, AutoModel

from symptoms_recognizer.llm import build_provider
from symptoms_recognizer.llm.base import LLMInvalidResponseError
from symptoms_recognizer.llm.json_parsing import parse_json_object
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import glob
import logging
import torch.nn.functional as F
import os
import torch
import heapq

logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).parent.resolve()
HPO_ONTOLOGY_CODE = "hpo"
ACCEPTED_ONTOLOGIES = { HPO_ONTOLOGY_CODE }
HPO_FILE_RELATIVE_PATH = "hpo/hpo_batches/clinlinker-kb-gp"
HPO_ABSOLUTE_INPUT_FILE_PATH = os.path.join(CURRENT_DIR, HPO_FILE_RELATIVE_PATH)
ACCEPTED_ONTOLOGIES_FILES = { HPO_ONTOLOGY_CODE : HPO_ABSOLUTE_INPUT_FILE_PATH }

LOCAL_MODEL_RELATIVE_PATH = "semantic_model/clinlinker-kb-gp"
DEFAULT_LOCAL_MODEL_PATH = os.path.join(CURRENT_DIR, LOCAL_MODEL_RELATIVE_PATH)
MIN_DISTANCE_VECTORS = 0.5
N_POOL_THREADS = 5

class PhenotypeOntologyMapper:
    def __init__(self, model_path=None, tokenizer_path=None, ontology=None, ontology_file_path=None, 
                 api_provider=None, api_model_name=None, top_k=10, prompt=None):
        
        self.pytorch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.api_provider = api_provider
        self.api_model_name = api_model_name
        self.top_k = top_k
        self.min_required_similarity = 1.0 - MIN_DISTANCE_VECTORS
        self.last_mapping_logs = []

        if self.api_provider:
            self.provider = build_provider(
                provider=self.api_provider, model=self.api_model_name
            )
        else:
            self.provider = None

        if ontology in ACCEPTED_ONTOLOGIES:
            self.mapped_ontology = ontology
            self.ontology_file_path = ACCEPTED_ONTOLOGIES_FILES[ontology]
        elif ontology and not ontology_file_path: raise Exception("TODO!")
        else: raise Exception("No ontology defined.")

        self.prompt = prompt
        model_p = model_path or DEFAULT_LOCAL_MODEL_PATH
        tok_p = tokenizer_path or DEFAULT_LOCAL_MODEL_PATH
        self.model = self._get_el_model(model_p).to(self.pytorch_device)
        self.tokenizer = self._get_el_tokenizer(tok_p)

    def _get_el_model(self, path): return AutoModel.from_pretrained(path)
    def _get_el_tokenizer(self, path): return AutoTokenizer.from_pretrained(path)

    def map_phenotypes(self, phenotypes_with_context: list[tuple[str, str]]):
        if not phenotypes_with_context:
            return []

        self.last_mapping_logs = []

        phenotypes_list = [item[0] for item in phenotypes_with_context]
        
        encoded_phenotypes_matrix = self._get_encoded_phenotypes_list(phenotypes_list)
        total_phenotypes = len(phenotypes_list)

        heaps = [[] for _ in range(total_phenotypes)]

        for hpo_codes_batch in self._get_codes_batch():
            codes_in_batch = list(hpo_codes_batch.keys())

            vectors_in_batch = torch.stack([hpo_codes_batch[c]["vector"] for c in codes_in_batch]).to(self.pytorch_device)
            names_in_batch = [hpo_codes_batch[c]["name"] for c in codes_in_batch]
            
            if vectors_in_batch.dim() == 3:
                vectors_in_batch = vectors_in_batch.squeeze(1)

            similarities = torch.matmul(encoded_phenotypes_matrix, vectors_in_batch.T)

            k_batch = min(self.top_k, similarities.size(1))
            batch_topk_sims, batch_topk_indices = torch.topk(similarities, k_batch, dim=1)

            batch_topk_sims = batch_topk_sims.cpu().tolist()
            batch_topk_indices = batch_topk_indices.cpu().tolist()

            for i in range(total_phenotypes):
                for sim, idx in zip(batch_topk_sims[i], batch_topk_indices[i]):
                    if sim < self.min_required_similarity: continue
                    
                    code = codes_in_batch[idx]
                    name = names_in_batch[idx]
                    
                    if len(heaps[i]) < self.top_k:
                        heapq.heappush(heaps[i], (sim, code, name))
                    else:
                        heapq.heappushpop(heaps[i], (sim, code, name))

        mapped_phenotypes = ["None"] * total_phenotypes
        mapping_logs = [None] * total_phenotypes
        pending_calls = []

        executor_ctx = ThreadPoolExecutor(max_workers=N_POOL_THREADS) if self.api_provider else nullcontext()

        with executor_ctx as executor:
            for i in range(total_phenotypes):
                phenotype_name = phenotypes_with_context[i][0]
                context_sentence = phenotypes_with_context[i][1]

                if not heaps[i]:
                    mapping_logs[i] = {
                        "phenotype": phenotype_name,
                        "context": context_sentence,
                        "candidates_text": "Ninguno (No superaron umbral de similitud vectorial)",
                        "reasoning": "N/A",
                        "final_code": "None"
                    }
                    continue

                sorted_candidates = sorted(heaps[i], key=lambda x: x[0], reverse=True)

                if not self.api_provider:
                    mapped_phenotypes[i] = sorted_candidates[0][1]
                    continue

                candidates_text = ""
                for rank, (sim, code, name) in enumerate(sorted_candidates):
                    candidates_text += f"{rank+1}. Código: {code} | Nombre: {name} | (Score vectorial: {sim:.2f})\n"

                if self.prompt:
                    base_prompt = self.prompt
                else:
                    base_prompt = f"""Eres un experto en codificación clínica HPO.
Contexto clínico original del paciente (oración específica):
"{context_sentence}"

Fenotipo extraído a mapear: "{phenotype_name}"

Top {self.top_k} códigos candidatos pre-seleccionados de la ontología HPO:
{candidates_text}

INSTRUCCIONES OBLIGATORIAS:
1. Evalúa internamente todos los candidatos y selecciona los 2 o 3 más prometedores.
2. Escribe una breve justificación (máximo 4 oraciones) comparando ÚNICAMENTE a esos finalistas frente al contexto clínico para llegar a tu decisión. No menciones los candidatos que son evidentemente incorrectos.
3. Si es evidente que ninguno de los candidatos ofrecidos tiene relación anatómica o semántica con el fenotipo, explica brevemente por qué y tu decisión debe ser "None".
4. Al final de tu análisis, DEBES incluir ÚNICAMENTE un bloque de código JSON con tu respuesta final en este formato exacto:
```json
{{"hpo_code": "código_elegido_o_None"}}
```"""

                future = executor.submit(self._call_llm_rag, base_prompt)
                pending_calls.append((i, future, phenotype_name, context_sentence, candidates_text))

            for i, future, phenotype_name, context_sentence, candidates_text in pending_calls:
                hpo_code, full_reasoning = future.result()
                mapped_phenotypes[i] = hpo_code
                mapping_logs[i] = {
                    "phenotype": phenotype_name,
                    "context": context_sentence,
                    "candidates_text": candidates_text,
                    "reasoning": full_reasoning,
                    "final_code": hpo_code
                }

        self.last_mapping_logs = [log for log in mapping_logs if log is not None]

        return mapped_phenotypes

    def _call_llm_rag(self, prompt: str) -> tuple[str, str]:
        """Ask the model to pick one HPO code out of the retrieved candidates.

        A returned "None" means the model looked and found no code that fits,
        which is a real answer. A provider failure is not that answer, so it is
        raised rather than folded into the same value.
        """
        # The caller builds one self-contained prompt, so it stays the user
        # message and the system slot is left empty. Splitting it here would
        # change what the benchmark measured.
        response_text = self.provider.complete(system_prompt="", user_prompt=prompt)

        try:
            parsed_data = parse_json_object(response_text)
        except LLMInvalidResponseError:
            # The model replied but not with usable JSON. That is a genuine
            # "no code", unlike an unreachable provider.
            logger.warning("RAG reply was not usable JSON; recording no code")
            return "None", response_text

        return parsed_data.get("hpo_code", "None"), response_text

    def _get_codes_batch(self):
        batch_files = glob.glob(os.path.join(self.ontology_file_path, "*.pt"))
        for batch_file in batch_files:
            raw_dict = torch.load(batch_file, map_location=self.pytorch_device)
            yield raw_dict
            del raw_dict
            if self.pytorch_device.type == 'cuda': torch.cuda.empty_cache()

    def _get_encoded_phenotypes_list(self, phenotypes_list: list[str]):
        with torch.no_grad():
            tokenized_symptoms = self.tokenizer(
                phenotypes_list, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(self.pytorch_device)
            phenotypes_model_output = self.model(**tokenized_symptoms)
            phenotypes_cls_embedding = phenotypes_model_output.last_hidden_state[:, 0, :]
            encoded_phenotypes_list = F.normalize(phenotypes_cls_embedding, p=2, dim=1)
        return encoded_phenotypes_list