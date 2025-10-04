import sys
import types
import scipy.sparse.csr

# Construct a pseudo-module so that "scipy.sparse._csr" can be found during deserialization.
fake_csr = types.ModuleType("scipy.sparse._csr")
fake_csr.csr_matrix = scipy.sparse.csr.csr_matrix
sys.modules["scipy.sparse._csr"] = fake_csr

import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from scispacy.umls_linking import UmlsEntityLinker
from nltk.corpus import wordnet_ic, wordnet as wn
import nltk
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus.reader.wordnet import Synset
import scipy.sparse  

# =============== Global cache variable ===============
brown_ic = wordnet_ic.ic("ic-brown.dat")

entity_text_cache = {}   # { entity_text(str): [ {CUI,Name...}, ... ] }
cui_synset_cache = {}    # { cui(str): best_synset or None }
resnik_score_cache = {}  # { (cui1, cui2) or (cui2, cui1): float }
text_sim_cache = {}      # { (textA, textB): tf-idf相似度 }

def load_spacy_and_linker():

    print("Loading spaCy model and UMLS linker...")
    nlp = spacy.load("en_core_sci_lg")
    linker = UmlsEntityLinker(resolve_abbreviations=True)
    nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
    return nlp, linker

# ------------------- Retained but unused functions -------------------
def load_features_with_labels(file_path):
    """Load features and labels (if labels are still needed, they can be retained;) If not needed, you can delete this function by yourself."""
    print(f"Loading features and labels from {file_path}...")
    data = np.load(file_path, allow_pickle=True)
    features_with_labels = {}
    for idx, entry in enumerate(data):
        features_with_labels[entry['id']] = {
            'label': entry['label'],
            'feature': entry['feature']
        }
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(data)} features...")
    print(f"Loaded {len(features_with_labels)} entries from {file_path}")
    return features_with_labels

def load_report_content(filepath):
    """Load the report content"""
    try:
        with open(filepath, 'r', encoding="utf-8") as file:
            data = json.load(file)
            print(f"Loaded {len(data)} reports from {filepath}.")
            return data
    except Exception as e:
        print(f"Error loading content from {filepath}: {e}")
        return None

# ------------------- TF-IDF similarity calculation with cache -------------------
def text_cosine_similarity_cached(text1, text2):
    """TF-IDF similarity + caching to avoid duplicate calculation of the same text"""
    if not text1 and not text2:
        return 0.0
    key = (text1, text2) if text1 < text2 else (text2, text1)
    if key in text_sim_cache:
        return text_sim_cache[key]

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform([text1, text2])
    sim = (tfidf * tfidf.T).A[0, 1]

    text_sim_cache[key] = sim
    return sim

# ------------------- Resolution of SciSpaCy entities with cache -------------------
def get_umls_entities_cached(nlp, linker, text):
    """Entity resolution with cache to avoid repeated SciSpaCy Linking on the same text"""
    if text in entity_text_cache:
        return entity_text_cache[text]

    doc = nlp(text)
    umls_list = []
    for entity in doc.ents:
        for umls_ent in entity._.kb_ents:
            cui = umls_ent[0]
            possibility = umls_ent[1]
            umls_entity = linker.kb.cui_to_entity[cui]
            umls_list.append({
                "CUI": cui,
                "Name": umls_entity[1],
                "TUI": umls_entity[3],
                "Definition": umls_entity[4],
                "Aliases": umls_entity[2],
                "Possibility": possibility
            })
    entity_text_cache[text] = umls_list
    return umls_list

# ------------------- Map CUI to the WordNet synonym set -------------------
def map_cui_to_synset(cui, cui_to_name_map, cui_synset_cache):
    if cui in cui_synset_cache:
        return cui_synset_cache[cui]

    info = cui_to_name_map.get(cui)
    if not info:
        cui_synset_cache[cui] = None
        return None

    candidate_terms = [info["preferred_name"]] + info.get("synonyms", [])
    candidate_terms = list(set(candidate_terms))
    cui_definition = info.get("definition", "").strip()
    best_synset = None
    best_score = -1

    for term in candidate_terms:
        synsets = wn.synsets(term, pos=wn.NOUN)
        for syn in synsets:
            syn_definition = syn.definition()
            def_score = 0.0
            if cui_definition and syn_definition:
                def_score = text_cosine_similarity_cached(cui_definition, syn_definition)
            lemma_score = 1.0 if term.lower().replace(" ", "_") in [l.name().lower() for l in syn.lemmas()] else 0.0
            total_score = def_score + lemma_score 
            if total_score > best_score:
                best_score = total_score
                best_synset = syn

    cui_synset_cache[cui] = best_synset
    return best_synset

# ------------------- Calculate Resnik similarity and cache it -------------------
def calculate_cui_resnik_similarity(cui1, cui2, cui_to_name_map, cui_synset_cache):
    key = (cui1, cui2) if cui1 < cui2 else (cui2, cui1)
    if key in resnik_score_cache:
        return resnik_score_cache[key]

    syn1 = map_cui_to_synset(cui1, cui_to_name_map, cui_synset_cache)
    syn2 = map_cui_to_synset(cui2, cui_to_name_map, cui_synset_cache)
    if syn1 is None or syn2 is None:
        resnik_score_cache[key] = 0.0
        return 0.0

    try:
        sim = syn1.res_similarity(syn2, brown_ic)
        if sim is None:
            sim = 0.0
    except Exception as e:
        sim = 0.0

    resnik_score_cache[key] = sim
    return sim

# ------------------- Extract relational information -------------------
def extract_relation_info(relations):
    """Extract entity relationships and their type information"""
    relation_info = []
    for _, relation_list in relations.items():
        for relation in relation_list:
            source_entity = relation["source entity"]
            target_entity = relation["target entity"]
            if isinstance(source_entity, list) and source_entity:
                source_entity = source_entity[0]
            if isinstance(target_entity, list) and target_entity:
                target_entity = target_entity[0]
            relation_info.append({
                "source": source_entity,
                "relation": relation["relation"],
                "target": target_entity
            })
    return relation_info

# ------------------- Calculate the similarity of the relationship -------------------
def calculate_relation_similarity(rel1, rel2, linker, cui_to_name_map, cui_synset_cache, umls_entities_i, umls_entities_j):
    def get_cui_for_entity(entity_text, umls_entities_dict):
        if isinstance(entity_text, list) and entity_text:
            entity_text = entity_text[0]
        if entity_text in umls_entities_dict and umls_entities_dict[entity_text]:
            return umls_entities_dict[entity_text][0]["CUI"]
        return None

    source_cui_1 = get_cui_for_entity(rel1["source"], umls_entities_i)
    target_cui_1 = get_cui_for_entity(rel1["target"], umls_entities_i)
    source_cui_2 = get_cui_for_entity(rel2["source"], umls_entities_j)
    target_cui_2 = get_cui_for_entity(rel2["target"], umls_entities_j)

    if source_cui_1 is None or source_cui_2 is None or target_cui_1 is None or target_cui_2 is None:
        return 0.0

    source_similarity = calculate_cui_resnik_similarity(source_cui_1, source_cui_2, cui_to_name_map, cui_synset_cache)
    target_similarity = calculate_cui_resnik_similarity(target_cui_1, target_cui_2, cui_to_name_map, cui_synset_cache)
    relation_type_similarity = 1.0 if rel1["relation"] == rel2["relation"] else 0.0
    return source_similarity + relation_type_similarity + target_similarity

# ------------------- Cross-report knowledge alignment -------------------
def knowledge_alignment(reports, nlp, linker):
    """
    Cross-report knowledge alignment
    Align entities and relationships using CUI->WordNet mapping and Resnik similarity
    """
    print("Starting cross-report knowledge alignment...")

    cui_to_name_map = {}
    processed_reports = []
    report_ids = list(reports.keys())
    for report_id in report_ids:
        report = reports[report_id]
        entities = report.get("res", {})
        relations = report.get("res_relation", {})

        umls_entities = {}
        for entity_text in entities.keys():

            umls_list = get_umls_entities_cached(nlp, linker, entity_text)
            umls_entities[entity_text] = umls_list
            for cui_info in umls_list:
                cui = cui_info["CUI"]
                if cui not in cui_to_name_map:
                    cui_to_name_map[cui] = {
                        "preferred_name": cui_info["Name"],
                        "definition": cui_info["Definition"] if cui_info["Definition"] else "",
                        "synonyms": cui_info["Aliases"] if cui_info["Aliases"] else []
                    }

        relation_info = extract_relation_info(relations)

        processed_reports.append({
            "report_id": report_id,
            "umls_entities": umls_entities,
            "relation_info": relation_info
        })

    num_reports = len(processed_reports)
    alignment_matrix = np.zeros((num_reports, num_reports))

    for i in range(num_reports):
        for j in range(num_reports):
            if i == j:
                continue

            entities_i = processed_reports[i]["umls_entities"]
            entities_j = processed_reports[j]["umls_entities"]
            relation_info_i = processed_reports[i]["relation_info"]
            relation_info_j = processed_reports[j]["relation_info"]

            entity_pairs = 0
            entity_alignment_count = 0
            for entity1, umls1_list in entities_i.items():
                for entity2, umls2_list in entities_j.items():
                    for cui1 in umls1_list:
                        for cui2 in umls2_list:
                            sim = calculate_cui_resnik_similarity(cui1["CUI"], cui2["CUI"], cui_to_name_map, cui_synset_cache)
                            entity_pairs += 1
                            if sim > 0.5:
                                entity_alignment_count += 1

            relation_pairs = 0
            relation_alignment_count = 0
            for rel1 in relation_info_i:
                for rel2 in relation_info_j:
                    rel_similarity = calculate_relation_similarity(
                        rel1, rel2, linker,
                        cui_to_name_map, cui_synset_cache,
                        entities_i, entities_j
                    )
                    if rel_similarity > 0.5:
                        relation_alignment_count += 1
                    relation_pairs += 1

            entity_score = (entity_alignment_count / entity_pairs) if entity_pairs > 0 else 0
            relation_score = (relation_alignment_count / relation_pairs) if relation_pairs > 0 else 0
            alignment_score = entity_score + relation_score
            alignment_matrix[i, j] = alignment_score


    return alignment_matrix, report_ids

def select_worst_reports_from_alignment(alignment_matrix, top_k=5):
    """Sum the scores of each row directly and take the top k lowest indexes (the least similar)"""
    scores = alignment_matrix.sum(axis=1)
    # Sort in ascending order and take the smallest top_k
    worst_indices = np.argsort(scores)[:top_k]
    return worst_indices

def save_final_matching_result(test_id, best_train_ids, output_file_path):
    """Save the matching result and return the list of the best train ids corresponding to test_id"""
    result = {"testid": test_id, "trainids": best_train_ids}
    try:
        if os.path.exists(output_file_path):
            with open(output_file_path, 'r', encoding="utf-8") as output_file:
                existing_data = json.load(output_file)
        else:
            existing_data = []

        for entry in existing_data:
            if entry['testid'] == test_id:
                return

        existing_data.append(result)
        with open(output_file_path, 'w', encoding="utf-8") as output_file:
            json.dump(existing_data, output_file, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving result: {e}")

# ------------------- Batch processing helper function -------------------
def get_batches(data_list, batch_size):

    for i in range(0, len(data_list), batch_size):
        yield data_list[i:i + batch_size]

def clear_caches():

    entity_text_cache.clear()
    cui_synset_cache.clear()
    resnik_score_cache.clear()
    text_sim_cache.clear()

# ------------------- Main -------------------
def main():
    nlp, linker = load_spacy_and_linker()


    top10_file = "data/iu_xray/iu_kl_top10.json"
    with open(top10_file, 'r', encoding="utf-8") as f:
        top10_list = json.load(f)  

 
    report_file = "data/iu_xray/iu_entities_relations_chexpert_plus_post.json"
    report_data = load_report_content(report_file)
    if report_data is None:
        print(f"Failed to load report data from {report_file}. Exiting...")
        return

    output_file_path = "data/iu_xray/iu_kg_top5.json"

    batch_size = 10
    def get_batches(data_list, batch_size):
        for i in range(0, len(data_list), batch_size):
            yield data_list[i:i + batch_size]

    for batch in get_batches(top10_list, batch_size):
        for entry in batch:
            test_id = entry["testid"]
            candidate_train_ids = entry.get("trainid", [])
            print(f"\nProcessing test report: {test_id} with {len(candidate_train_ids)} candidates")

            sub_reports = {
                tid: report_data[tid]
                for tid in candidate_train_ids
                if tid in report_data
            }
            if not sub_reports:
                print(f"No valid sub-reports for {test_id}, skipping.")
                continue


            alignment_matrix, aligned_ids = knowledge_alignment(sub_reports, nlp, linker)
            worst_indices = select_worst_reports_from_alignment(alignment_matrix, top_k=5)
            best_train_ids = [aligned_ids[i] for i in worst_indices]

            save_final_matching_result(test_id, best_train_ids, output_file_path)

        clear_caches()

    print("All semantic matches completed.")

if __name__ == "__main__":
    main()