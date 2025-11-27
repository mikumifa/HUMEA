from collections import defaultdict
import pickle
import torch
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer
import re

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base").to(device).eval()


def get_embedding_batch(texts, batch_size=32):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)
        all_embeddings.extend(cls_embeddings.cpu())
    return all_embeddings  # List[Tensor]


def entity_id_read(file_path):
    ent2id_dict = {}
    with open(file_path, "r", encoding="utf-8") as fr:
        for line in fr:
            params = line.strip("\n").split("\t")
            ent2id_dict[params[1]] = int(params[0])
    return ent2id_dict


def text_process(originate_text):
    def extract_last_part(match):
        url = match.group(0)
        return url.rstrip("/").split("/")[-1]

    text_str = originate_text.replace("<", "").replace(">", "")
    text_str = re.sub(r"/[^\s;]+", extract_last_part, text_str)
    text_str = text_str.replace("\t", " ")
    return text_str


def relation_id_read(entity_id_dict, entity_triple_path):
    entity_embedding_map: dict[int, list[torch.Tensor]] = defaultdict(list)

    triple_pairs = []
    entity_pairs = []

    with open(entity_triple_path, "r", encoding="utf-8") as fr:
        for line in fr:
            parts = line.strip().split(" ")
            if len(parts) < 3:
                continue
            e1, r, e2 = parts[0:3]

            if e1 in entity_id_dict and e2 in entity_id_dict:
                id1 = entity_id_dict[e1]
                id2 = entity_id_dict[e2]
                text = text_process(f"{r}")
                print(text)
                triple_pairs.append(text)
                entity_pairs.append((id1, id2))

    print(f"Encoding {len(triple_pairs)} triples...")
    embeddings = get_embedding_batch(triple_pairs, batch_size=32)

    for (id1, id2), embedding in zip(entity_pairs, embeddings):
        entity_embedding_map[id1].append(embedding)
        entity_embedding_map[id2].append(embedding)

    # 平均并转为 list
    averaged_entity_embedding = {
        entity_id: torch.stack(embeddings).mean(dim=0).tolist()
        for entity_id, embeddings in entity_embedding_map.items()
    }

    return averaged_entity_embedding


def expand(names=["DB15K", "YAGO15K"]):
    ent2id_1 = entity_id_read("data/mmkb-datasets/FB15K_DB15K/ent_ids_1")
    embeddings_1 = relation_id_read(
        ent2id_1, r"data/mmkb-master/FB15K/FB15K_EntityTriples.txt"
    )
    for name in names:
        file_dir = f"data/mmkb-datasets/FB15K_{name}"
        ent2id_2 = entity_id_read(file_dir + "/ent_ids_2")

        embeddings_2 = relation_id_read(
            ent2id_2, f"data/mmkb-master/{name}/{name}_EntityTriples.txt"
        )
        embeddings = {**embeddings_1, **embeddings_2}
        assert len(embeddings) == (len(embeddings_1) + len(embeddings_2))
        with open(
            f"data/mmkb-datasets/FB15K_{name}/triples_feature_dict.pkl",
            "wb",
        ) as f:
            pickle.dump(embeddings, f)


if __name__ == "__main__":
    expand()
