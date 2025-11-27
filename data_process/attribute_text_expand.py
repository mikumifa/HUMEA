from collections import defaultdict
import pickle
import torch
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer
import re

# --------- 初始化 RoBERTa ----------
device = "cuda:1" if torch.cuda.is_available() else "cpu"
print("Device:", device)

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base").to(device).eval()


def get_embedding_batch(texts, batch_size=32):
    """
    输入一批文本，返回每个文本的 [CLS] embedding。
    """
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
    text_str = re.sub(r"http[s]?://[^\s;]+", extract_last_part, text_str)
    text_str = text_str.replace("\t", " ")
    return text_str


def attribute_id_read(entity_id_dict, entity_triple_path):
    entity_attribute_map: dict[int, list[torch.Tensor]] = defaultdict(list)
    texts = []
    ids = []

    with open(entity_triple_path, "r", encoding="utf-8") as fr:
        for line in fr:
            parts = line.strip("\n").split("\t")

            if parts[0] in entity_id_dict:
                id = entity_id_dict[parts[0]]
                text = text_process(line)
                print(text)
                texts.append(text)
                ids.append(id)

    embeddings = get_embedding_batch(texts, batch_size=32)

    for id, embedding in zip(ids, embeddings):
        entity_attribute_map[id].append(embedding)

    averaged_entity_embedding = {
        entity_id: torch.stack(embeddings).mean(dim=0).tolist()
        for entity_id, embeddings in entity_attribute_map.items()
    }

    return averaged_entity_embedding


def expand(names=["DB15K", "YAGO15K"]):
    ent2id_1 = entity_id_read("data/mmkb-datasets/FB15K_DB15K/ent_ids_1")
    embeddings_1 = attribute_id_read(
        ent2id_1, "data/mmkb-datasets/FB15K_DB15K/training_attrs_1"
    )
    for name in names:
        file_dir = f"data/mmkb-datasets/FB15K_{name}"
        ent2id_2 = entity_id_read(file_dir + "/ent_ids_2")

        embeddings_2 = attribute_id_read(
            ent2id_2, f"data/mmkb-datasets/FB15K_{name}/training_attrs_2"
        )
        embeddings = {**embeddings_1, **embeddings_2}
        assert len(embeddings) == (len(embeddings_1) + len(embeddings_2))
        with open(
            f"data/mmkb-datasets/FB15K_{name}/attribute_feature_dict.pkl",
            "wb",
        ) as f:
            pickle.dump(embeddings, f)


if __name__ == "__main__":
    expand()
