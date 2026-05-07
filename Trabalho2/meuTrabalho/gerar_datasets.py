import os
import shutil
import json
import xml.etree.ElementTree as ET
import torchvision.datasets as tv_datasets
import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
DIRETORIO_DATASET = os.path.join(BASE, "dataset")
DIRETORIO_QUERIES = os.path.join(BASE, "queries")
ARQUIVO_ANNOTATIONS = os.path.join(BASE, "annotations.json")

os.makedirs(DIRETORIO_DATASET, exist_ok=True)
os.makedirs(DIRETORIO_QUERIES, exist_ok=True)

CLASSES_ALVO = ["cat", "dog", "car", "person", "bird"]
NUM_DOCUMENTOS = 25
NUM_QUERIES = 5


def baixar_voc():
    print(f"[DOWNLOAD] Baixando Pascal VOC 2007 via torchvision...")
    print("  (~500MB, pode demorar alguns minutos...)")
    dataset = tv_datasets.VOCDetection(
        root=BASE,
        year="2007",
        image_set="trainval",
        download=True,
    )
    return dataset


def parse_dataset(dataset):
    resultados = []
    for idx in range(len(dataset)):
        img, target = dataset[idx]
        annos = target["annotation"]
        objects = []
        for obj in annos["object"]:
            classe = obj["name"]
            if classe not in CLASSES_ALVO:
                continue
            bndbox = obj["bndbox"]
            xmin = int(float(bndbox["xmin"]))
            ymin = int(float(bndbox["ymin"]))
            xmax = int(float(bndbox["xmax"]))
            ymax = int(float(bndbox["ymax"]))
            objects.append({"classe": classe, "bbox": (xmin, ymin, xmax, ymax)})
        if objects:
            filename = annos["filename"]
            resultados.append({
                "filename": filename,
                "objects": objects,
                "index": idx,
            })
    return resultados


def selecionar_imagens(imgs_com_objetos):
    por_classe = {c: [] for c in CLASSES_ALVO}
    for img in imgs_com_objetos:
        for obj in img["objects"]:
            por_classe[obj["classe"]].append(img)

    selecionados_doc = []
    selecionados_query = []

    docs_por_classe = NUM_DOCUMENTOS // len(CLASSES_ALVO)
    queries_por_classe = NUM_QUERIES // len(CLASSES_ALVO)
    resto_docs = NUM_DOCUMENTOS - docs_por_classe * len(CLASSES_ALVO)
    resto_queries = NUM_QUERIES - queries_por_classe * len(CLASSES_ALVO)

    usados = set()
    for i, classe in enumerate(CLASSES_ALVO):
        disponiveis = [img for img in por_classe[classe] if img["filename"] not in usados]
        n_docs = docs_por_classe + (1 if i < resto_docs else 0)
        n_queries = queries_por_classe + (1 if i < resto_queries else 0)

        docs = disponiveis[:n_docs]
        usados.update(d["filename"] for d in docs)

        disponiveis = [img for img in por_classe[classe] if img["filename"] not in usados]
        queries = disponiveis[:n_queries]
        usados.update(q["filename"] for q in queries)

        for img in docs:
            selecionados_doc.append(img)
        for img in queries:
            selecionados_query.append(img)

    return selecionados_doc, selecionados_query


def copiar_imagens(dataset, selecionados, diretorio_destino, sufixo, annotations):
    for idx, img in enumerate(selecionados):
        classes_presentes = sorted(set(obj["classe"] for obj in img["objects"]))
        label = "+".join(classes_presentes)
        nome_final = f"{sufixo}_{label}_{idx:03d}.jpg"
        pil_img = dataset[img["index"]][0]
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        dst = os.path.join(diretorio_destino, nome_final)
        cv2.imwrite(dst, cv_img)

        h, w = cv_img.shape[:2]
        bboxes_normalizadas = []
        for obj in img["objects"]:
            xmin, ymin, xmax, ymax = obj["bbox"]
            bboxes_normalizadas.append({
                "classe": obj["classe"],
                "bbox": (xmin / w, ymin / h, xmax / w, ymax / h)
            })
        annotations[nome_final] = bboxes_normalizadas


if __name__ == "__main__":
    voc_dir = os.path.join(BASE, "VOCdevkit")
    if os.path.isdir(voc_dir) and os.path.isdir(os.path.join(voc_dir, "VOC2007", "Annotations")):
        print("[OK] VOCdevkit ja existe.")
        dataset = tv_datasets.VOCDetection(
            root=BASE,
            year="2007",
            image_set="trainval",
            download=False,
        )
    else:
        dataset = baixar_voc()

    print(f"[PARSE] Lendo annotations das classes: {CLASSES_ALVO}")
    imgs_com_objetos = parse_dataset(dataset)
    print(f"  Encontradas {len(imgs_com_objetos)} imagens com as classes alvo.")

    docs, queries = selecionar_imagens(imgs_com_objetos)
    print(f"  Selecionados {len(docs)} para dataset, {len(queries)} para queries.")

    annotations = {}
    copiar_imagens(dataset, docs, DIRETORIO_DATASET, "doc", annotations)
    copiar_imagens(dataset, queries, DIRETORIO_QUERIES, "query", annotations)

    with open(ARQUIVO_ANNOTATIONS, "w") as f:
        json.dump(annotations, f, indent=2)
    print(f"  Annotations salvas em {ARQUIVO_ANNOTATIONS}")

    print("\n[OK] Dataset preparado:")
    print(f"  {len(os.listdir(DIRETORIO_DATASET))} imagens em dataset/")
    print(f"  {len(os.listdir(DIRETORIO_QUERIES))} imagens em queries/")
