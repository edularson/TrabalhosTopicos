import os
import json
import pickle
import numpy as np
import cv2
from skimage.feature import hog

DIRETORIO_BASE     = os.path.dirname(os.path.abspath(__file__))
DIRETORIO_DATASET  = os.path.join(DIRETORIO_BASE, "dataset")
DIRETORIO_QUERIES  = os.path.join(DIRETORIO_BASE, "queries")
DIRETORIO_SAIDA    = os.path.join(DIRETORIO_BASE, "output")
ARQUIVO_INDICE     = os.path.join(DIRETORIO_BASE, "indice.pkl")
ARQUIVO_ANNOTATIONS = os.path.join(DIRETORIO_BASE, "annotations.json")
TAMANHO_IMAGEM     = (256, 256)

os.makedirs(DIRETORIO_SAIDA, exist_ok=True)


def bbox_para_pixels(bbox_norm, w, h):
    xmin, ymin, xmax, ymax = bbox_norm
    return (int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h))


def gerar_propostas_regioes(imagem, annotations_nome=None):
    H, W = imagem.shape[:2]
    propostas = []

    if annotations_nome:
        with open(ARQUIVO_ANNOTATIONS, "r") as f:
            annotations = json.load(f)
        if annotations_nome in annotations:
            for obj in annotations[annotations_nome]:
                bbox_px = bbox_para_pixels(tuple(obj["bbox"]), W, H)
                padding = 20
                x1 = max(0, bbox_px[0] - padding)
                y1 = max(0, bbox_px[1] - padding)
                x2 = min(W, bbox_px[2] + padding)
                y2 = min(H, bbox_px[3] + padding)
                propostas.append((x1, y1, x2, y2))

    if not propostas:
        escalas = [0.5, 0.75, 1.0]
        for escala in escalas:
            jan_h = int(H * escala)
            jan_w = int(W * escala)
            passo_h = max(1, jan_h // 4)
            passo_w = max(1, jan_w // 4)
            for y in range(0, H - jan_h + 1, passo_h):
                for x in range(0, W - jan_w + 1, passo_w):
                    propostas.append((x, y, x + jan_w, y + jan_h))
        cx, cy = W // 4, H // 4
        propostas.append((cx, cy, cx + W // 2, cy + H // 2))

    propostas = list(set(propostas))
    return propostas


def extrair_histograma_cor(regiao, bins=16):
    hsv = cv2.cvtColor(regiao, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256]).flatten()
    hist = np.concatenate([hist_h, hist_s, hist_v])
    hist = hist / (hist.sum() + 1e-7)
    return hist


def extrair_hog(regiao, tamanho=(64, 64)):
    cinza = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
    cinza = cv2.resize(cinza, tamanho)
    caracteristicas = hog(
        cinza,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        transform_sqrt=True,
        feature_vector=True
    )
    return caracteristicas


def extrair_descritor(imagem, regiao):
    x1, y1, x2, y2 = regiao
    h, w = imagem.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 - x1 < 8 or y2 - y1 < 8:
        return None

    recorte = imagem[y1:y2, x1:x2]
    recorte = cv2.resize(recorte, (64, 64))

    caracteristica_cor = extrair_histograma_cor(recorte)
    caracteristica_hog = extrair_hog(recorte)
    descritor = np.concatenate([caracteristica_cor, caracteristica_hog])
    return descritor


def construir_indice(diretorio_dataset):
    print("[INDICE] Iniciando indexacao...")
    indice = {}

    with open(ARQUIVO_ANNOTATIONS, "r") as f:
        annotations = json.load(f)

    arquivos = sorted([f for f in os.listdir(diretorio_dataset)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    for nome_arquivo in arquivos:
        caminho = os.path.join(diretorio_dataset, nome_arquivo)
        imagem  = cv2.imread(caminho)
        if imagem is None:
            continue
        imagem = cv2.resize(imagem, TAMANHO_IMAGEM)

        propostas = gerar_propostas_regioes(imagem, annotations_nome=nome_arquivo)
        regioes_doc = []

        for regiao in propostas:
            desc = extrair_descritor(imagem, regiao)
            if desc is not None:
                regioes_doc.append({
                    "regiao":    regiao,
                    "descritor": desc
                })

        indice[nome_arquivo] = regioes_doc
        print(f"  {nome_arquivo}: {len(regioes_doc)} regioes indexadas")

    with open(ARQUIVO_INDICE, "wb") as f:
        pickle.dump(indice, f)

    print(f"[INDICE] Concluido. {len(indice)} documentos indexados.")
    return indice


def carregar_indice():
    with open(ARQUIVO_INDICE, "rb") as f:
        return pickle.load(f)


def calcular_iou(caixaA, caixaB):
    xA = max(caixaA[0], caixaB[0])
    yA = max(caixaA[1], caixaB[1])
    xB = min(caixaA[2], caixaB[2])
    yB = min(caixaA[3], caixaB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    intersecao = inter_w * inter_h

    areaA = (caixaA[2] - caixaA[0]) * (caixaA[3] - caixaA[1])
    areaB = (caixaB[2] - caixaB[0]) * (caixaB[3] - caixaB[1])
    uniao = areaA + areaB - intersecao + 1e-7

    return intersecao / uniao


def similaridade_cosseno(a, b):
    a_norm = a / (np.linalg.norm(a) + 1e-7)
    b_norm = b / (np.linalg.norm(b) + 1e-7)
    return float(np.dot(a_norm, b_norm))


def recuperar(caminho_query, indice, top_k=10, alfa=0.7, beta=0.3):
    nome_query = os.path.basename(caminho_query)
    imagem   = cv2.imread(caminho_query)
    imagem   = cv2.resize(imagem, TAMANHO_IMAGEM)
    propostas = gerar_propostas_regioes(imagem, annotations_nome=nome_query)

    regioes_query = []
    for regiao in propostas:
        desc = extrair_descritor(imagem, regiao)
        if desc is not None:
            regioes_query.append({"regiao": regiao, "descritor": desc})

    if not regioes_query:
        return []

    pontuacoes = {}

    for nome_arquivo, regioes_doc in indice.items():
        melhor_pontuacao = 0.0
        melhor_resultado = {}

        for reg_q in regioes_query:
            for reg_d in regioes_doc:
                sim_visual = similaridade_cosseno(reg_q["descritor"], reg_d["descritor"])
                iou        = calcular_iou(reg_q["regiao"], reg_d["regiao"])
                pontuacao  = alfa * sim_visual + beta * iou

                if pontuacao > melhor_pontuacao:
                    melhor_pontuacao = pontuacao
                    melhor_resultado = {
                        "regiao_query": reg_q["regiao"],
                        "regiao_doc":   reg_d["regiao"],
                        "sim_visual":   round(sim_visual, 4),
                        "iou":          round(iou, 4),
                        "pontuacao":    round(pontuacao, 4)
                    }

        pontuacoes[nome_arquivo] = melhor_resultado if melhor_pontuacao > 0 else {
            "pontuacao": 0.0, "sim_visual": 0.0, "iou": 0.0,
            "regiao_query": (0,0,0,0), "regiao_doc": (0,0,0,0)
        }

    ranqueados = sorted(pontuacoes.items(), key=lambda x: x[1]["pontuacao"], reverse=True)
    return ranqueados[:top_k]


def obter_categoria(nome_arquivo):
    partes = nome_arquivo.split("_")
    if len(partes) >= 2:
        return partes[1]
    return nome_arquivo.rsplit("_", 1)[0]


def precisao_em_k(resultados, nome_query, k=5):
    categoria_query = obter_categoria(nome_query)
    relevantes = 0
    for nome, _ in resultados[:k]:
        cat_doc = obter_categoria(nome)
        classes_query = categoria_query.split("+")
        if any(classe in cat_doc for classe in classes_query):
            relevantes += 1
    return relevantes / k


def visualizar_resultados(caminho_query, resultados, diretorio_dataset, caminho_saida, top_k=5):
    imagem_query = cv2.imread(caminho_query)
    imagem_query = cv2.resize(imagem_query, TAMANHO_IMAGEM)

    colunas = [imagem_query.copy()]
    cv2.putText(colunas[0], "QUERY", (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    cat_query = obter_categoria(os.path.basename(caminho_query))

    for rank, (nome_arquivo, info) in enumerate(resultados[:top_k]):
        imagem_doc = cv2.imread(os.path.join(diretorio_dataset, nome_arquivo))
        imagem_doc = cv2.resize(imagem_doc, TAMANHO_IMAGEM)

        x1, y1, x2, y2 = info["regiao_doc"]
        cat_doc = obter_categoria(nome_arquivo)
        classes_query = cat_query.split("+")
        cor = (0, 200, 0) if any(classe in cat_doc for classe in classes_query) else (0, 0, 200)
        cv2.rectangle(imagem_doc, (x1, y1), (x2, y2), cor, 2)

        display_name = nome_arquivo[:18]
        cv2.putText(imagem_doc, f"#{rank+1} {display_name}",  (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
        cv2.putText(imagem_doc, f"P:{info['pontuacao']:.2f} IoU:{info['iou']:.2f}", (2, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0,0,0), 1)
        colunas.append(imagem_doc)

    combinada = np.hstack(colunas)
    cv2.imwrite(caminho_saida, combinada)
    return combinada


if __name__ == "__main__":
    if not os.path.exists(ARQUIVO_INDICE):
        indice = construir_indice(DIRETORIO_DATASET)
    else:
        print("[INDICE] Carregando indice existente...")
        indice = carregar_indice()

    queries = sorted([f for f in os.listdir(DIRETORIO_QUERIES)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    todos_resultados = {}
    precisoes        = []

    print("\n[RECUPERACAO] Executando queries...\n")
    for nome_query in queries:
        caminho_query = os.path.join(DIRETORIO_QUERIES, nome_query)
        resultados    = recuperar(caminho_query, indice, top_k=10)

        p5 = precisao_em_k(resultados, nome_query, k=5)
        precisoes.append(p5)
        todos_resultados[nome_query] = resultados

        print(f"Query: {nome_query}  |  Precisao@5 = {p5:.2f}")
        for rank, (nome_arquivo, info) in enumerate(resultados[:5]):
            cat_doc = obter_categoria(nome_arquivo)
            classes_query = obter_categoria(nome_query).split("+")
            marcador = "OK" if any(classe in cat_doc for classe in classes_query) else "X"
            print(f"  {rank+1}. [{marcador}] {nome_arquivo:30s}  pontuacao={info['pontuacao']:.4f}  "
                  f"visual={info['sim_visual']:.4f}  iou={info['iou']:.4f}")
        print()

        caminho_saida = os.path.join(DIRETORIO_SAIDA, f"resultado_{nome_query}")
        visualizar_resultados(caminho_query, resultados, DIRETORIO_DATASET, caminho_saida, top_k=5)

    media_p5 = np.mean(precisoes)
    print(f"\n[AVALIACAO] Media Precisao@5 = {media_p5:.4f}")

    resultados_json = {}
    for nq, res in todos_resultados.items():
        resultados_json[nq] = [
            {"rank": i+1, "documento": nome, **info}
            for i, (nome, info) in enumerate(res)
        ]
    with open(os.path.join(DIRETORIO_SAIDA, "resultados.json"), "w") as f:
        json.dump(resultados_json, f, indent=2)

    print("\n[CONCLUIDO] Resultados salvos em", DIRETORIO_SAIDA)
