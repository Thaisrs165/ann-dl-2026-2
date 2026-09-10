# ANN-DL — Portfólio individual

Portfólio pessoal das entregas individuais da disciplina de Redes Neurais Artificiais &
Deep Learning, semestre **2026.2** — [enunciados](https://insper.github.io/ann-dl/).

Site gerado com [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) e publicado
no GitHub Pages.

## Estrutura

```
docs/
  index.md                     # capa: identificação e status das entregas
  exercises/
    data/{index.md,code/,figures/}
    perceptron/{index.md,code/,figures/}
    mlp/{index.md,code/,figures/}
    vae/{index.md,code/,figures/}
```

Cada exercício tem a própria pasta: `index.md` para o relatório, `code/` para os scripts e
`figures/` para as imagens geradas.

## Setup

```shell
python3 -m venv env
source ./env/bin/activate          # Windows: .\env\Scripts\activate
python3 -m pip install -r requirements.txt --upgrade
```

## Rodando localmente

```shell
mkdocs serve -o
```

Antes de dar push, valide localmente:

```shell
mkdocs build --strict
```

## Publicação

O workflow em [.github/workflows/main.yaml](.github/workflows/main.yaml) roda
`mkdocs gh-deploy --force` a cada push na `main`, publicando a branch `gh-pages` no GitHub
Pages.

Para publicar manualmente, sem passar pelo CI:

```shell
mkdocs gh-deploy
```

## Prazo

O prazo de uma entrega é o **timestamp do último commit que toca a pasta daquela entrega**
— não a hora do formulário nem a da publicação. Commite progressivamente.
