# GAN-GI
Using a Generative Adversarial Network to Generate Images for Academic Works based in job of Atin Sakkeer Hussain, 2019

# Requirements
- Docker
- Docker-Compose

> [!NOTE]
> <sup> This work Using a Generative Adversarial Network to Generate Images for Academic Final Work is based in job of Atin Sakkeer Hussain, 2019[See more](https://github.com/crypto-code/GAN)</sup>

# Build Start-Node
## 1. Subir imagem local para primeiro nó da rede

Coloque as imagens que serviram de base para o GAN dentro do diretorio data


## 2. Subir imagem local para primeiro nó da rede
```
docker build --no-cache -f Dockerfile -t gan-gi:1.0 .
```
## 3. Subir o container para o primeiro nó
```
docker network create gan-network
cd start-GAN/
docker-compose up -d gan-gi
```