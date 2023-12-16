# GAN-GI
Using a Generative Adversarial Network to Generate Images for Academic Works based in job of Atin Sakkeer Hussain, 2019

# Requirements
- Docker
- Docker-Compose

> [!NOTE]
> <sup> This work Using a Generative Adversarial Network to Generate Images for Academic Final Work is based in job of Madhu Sanjeevi's, 2019 [See more](https://github.com/crypto-code/GAN)</sup>

# Build Start-GAN 

## 1. Select the dataset

Place the images that served as the basis for the GAN within the data directory

## 2. Upload local image to test the model
```
docker build --no-cache -f Dockerfile -t gan-gi:1.0 .
```
> [!NOTE]
> <sup> Depending on your machine and internet connection, this step can take from 30 to 120 minutes. Be patient.</sup>

## 3. Upload the local container to test the model
```
docker network create gan-network
cd start-GAN/
docker-compose up -d
```
### 3.1 Customize training

The standard results are not perfect because perfecting the generator requires a lot of computational power. And of course the results are not precise, as we have created a small network with few iterations (20,000), but this already provides a good idea of how to train GANs. To improve the result, you can change the number of iterations (epochs) that the algorithm will train, this is an exclusive contribution to this project. Just go to the ./start-GAN directory and modify the number of epochs in the .env file.
Good study!