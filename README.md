# RecomFlow 🎬

Plataforma de recomendación de películas sin registro, usando IA y tecnologías modernas.

## 🚀 Estructura del Proyecto

```plaintext
recomflow/
├── frontend/          # Aplicación React
├── backend/           # Microservicios (API Gateway, Recomendación, Datos)
├── scripts/           # ETL y entrenamiento del modelo
├── infra/             # Configuración de Docker/Kubernetes
└── README.md
```

## 📋 Prerrequisitos
- Docker y Docker Compose
- Python 3.8+
- Node.js 16+

## 🛠️ Configuración Inicial

1. Clonar repositorio

    ```bash
    git clone https://github.com/bautimercado/recomflow.git
    cd recomflow
    ```

2. Levanta los servicios con Docker Compose:
    ```bash
    cd infra
    docker-compose up --build
    ```

