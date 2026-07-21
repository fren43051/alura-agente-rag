#!/bin/bash
# ============================================
# AluraAgente RAG — Deploy en Oracle Cloud
# ============================================

set -e

echo "🚀 Iniciando deploy de AluraAgente RAG en OCI..."

# Variables
REGION="us-ashburn-1"
IMAGE_NAME="alura-agente-rag"
CONTAINER_PORT=7860

# 1. Build de la imagen Docker
echo "📦 Construyendo imagen Docker..."
docker build -t $IMAGE_NAME .

# 2. Tag para OCI Container Registry
echo "🏷️  Etiquetando imagen para OCI..."
docker tag $IMAGE_NAME $REGION.ocir.io/$OCI_TENANCY_NAMESPACE/$IMAGE_NAME:latest

# 3. Push a OCI Container Registry
echo "⬆️  Subiendo imagen a OCI Container Registry..."
docker push $REGION.ocir.io/$OCI_TENANCY_NAMESPACE/$IMAGE_NAME:latest

# 4. Crear instancia en OCI (Container Instance)
echo "☁️  Creando Container Instance en OCI..."
oci container-instances container-instance create \
  --compartment-id $OCI_COMPARTMENT_ID \
  --availability-domain $OCI_AD \
  --display-name alura-agente-rag \
  --containers "[{\"imageUrl\":\"$REGION.ocir.io/$OCI_TENANCY_NAMESPACE/$IMAGE_NAME:latest\",\"displayName\":\"alura-agente\"}]" \
  --shape CI.Standard.E4.Flex \
  --shape-config '{"ocpus": 1, "memoryInGBs": 4}'

echo "✅ Deploy completado!"
echo "🌐 Accede a tu agente en: http://[TU-IP-OCI]:$CONTAINER_PORT"
