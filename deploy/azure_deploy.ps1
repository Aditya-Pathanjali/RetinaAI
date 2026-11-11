# Deploy RetinaAI to Azure Container Apps (Consumption plan, scale-to-zero).
# Prereqs: Azure CLI installed, `az login` done, image pushed to GHCR and set to PUBLIC.
# Usage:   .\deploy\azure_deploy.ps1            (first deploy)
#          .\deploy\azure_deploy.ps1 -Update    (roll out a newer :latest image)
param(
    [string]$Location = "uksouth",
    [string]$Image    = "ghcr.io/aditya-pathanjali/retinaai:latest",
    [switch]$Update
)
$ErrorActionPreference = "Stop"
$rg = "rg-retinaai"; $env_ = "retinaai-env"; $app = "retinaai"

if ($Update) {
    az containerapp update -n $app -g $rg --image $Image --revision-suffix ("r" + (Get-Date -Format "MMddHHmm")) -o none
}
else {
    az extension add --name containerapp --upgrade -y
    az provider register --namespace Microsoft.App --wait
    az provider register --namespace Microsoft.OperationalInsights --wait

    az group create -n $rg -l $Location -o none
    # --logs-destination none: avoids a Log Analytics workspace (no ingestion cost)
    az containerapp env create -n $env_ -g $rg -l $Location --logs-destination none -o none

    # 1 vCPU / 2 GiB, scale to zero when idle (no cost while idle), max 1 replica (cost cap)
    az containerapp create -n $app -g $rg --environment $env_ `
        --image $Image `
        --target-port 7860 --ingress external `
        --cpu 1.0 --memory 2.0Gi `
        --min-replicas 0 --max-replicas 1 `
        --env-vars HF_HUB_OFFLINE=1 -o none
}

$fqdn = az containerapp show -n $app -g $rg --query properties.configuration.ingress.fqdn -o tsv
Write-Host "Live URL: https://$fqdn"
Write-Host "Health:   https://$fqdn/api/v1/health   (first request after idle takes ~30s)"
