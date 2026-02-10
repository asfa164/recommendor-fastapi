# Objective Recommender FastAPI (Vercel, Cognito + Bedrock)

FastAPI backend that wraps AWS Bedrock using AWS Cognito (User Pool + Identity Pool)
to recommend clearer defining objectives.

## Endpoints

- `GET /` – Health/status
- `POST /recommend` – Takes a CompositeObjective JSON and returns recommended defining objectives.

## Auth

Send `X-API-Key` header. Configure the API key as `API_KEY` environment variable in Vercel.

## Required environment variables

- API_KEY

- COGNITO_REGION
- COGNITO_USER_POOL_ID
- COGNITO_CLIENT_ID
- COGNITO_CLIENT_SECRET (optional if your app client has no secret)
- COGNITO_USERNAME
- COGNITO_PASSWORD
- COGNITO_IDENTITY_POOL_ID

- BEDROCK_MODEL_ID
- BEDROCK_REGION (optional, falls back to COGNITO_REGION or eu-west-1)

## Deploy

1. Push this folder to a Git repo.
2. Create a new Vercel project from the repo.
3. Ensure Python is detected, no extra build command needed.
4. Set environment variables in Vercel as above.

Vercel will route all requests to `api/main.py`.
