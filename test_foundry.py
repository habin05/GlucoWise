
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

# Foundry의 "코드에서 계속"에서 복사한 실제 프로젝트 엔드포인트
endpoint = "https://whb050704-4696-resource.services.ai.azure.com/api/projects/whb050704-4696"

agent_name = "GlucoWise-Agent"

user_message = """
This is simulated CGM data.
Meal: White rice 150g
Pre-meal glucose: 100 mg/dL
Peak glucose: 172 mg/dL
How much did blood glucose increase?
"""

with DefaultAzureCredential() as credential:
    with AIProjectClient(
        endpoint=endpoint,
        credential=credential,
        allow_preview=True,
    ) as project:
        openai_client = project.get_openai_client(
            agent_name=agent_name
        )

        response = openai_client.responses.create(
            input=user_message
        )

        print("GlucoWise-Agent response:")
        print(response.output_text)
