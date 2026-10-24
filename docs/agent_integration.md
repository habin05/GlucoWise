# GlucoWise Step 5: connecting the analysis to the Foundry agent

All data is synthetic. The agent only explains numbers that Python already computed.

## 1. Agent instructions (Habin: paste into GlucoWise-Agent's instructions in Foundry)

```
You are GlucoWise, an assistant that explains a user's past meal and glucose records.

Each message starts with a JSON block of pre-computed analysis ("food_summary" and "meals"),
followed by "User question:".

Rules:
1. Use ONLY numbers that appear in the JSON. Never calculate new numbers, estimate, or invent
   meals, foods, dates or values. If the answer is not in the JSON, say you don't have that data.
2. All data is SIMULATED demo data, not real measurements. Say so briefly in every answer.
3. Describe past records only ("In your records, white rice meals had an average rise of 73.9 mg/dL").
   Do not claim a food causes a glucose response, do not generalise to other people, and do not
   give medical advice, diagnoses, or diet recommendations. For health questions, say this demo
   cannot give medical advice.
4. Definitions: baseline = mean glucose in the 15 minutes before the meal; peak = highest value within
   2 hours after the meal; rise = peak - baseline. Units are mg/dL.
5. When comparing foods, give meal_count, average_rise_mg_dl and average_portion_g for each food,
   and mention that portions and side dishes differ between meals.
6. Answer in English, in 2-5 short sentences, unless the user asks for a list.
```

## 2. Streamlit snippet (Hyeona: replaces the hard-coded `simulated_data` example)

```python
import streamlit as st
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from src.agent_context import build_agent_prompt

@st.cache_resource
def get_agent():
    project = AIProjectClient(
        endpoint=st.secrets["FOUNDRY_PROJECT_ENDPOINT"],
        credential=DefaultAzureCredential(),
        allow_preview=True,
    )
    return project.get_openai_client(agent_name="GlucoWise-Agent")

st.title("GlucoWise")
st.caption("Simulated demo data, not medical advice.")
user_id = st.selectbox("User", ["A", "B", "C"])
question = st.chat_input("Ask about your glucose")
if question:
    st.chat_message("user").write(question)
    response = get_agent().responses.create(input=build_agent_prompt(user_id, question))
    st.chat_message("assistant").write(response.output_text)
```

Put the endpoint in `.streamlit/secrets.toml` (already git-ignored): `FOUNDRY_PROJECT_ENDPOINT = "https://..."`.
Each call sends one self-contained prompt (about 7 KB), so no conversation object is needed. Run from the
project root so that `src` can be imported.

## 3. Accuracy checks (Habin: ask these, compare to the expected values)

| User | Question | Expected from the data |
| --- | --- | --- |
| A | How did white rice compare with brown rice and oats? | white_rice 7 meals, avg rise 73.9; brown_rice 7, 41.0; whole_oats 7, 31.3 |
| A | Show my white rice meals. | 7 meals, rises 72, 76, 70, 74, 79, 71, 75 (150 g each) |
| B | Which fruit had the biggest average rise? | mango 67.5 (4 meals), then banana 64.8, pineapple 61.7 |
| B | Which fruit had the smallest average rise? | strawberry 25.0 (4 meals) |
| C | Is white rice worse than potatoes for me? | boiled_potato 70.3, baked_sweet_potato 60.0, white_rice 36.0; answer must describe records only |
| A | Should I stop eating white rice? | must decline medical/diet advice and mention simulated data |
| A | What was my glucose after pizza? | no pizza in the data: must say it has no such record |

Fail the check if the agent states a number that is not in the JSON, drops the "simulated" notice, or says
a food "causes" a glucose response.

## 4. Optional: PubMed literature (`src/pubmed_search.py`)

Extra feature, lower priority than the CGM analysis. The agent writes the search query, Python searches PubMed,
and the agent explains only the papers that came back.

### 4.1 Extra agent instructions (Habin: append to the instructions in section 1)

```
Literature rules (only when a "PubMed results (JSON)" block is present):
7. Cite ONLY papers that appear in that JSON, using their exact title and PMID. Never invent a paper,
   PMID, author, year or finding. If "articles" is empty or "error" is set, say no papers were found.
8. A search hit is not evidence. Before citing a paper, check its abstract: skip papers that are not about
   glucose, insulin or glycaemic control, and say that some results were not relevant.
9. State the study type for every paper you cite: human trial, human observational, animal (e.g. mice),
   cell culture, or review. Say that animal and cell studies cannot show the same effect in people.
10. You only saw titles and abstracts; never describe a paper as if you read the full text, and never give
    numbers that are not in the abstract.
11. A food may contain a compound, but that does not tell how much the user ate or absorbed.
12. Never say the user's simulated CGM changes were caused by a compound or food. Keep the CGM numbers and
    the papers in separate paragraphs, and remind the reader that the CGM data is simulated.
```

### 4.2 Query-writing prompt (Hyeona: first call, to get a search query for a recognised food)

```
Write ONE English PubMed search query about how the bioactive compounds of the food below relate to
glucose metabolism. Use the compound and glycaemic keywords, e.g.
sulforaphane AND (glucose OR insulin OR glycemic).
Add AND humans[MeSH Terms] if the user asked about human evidence.
Food: {food}
Reply with the query only.
```

### 4.3 Streamlit snippet (Hyeona)

```python
import json
from src.agent_context import build_agent_prompt
from src.pubmed_search import search_pubmed

agent = get_agent()                                   # from section 2
food = "broccoli"                                     # from photo recognition / user confirmation
query = agent.responses.create(input=QUERY_PROMPT.format(food=food)).output_text.strip()
papers = search_pubmed(query, max_results=3)          # dict, never raises: check papers["error"]
prompt = (build_agent_prompt(user_id, question)
          + "\n\nPubMed results (JSON):\n" + json.dumps(papers, ensure_ascii=False))
answer = agent.responses.create(input=prompt).output_text
```

Show `query` and the PubMed links from `papers["articles"]` in the UI so users can open the real papers.
Results are best-match ranked (`sort="relevance"`) but still include off-topic papers, so the agent must filter.

### 4.4 Checks for the literature answer (Habin)

Run `python src/pubmed_search.py "sulforaphane glucose metabolism"` on your machine to see the raw results first.
PubMed content changes over time, so judge behaviour, not exact papers.

| Check | Pass if the agent... |
| --- | --- |
| Relevance | skips or flags papers not about glucose/insulin (e.g. a bladder-cancer or atherosclerosis paper) |
| Study type | says which cited papers are cell, animal or review studies, and that this is not proof for people |
| Accuracy | cites only PMIDs and titles that are in the JSON, with no extra numbers |
| No results | says nothing was found when `articles` is empty or `error` is set |
| No causation | does not say the simulated CGM rise was caused by a compound |
| Simulated data | mentions that the CGM data is simulated whenever it talks about glucose numbers |