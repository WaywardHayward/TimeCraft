

## Requirement
You may need visit https://developers.google.com/custom-search/v1/overview to collect google search key.
## get template

### 1. Collect Candidate Text Descriptions from the Web
Use this option to collect candidate texts that describe time series data from open web sources. The agent will search and return content-rich examples (e.g., essays, reports, papers).

```bash
python self_refine_main.py \
  --openai_key YOUR_OPENAI_API_KEY \
  --serpapi_key YOUR_SERPAPI_API_KEY \
  --google_key YOUR_GOOGLE_API_KEY \
  --google_search_engine YOUR_GOOGLE_SEARCH_ENGINE_ID \
  --collect_candidate
```
### 2. Extract Template
📌Extract Templates after refinement
```bash
python self_refine_main.py \
    --openai_key YOUR_OPENAI_KEY \
    --extract_template \
    --template_input_file refinement_result.json
```
📌 Extract Templates from a Text Document, CSV, or JSONL File
```bash
python self_refine_main.py \
    --openai_key YOUR_OPENAI_KEY \
    --extract_template \
    --template_input_file document.txt
```
## get text 
### 1. Generate Text Descriptions from Time Series Data
Convert your time series data into descriptive text by providing the time series CSV file and a corpus file containing example templates.

```bash
python self_refine_main.py \
    --ts_to_text \
    --ts_file ./data/electricity.csv \
    --dataset_name Electricity \
    --dataset_template_file ./data/dataset_templates.json \
    --description_template_file ./data/description_templates.json \
    --openai_key YOUR_OPENAI_KEY\
    --llm_optimize
```
### 2. Self-Refinement for Improving Time Series Descriptions
Run the self-refinement process to iteratively improve time series text descriptions based on feedback. You can configure the number of iterations and test runs.
```bash
python self_refine_main.py \
    --openai_key YOUR_OPENAI_KEY \
    --refine \
    --ts_file path/to/your/timeseries.csv \
    --dataset_name xxx \
    --global_iterations 2 \
    --team_iterations 3 \
    --output_file refinement_result.json
```




### Feedback
```json
=== FEEDBACK REPORT ===
{
    "mse": 31.95833333333334,
    "ks_stat": 0.041666666666666664,
    "ks_p_value": 0.9999999999999998,
    "wasserstein_distance": 8.625,
    "text_quality_scores": {
        "accuracy_of_trend": "5/5",
        "mention_of_seasonality": "3/5",
        "reference_to_external_factors": "1/5",
        "clarity_of_description": "4/5",
        "completeness_of_information": "3/5"
    },
    "suggestions": {
        "accuracy_of_trend": "No changes needed.",
        "mention_of_seasonality": "Consider mentioning the lack or presence of seasonality explicitly.",
        "reference_to_external_factors": "Mention external factors such as policy changes, economic events, or other influences that could impact the trend.",
        "clarity_of_description": "Use more precise language and avoid vague terms to enhance clarity.",
        "completeness_of_information": "Include specific peak and dip values along with their corresponding timestamps to provide a more complete picture."
    },
    "text_feedback_summary": "The description captures the trend well, but lacks references to seasonality and external factors. Improving clarity and completeness by including specific values and potential external influences would enhance the overall quality."
}
```
