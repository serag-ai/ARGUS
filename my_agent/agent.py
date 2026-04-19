# my_agent/agent.py
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from .utils import tools as mri_tools

from dotenv import load_dotenv
import os

# API KEY

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. "
        "Add it to your environment or to a .env file in the project root."
    )

# SEGMENTATION AGENT  

segmentation_agent = create_react_agent(
    model=init_chat_model(
        "openai:gpt-4.1-mini",
        api_key=OPENAI_API_KEY,
    ),
    tools=[mri_tools.segment_mri_quicknat],
    prompt=(
    "You are the SEGMENTATION AGENT.\n"
    "Task: run QuickNat once and return its JSON.\n"
    "- Always call `segment_mri_quicknat` exactly once.\n"
    "- IMPORTANT: Copy `image_path`, `model_path`, and `orientation` EXACTLY as provided in the latest user message.\n"
    "- Do not leave any argument blank.\n"
    "- If the tool returns ok=false, still return the JSON exactly as-is (do not retry).\n"
    "- Respond ONLY with the tool JSON. No extra text."
),


    name="segmentation_agent",
)

# CHECK AGENT  (validate segmentation)

check_agent = create_react_agent(
    model=init_chat_model("openai:gpt-4.1-mini", api_key=OPENAI_API_KEY),
    tools=[mri_tools.check_segmentation_validity],
    prompt=(
        "You are the CHECK AGENT.\n"
        "Your job is to validate segmentation output using the provided tool and decide whether the workflow should continue.\n\n"

        "Instructions:\n"
        "Call check_segmentation_validity exactly once.\n"
        "Use the modality exactly as provided by the Supervisor Agent.\n"
        "Do not guess, change, or override any input parameter.\n"
        "Do not call any other tool.\n\n"

        "Interpretation of tool result:\n"
        "If valid is true, return decision='PROCEED'.\n"
        "If valid is false, return decision='STOP'.\n"
        "If missing_class_ids is null, return an empty list instead.\n"
        "If present_class_ids is missing or null, return an empty list instead.\n\n"

        "Return format:\n"
        "Return one JSON object only with the following fields:\n"
        '{"decision":"PROCEED or STOP","is_valid":true or false,"present_class_ids":[],"missing_class_ids":[],"reason":"brief explanation"}\n'
        "No extra text."
    ),
    name="check_agent",
)


# VOXEL AGENT 

voxel_agent = create_react_agent(
    model=init_chat_model("openai:gpt-4.1-mini", api_key=OPENAI_API_KEY),
    tools=[mri_tools.compute_volumes_from_segmentation],
    prompt=(
        "You are the VOXEL/VOLUME AGENT.\n"
        "Task: compute voxel counts and volumes from an existing segmentation.\n"
        "- Always call `compute_volumes_from_segmentation` exactly once.\n"
        "- Use image_path, model_path, orientation, device, batch_size, brain_region.\n"
        "- Respond ONLY with the tool JSON. No extra text."
    ),
    name="voxel_agent",
)


# REPORT AGENT 
report_agent = create_react_agent(
    model=init_chat_model(
        "openai:gpt-4.1-mini",
        api_key=OPENAI_API_KEY,
    ),
    tools=[mri_tools.get_case_data_from_cache, mri_tools.merge_case_payloads],
    prompt= (
    "You are the CLINICAL REPORT AGENT.\n"
    "Write a clinical style report using ONLY the tool provided data.\n"
    "Rules:\n"
    "Do not invent, estimate, or interpret findings.\n"
    "If a value is missing, omit it or write Not provided.\n"
    "Use this structure when possible:\n"
    "   Title\n"
    "   Modality\n"
    "   Indication (only if provided)\n"
    "   Findings (short paragraphs)\n"
    "   Measurements or Results (only provided numbers)\n"
    "Keep it concise and clinically worded.\n"
    "Use plain text. No special characters beyond punctuation.\n"
    "Process:\n"
    "Call the appropriate data tool(s) to obtain case data.\n"
    "Write the report from that data.\n"

    ),
    name="report_agent",
)

# WEB SEARCH AGENT (Tavily)

web_search_agent = create_react_agent(
    model=init_chat_model("openai:gpt-4.1-mini", api_key=OPENAI_API_KEY),
    tools=[mri_tools.tavily_web_search],
    prompt=(
    "You are the WEB SEARCH AGENT.\n"
    "Task: run Tavily web search once and return its JSON.\n"
    "- Always call `tavily_web_search` exactly once.\n"
    "- Use ONE concise query only.\n"
    "- Do NOT retry even if results_count is 0 or ok=false.\n"
    "- Respond ONLY with the tool JSON. No extra text."
),
    name="web_search_agent",
)


# PATHOLOGY SEGMENTATION AGENT
pathology_segmentation_agent = create_react_agent(
    model=init_chat_model("openai:gpt-4.1-mini", api_key=OPENAI_API_KEY),
    tools=[mri_tools.pathology_segment],
    prompt=(
        "You are the PATHOLOGY SEGMENTATION AGENT.\n"
        "Task: run pathology segmentation once and return its JSON.\n"
        "- Always call `pathology_segment` exactly once.\n"
        "- Copy `images_folder`, `model_path`, `out_root`, `device`, `batch_size`, `threshold` EXACTLY.\n"
        "- Do not leave any argument blank.\n"
        "- If ok=false, return the JSON as-is (do not retry).\n"
        "- Respond ONLY with the tool JSON. No extra text."
    ),
    name="pathology_segmentation_agent",
)


# CELL COUNT AGENT
cell_count_agent = create_react_agent(
    model=init_chat_model("openai:gpt-4.1-mini", api_key=OPENAI_API_KEY),
    tools=[mri_tools.pathology_count_cells],
    prompt=(
        "You are the CELL COUNT AGENT.\n"
        "Task: count cells using the cached pathology segmentation output and classify the patient.\n"
        
        "Always call pathology_count_cells exactly once.\n"
        "IMPORTANT: Do NOT run segmentation.\n"
        
        "Input: you will be given the cache_key string from pathology_segment output.\n"
        "Call with:\n"
        "cache_key = EXACT provided cache_key string\n"
        "threshold = provided value if any, else keep default\n"
        "min_distance = provided value if any, else keep default\n"
        
        "After receiving the tool result Classify patients as the following:\n"
        "If cell_count is equal to 0 then classification is Healthy\n"
        "If cell_count is more than 0 then classification is Acute Lymphoblastic Leukaemia (ALL)\n"
        
        "Final output MUST be JSON in this format:\n"
        "{\n"
        '  "cell_count": <int>,\n'
        '  "classification": "<Healthy or Acute Lymphoblastic Leukaemia (ALL)>"\n'
        "}\n"
        
        "Do NOT return the raw tool output alone.\n"
        "Do NOT add explanations.\n"
        "Return ONLY the final JSON."
    ),
    name="cell_count_agent",
)

### OCT 

#OCT SEGMENTATION 
oct_segmentation_agent = create_react_agent(
    model=init_chat_model("openai:gpt-4.1-mini", api_key=OPENAI_API_KEY),
    tools=[mri_tools.oct_segment],
    prompt=(
        "You are the OCT SEGMENTATION AGENT.\n"
        "Task: run OCT segmentation once and return its JSON.\n"
        "Always call `oct_segment` exactly once.\n"
        "Copy `image_path`, `model_path`, `out_dir` EXACTLY as provided.\n"
        "Do not leave any argument blank.\n"
        "If ok=false, return the JSON as-is (do not retry).\n"
        "Respond ONLY with the tool JSON. No extra text."
    ),
    name="oct_segmentation_agent",
)

#Retinal Layer thickness calculation
oct_thickness_agent = create_react_agent(
        model=init_chat_model(
            "openai:gpt-4.1-mini",
            api_key=OPENAI_API_KEY,
        ),
        tools=[mri_tools.compute_oct_retinal_thickness_from_prob],
        prompt=(
            "You are the OCT THICKNESS AGENT.\n"
            "Task: compute retinal thickness from an OCT probability map.\n"
            "Rules:\n"
            "Always call `compute_oct_retinal_thickness_from_prob` exactly once.\n"
            "Copy `prob_path` and `axial_um_per_px` EXACTLY from the user input.\n"
            "Do not leave any argument blank.\n"
            "If axial_um_per_px is missing, set it to null.\n"
            "If ok=false, return the tool JSON as-is (do not retry).\n"
            "Respond ONLY with the tool JSON. No extra text."
        ),
    name="oct_thickness_agent"
)
