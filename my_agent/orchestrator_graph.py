# my_agent/supervisor_graph.py
from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

from .agent import (
    MRI_processing_agent,
    verification_agent,
    voxel_agent,
    pathology_processing_agent,
    cell_count_agent,
    oct_thickness_agent,
    knowledge_retrieval_agent,
    report_agent,
    oct_processing_agent,
)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise RuntimeError("OPENAI_API_KEY is not set.")

orchestrator = create_supervisor(
    model=init_chat_model("openai:gpt-4.1-mini", api_key=OPENAI_API_KEY),
    agents=[
        MRI_processing_agent,
        verification_agent,
        voxel_agent,
        pathology_processing_agent,
        cell_count_agent,
        oct_thickness_agent,
        knowledge_retrieval_agent,
        report_agent,
        oct_processing_agent,
    ],
    prompt=(
        "You are the orchestrator AGENT. Select one workflow MRI, Pathology or OCT and do not repeat work.\n\n"

        "At the beginning state which workflow you selected and list the agents that will be used to complete the task\n\n"

        "General rules\n"
        "Each agent can be called only once\n"
        "Check history first If output exists do not call again\n"
        "Do not change or invent paths or parameters\n"
        "If any agent returns ok false return that error and stop\n"
        "Always call report_agent at the end of a successful workflow\n\n"

        "Workflow selection\n"
        "Volume or brain MRI or nii use MRI\n"
        "Count cells nuclei tumor use Pathology\n"
        "Thickness, Retina or OCT use OCT\n"
        "If conflict prioritize user intent volume count thickness\n\n"

        "MRI\n"
        "If segmentation missing call MRI_processing_agent\n"
        "Always call verification_agent after segmentation\n"
        "If decision is STOP return and stop\n"
        "If volumes missing call voxel_agent\n"
        "If literature requested call knowledge_retrieval_agent\n"
        "Then call report_agent and stop\n\n"

        "Pathology\n"
        "If heatmaps missing call pathology_processing_agent\n"
        "If ok true and counts missing call cell_count_agent\n"
        "If literature requested call knowledge_retrieval_agent\n"
        "Then call report_agent and stop\n\n"

        "OCT\n"
        "If segmentation missing call oct_processing_agent\n"
        "Always call verification_agent after segmentation\n"
        "If decision is STOP return and stop\n"
        "If thickness missing call oct_thickness_agent\n"
        "If literature requested call knowledge_retrieval_agent\n"
        "Then call report_agent and stop\n\n"

        "Stop conditions\n"
        "Stop on any error\n"
        "Stop if required output missing after agent already called\n"
        "Stop after report_agent completes\n"
    ),
    add_handoff_back_messages=True,
    output_mode="full_history",
).compile()

# graph = supervisor.get_graph()

# with open("supervisor_graph_fullframework.png", "wb") as f:
#     f.write(graph.draw_png())
