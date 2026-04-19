#!/usr/bin/env python
# coding: utf-8
"""
my_agent/main.py
Unified runner for MRI + Pathology + OCT.
"""

from __future__ import annotations

from my_agent.supervisor_graph import supervisor

from collections import defaultdict
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from dotenv import load_dotenv
import traceback
import time
import os
import json
import datetime
from pathlib import Path
from glob import glob

# MRI tools are used only for PDF generation here
from my_agent.utils import tools as mri_tools

load_dotenv()



# Modality inference


_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
_NIFTI_EXTS = (".nii", ".nii.gz")

def infer_modality(input_path: str) -> str:
    """
    Conservative modality inference:
      - nifti -> mri
      - folder -> pathology (tiles/images folder) OR ambiguous (if mixed)
      - image -> image_ambiguous (could be OCT or pathology)
      - xyc -> pathology_xyc (optional)
      - else -> unknown

    We keep image inputs ambiguous so the SUPERVISOR can decide OCT vs Pathology using the question.
    """
    p = (input_path or "").strip()
    if not p:
        return "unknown"
    pl = p.lower()

    if pl.endswith(_NIFTI_EXTS):
        return "mri"

    if os.path.isdir(p):
        try:
            files = os.listdir(p)
        except Exception:
            return "unknown"

        has_nifti = any(f.lower().endswith(_NIFTI_EXTS) for f in files)
        has_img = any(f.lower().endswith(_IMAGE_EXTS) for f in files)

        if has_nifti and not has_img:
            return "mri_folder" 
        if has_img:
            return "pathology"
        return "unknown"

    if pl.endswith(_IMAGE_EXTS):
   
        return "image_ambiguous"

    return "unknown"


# Supervisor runner

def run_case_agent(
    user_question: str,
    input_path: str,
) -> dict:
    """
    Runs the supervisor with streaming enabled.
    Injects default model paths/params from env.
    Supervisor picks correct pipeline among MRI, Pathology, OCT.
    """
    if not input_path or not os.path.exists(input_path):
        raise FileNotFoundError(f"input_path does not exist: {input_path}")

    modality_guess = infer_modality(input_path)

    #  MRI defaults 
    mri_model_path = os.getenv("MRI_MODEL_PATH", "").strip()
    mri_orientation = os.getenv("MRI_ORIENTATION", "COR").strip().upper()
    mri_device = os.getenv("MRI_DEVICE", "cpu").strip()
    mri_batch_size = int(os.getenv("MRI_BATCH_SIZE", "20"))

    # Pathology defaults 
    path_model_path = os.getenv("PATH_MODEL_PATH", "").strip()
    path_device = os.getenv("PATH_DEVICE", "cpu").strip()
    path_batch_size = int(os.getenv("PATH_BATCH_SIZE", "8"))
    path_seg_threshold = float(os.getenv("PATH_SEG_THRESHOLD", "0.5"))
    path_count_threshold = float(os.getenv("PATH_COUNT_THRESHOLD", "0.5"))
    path_min_distance = int(os.getenv("PATH_MIN_DISTANCE", "10"))

    #OCT defaults
    oct_model_path = os.getenv("OCT_MODEL_PATH", "").strip()
    oct_out_dir_name = os.getenv("OCT_OUT_DIR_NAME", "oct_outputs").strip()

    injected_lines = [
        f"input_path: {input_path}",
        f"modality_guess: {modality_guess}",
        f"user_question: {user_question}",
    ]

   
    base_dir = input_path if os.path.isdir(input_path) else str(Path(input_path).parent)

    if modality_guess == "mri":
        if not mri_model_path or not os.path.exists(mri_model_path):
            raise FileNotFoundError(f"MRI_MODEL_PATH is not set or missing: {mri_model_path!r}")

        injected_lines += [
            "PIPELINE_HINT: MRI",
            f"image_path: {input_path}",
            f"mri_model_path: {mri_model_path}",
            f"orientation: {mri_orientation}",
            f"device: {mri_device}",
            f"batch_size: {mri_batch_size}",
        ]

   
    elif modality_guess == "pathology":
        if not path_model_path or not os.path.exists(path_model_path):
            raise FileNotFoundError(f"PATH_MODEL_PATH is not set or missing: {path_model_path!r}")

        images_folder = input_path  
        out_dir = os.path.join(base_dir, "pathology_outputs")

        injected_lines += [
            "PIPELINE_HINT: PATHOLOGY",
            f"images_folder: {images_folder}",
            f"path_model_path: {path_model_path}",
            f"out_dir: {out_dir}",
            f"device: {path_device}",
            f"batch_size: {path_batch_size}",
            f"threshold: {path_seg_threshold}",
            f"seg_threshold: {path_seg_threshold}",
            f"count_threshold: {path_count_threshold}",
            f"min_distance: {path_min_distance}",
        ]

    elif modality_guess == "image_ambiguous":
        injected_lines += [
            "PIPELINE_HINT: IMAGE_AMBIGUOUS (OCT vs PATHOLOGY)",
            f"image_path: {input_path}",
            f"base_dir: {base_dir}",
        ]

        if path_model_path and os.path.exists(path_model_path):
            out_dir = os.path.join(base_dir, "pathology_outputs")
            injected_lines += [
                f"images_folder: {input_path}",
                f"path_model_path: {path_model_path}",
                f"path_out_dir: {out_dir}",
                f"path_device: {path_device}",
                f"path_batch_size: {path_batch_size}",
                f"path_threshold: {path_seg_threshold}",
                f"path_seg_threshold: {path_seg_threshold}",
                f"path_count_threshold: {path_count_threshold}",
                f"path_min_distance: {path_min_distance}",
            ]

        if oct_model_path and os.path.exists(oct_model_path):
            oct_out_dir = os.path.join(base_dir, oct_out_dir_name)
            injected_lines += [
                f"oct_model_path: {oct_model_path}",
                f"oct_out_dir: {oct_out_dir}",
            ]


    else:
        injected_lines += ["PIPELINE_HINT: UNKNOWN (Supervisor must decide using question keywords)"]

        if mri_model_path and os.path.exists(mri_model_path):
            injected_lines += [
                f"mri_model_path: {mri_model_path}",
                f"mri_orientation: {mri_orientation}",
                f"mri_device: {mri_device}",
                f"mri_batch_size: {mri_batch_size}",
            ]

        if path_model_path and os.path.exists(path_model_path):
            injected_lines += [
                f"path_model_path: {path_model_path}",
                f"path_device: {path_device}",
                f"path_batch_size: {path_batch_size}",
                f"path_seg_threshold: {path_seg_threshold}",
                f"path_count_threshold: {path_count_threshold}",
                f"path_min_distance: {path_min_distance}",
            ]

        if oct_model_path and os.path.exists(oct_model_path):
            injected_lines += [
                f"oct_model_path: {oct_model_path}",
                f"oct_out_dir_name: {oct_out_dir_name}",
            ]

    system_msg = SystemMessage(
        content=(
            "You are the SUPERVISOR coordinating MRI, Pathology, and OCT agents.\n"
            "HARD REQUIREMENTS:\n"
            "Never modify any provided paths or parameters.\n"
            "Choose the correct pipeline using input_path + user_question.\n"
            "Each agent may be called only once.\n"
            "Stop after final output for the chosen pipeline.\n\n"
            "BRANCHING GUIDANCE:\n"
            "If input is .nii/.nii.gz then it could MRI.\n"
            "If input is a folder of many images, decide between OCT and Pathology based on question keywords.\n"
            "If input is a single image file, decide between OCT and Pathology based on question keywords.\n"
            "OCT keywords: oct, retina, macula, rnfl, gcl, ipl, onl, rpe, choroid, optic disc or calculate thickness\n"
            "Pathology keywords: cell, nuclei, histology, slide, stain, tumor, mitosis.\n"
            "If both model paths exist, pick the best match; if one is missing, use the available one.\n"
        )
    )

    user_msg = HumanMessage(
        content=(
            "Resolved inputs/params (copy EXACTLY when calling agents/tools):\n"
            + "\n".join(injected_lines)
        )
    )

    events = []
    for chunk in supervisor.stream({"messages": [system_msg, user_msg]}, subgraphs=True):
        events.append(chunk)

    return {"events": events, "modality_guess": modality_guess, "input_path": input_path}



def _get_msg_role_content(msg):
    role = None
    content = getattr(msg, "content", None)

    if isinstance(msg, SystemMessage):
        role = "system"
    elif isinstance(msg, HumanMessage):
        role = "user"
    elif isinstance(msg, AIMessage):
        role = "assistant_tool_call" if getattr(msg, "tool_calls", None) else "assistant"
    elif isinstance(msg, ToolMessage):
        role = "tool"

    if isinstance(msg, dict):
        role = msg.get("role", role) or msg.get("type", role)
        content = msg.get("content", content)

    if role is None:
        role = getattr(msg, "type", None) or msg.__class__.__name__
    if content is None:
        content = str(msg)

    return role, content


def unpack_events(run_result: dict) -> dict:
    events = run_result.get("events", [])
    agent_traces = defaultdict(list)
    seen = defaultdict(set)
    final_answer = None

    for event in events:
        if not isinstance(event, tuple) or len(event) != 2:
            continue

        key, payload = event
        if key == () and isinstance(payload, dict):
            continue

        node_name = None
        msgs = []

        if isinstance(key, tuple) and len(key) > 0 and isinstance(payload, dict):
            raw = key[0]
            node_name = raw.split(":", 1)[0]

            if "agent" in payload and isinstance(payload.get("agent"), dict):
                msgs = payload["agent"].get("messages", []) or []
            elif "tools" in payload and isinstance(payload.get("tools"), dict):
                msgs = payload["tools"].get("messages", []) or []

        if not node_name or not msgs:
            continue

        for m in msgs:
            role, content = _get_msg_role_content(m)
            fp = (role, content)
            if fp in seen[node_name]:
                continue
            seen[node_name].add(fp)
            agent_traces[node_name].append({"role": role, "content": content})

        if node_name == "supervisor":
            for m in reversed(msgs):
                role, content = _get_msg_role_content(m)
                if role == "assistant" and isinstance(content, str) and content.strip():
                    final_answer = content.strip()
                    break

    if final_answer is None and "supervisor" in agent_traces:
        for step in reversed(agent_traces["supervisor"]):
            if step.get("role") == "assistant" and isinstance(step.get("content"), str) and step["content"].strip():
                final_answer = step["content"].strip()
                break

    return {"final_answer": final_answer, "agent_traces": dict(agent_traces)}



# Tool JSON extraction 


def _safe_json_loads(x):
    if x is None:
        return None
    if isinstance(x, dict):
        return x
    if isinstance(x, (list, tuple)):
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    s2 = s.strip("`")
    try:
        return json.loads(s2)
    except Exception:
        return None


def _unwrap_possible_output_dict(d: dict) -> dict:
    if not isinstance(d, dict):
        return {}
    for k in ("output", "result", "data"):
        if k in d and isinstance(d[k], dict):
            return d[k]
    return d


def _find_key_anywhere(agent_traces: dict, keys: tuple[str, ...]) -> str:
    for _, steps in agent_traces.items():
        for step in reversed(steps):
            obj = _safe_json_loads(step.get("content"))
            if not isinstance(obj, dict):
                continue
            obj2 = _unwrap_possible_output_dict(obj)
            for k in keys:
                v = obj2.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
    return ""



# MRI PDF creation 


def _call_create_pdf_tool(payload: dict) -> dict:
    tool_obj = mri_tools.create_pdf_final_report
    if hasattr(tool_obj, "invoke"):
        return tool_obj.invoke(payload)
    return tool_obj(**payload)

def _extract_last_tool_json(agent_traces: dict, agent_name: str):
    steps = agent_traces.get(agent_name, [])

    # 1) Prefer role == "tool"
    for step in reversed(steps):
        if step.get("role") != "tool":
            continue
        obj = _safe_json_loads(step.get("content"))
        if isinstance(obj, dict):
            return obj

    # 2) Fallback: any dict (kept for backward compatibility)
    for step in reversed(steps):
        obj = _safe_json_loads(step.get("content"))
        if isinstance(obj, dict):
            return obj

    return None

def _extract_last_decision_json(agent_traces: dict) -> dict | None:
    for _, steps in agent_traces.items():
        for step in reversed(steps):
            if step.get("role") != "tool":
                continue
            obj = _safe_json_loads(step.get("content"))
            if isinstance(obj, dict):
                obj2 = _unwrap_possible_output_dict(obj)
                if "decision" in obj2:
                    return obj
    return None

def _extract_report_text(agent_traces: dict, parsed_final_answer: str | None):
    steps = agent_traces.get("report_agent", [])
    for step in reversed(steps):
        c = step.get("content")
        if isinstance(c, str) and c.strip():
            if _safe_json_loads(c) is None:
                return c.strip()
    if isinstance(parsed_final_answer, str) and parsed_final_answer.strip():
        return parsed_final_answer.strip()
    return None



def create_pdf_if_complete(
    parsed: dict,
    *,
    modality: str,
    image_path: str,
    model_path: str,
    orientation: str | None = None, 
) -> dict:
    agent_traces = parsed.get("agent_traces", {}) or {}

    modality_u = (modality or "").strip().upper()

    if modality_u in ("MRI", "OCT"):
            chk = _extract_last_tool_json(agent_traces, "check_agent")
            if not isinstance(chk, dict):
                return {"created": False, "pdf_result": None, "reason": "Missing check_agent output (no PDF)."}

            chk2 = _unwrap_possible_output_dict(chk)
            if not isinstance(chk2, dict):
                chk2 = chk

            # {"decision": "PROCEED"/"STOP", ...}
            # {"segmentation_valid": true/false, ...}
            if "decision" in chk2:
                decision = str(chk2.get("decision", "")).strip().upper()
                if decision != "PROCEED":
                    return {"created": False, "pdf_result": None, "reason": f"check_agent decision={decision} (no PDF)."}

            elif "valid" in chk2:
                seg_valid = chk2.get("valid", False)
                if seg_valid is not True:
                    reason = str(chk2.get("reason", "valid=false")).strip()
                    return {"created": False, "pdf_result": None, "reason": f"check_agent invalid: {reason} (no PDF)."}

            else:
                return {
                    "created": False,
                    "pdf_result": None,
                    "reason": "check_agent output missing 'decision' and 'valid' (no PDF).",
                }

    elif modality_u == "PATHOLOGY":
        patho = _extract_last_tool_json(agent_traces, "pathology_segmentation_agent")
        if not isinstance(patho, dict):
            return {"created": False, "pdf_result": None, "reason": "Missing pathology_segmentation_agent output (no PDF)."}

        
        ok_flag = patho.get("ok", None)
        if ok_flag is not True:
            return {"created": False, "pdf_result": None, "reason": f"Pathology segmentation ok!=true (ok={ok_flag!r}) (no PDF)."}

    else:
        return {"created": False, "pdf_result": None, "reason": f"Unsupported modality={modality!r} (no PDF)."}


    # REPORT TEXT 
    report_steps = agent_traces.get("report_agent", []) or []
    report_text = None
    for step in reversed(report_steps):
        c = step.get("content")
        if isinstance(c, str) and c.strip():
            if _safe_json_loads(c) is None:  
                report_text = c.strip()
                break

    if not report_text:
        return {"created": False, "pdf_result": None, "reason": "Missing report_agent output (no PDF)."}


   
    base = os.path.basename(image_path)
    patient_id = base.replace(".nii.gz", "").replace(".nii", "")
    patient_id = os.path.splitext(patient_id)[0]

  
    payload = {
        "modality": modality_u,
        "patient_id": patient_id,
        "image_path": image_path,
        "model_path": model_path,
        "report_text": report_text,
        "out_pdf_path": None,
        "web_search_agent": None,
    }

    if modality_u == "MRI":
        if not orientation:
            return {"created": False, "pdf_result": None, "reason": "Missing orientation for MRI (no PDF)."}

        seg = _extract_last_tool_json(agent_traces, "segmentation_agent") or {}
        seg2 = _unwrap_possible_output_dict(seg)

        segmentation_path = (seg2.get("segmentation_path") or "").strip()
        if not segmentation_path:
            segmentation_path = _find_key_anywhere(agent_traces, ("segmentation_path", "mask_path", "seg_path"))

        if not segmentation_path:
            return {"created": False, "pdf_result": None, "reason": "Could not locate segmentation_path (no PDF)."}
        if not os.path.exists(segmentation_path):
            return {"created": False, "pdf_result": None, "reason": f"segmentation_path does not exist: {segmentation_path!r}"}

        payload["segmentation_path"] = segmentation_path
        payload["orientation"] = orientation

    elif modality_u == "OCT":
        octo = _extract_last_tool_json(agent_traces, "oct_segmentation_agent") or {}
        octo2 = _unwrap_possible_output_dict(octo)

        prob_path = (octo2.get("prob_path") or "").strip()
        if not prob_path:
            prob_path = _find_key_anywhere(agent_traces, ("prob_path", "probs_path", "logits_path", "prediction_path"))

        if not prob_path:
            return {"created": False, "pdf_result": None, "reason": "Could not locate prob_path (no PDF)."}
        if not os.path.exists(prob_path):
            return {"created": False, "pdf_result": None, "reason": f"prob_path does not exist: {prob_path!r}"}

        payload["prob_path"] = prob_path

    elif modality_u == "PATHOLOGY":
        patho = _extract_last_tool_json(agent_traces, "pathology_segmentation_agent") or {}
        art = patho.get("artifacts", {})
        if not isinstance(art, dict):
            art = {}

        heatmap_path = ""
        hp = art.get("heatmap_paths")
        if isinstance(hp, list) and len(hp) > 0 and isinstance(hp[0], str):
            heatmap_path = hp[0].strip()

        if not heatmap_path:
            heatmap_path = (art.get("heatmap_path") or "").strip()

        if not heatmap_path:
            heatmap_path = _find_key_anywhere(agent_traces, ("heatmap_paths", "heatmap_path", "pred_heatmap_path"))

        if not heatmap_path:
            return {"created": False, "pdf_result": None, "reason": "Could not locate heatmap_path (no PDF)."}

        if not os.path.exists(heatmap_path):
            return {"created": False, "pdf_result": None, "reason": f"heatmap_path does not exist: {heatmap_path!r}"}

        payload["heatmap_path"] = heatmap_path

    pdf_result = _call_create_pdf_tool(payload)
    if not bool((pdf_result or {}).get("ok")):
        return {"created": False, "pdf_result": pdf_result, "reason": "PDF tool returned ok=False."}

    return {"created": True, "pdf_result": pdf_result, "reason": "PDF created successfully."}

# Logging
def write_agent_logs(parsed: dict, input_path: str, log_root: str | None = None) -> dict:
    agent_traces = parsed.get("agent_traces", {}) or {}
    base = os.path.basename(input_path.rstrip("/"))
    case_id = base.replace(".nii.gz", "").replace(".nii", "")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if log_root is None:
        parent = os.path.dirname(input_path) if os.path.isfile(input_path) else input_path
        if not parent or not os.path.exists(parent):
            parent = os.getcwd()
        log_root = os.path.join(parent, "logs")

    log_dir = Path(log_root) / case_id / ts
    log_dir.mkdir(parents=True, exist_ok=True)

    out_files: dict[str, str] = {}

    for agent_name, steps in agent_traces.items():
        fp = log_dir / f"{agent_name}.log"
        with fp.open("w", encoding="utf-8") as f:
            f.write(f"agent={agent_name}\ncase_id={case_id}\ntimestamp={ts}\n\n")
            for i, step in enumerate(steps, start=1):
                role = step.get("role") or "unknown"
                content = step.get("content") or ""
                if not isinstance(content, str):
                    try:
                        content = json.dumps(content, ensure_ascii=False, indent=2)
                    except Exception:
                        content = str(content)
                f.write(f"--- step {i} | role={role} ---\n{content}\n\n")
        out_files[agent_name] = str(fp)

    combined_fp = log_dir / "ALL_AGENTS.log"
    with combined_fp.open("w", encoding="utf-8") as f:
        for agent_name in sorted(agent_traces.keys()):
            f.write("=" * 80 + "\n")
            f.write(f"AGENT: {agent_name}\n")
            f.write("=" * 80 + "\n\n")
            for i, step in enumerate(agent_traces[agent_name], start=1):
                role = step.get("role") or "unknown"
                content = step.get("content") or ""
                if not isinstance(content, str):
                    try:
                        content = json.dumps(content, ensure_ascii=False, indent=2)
                    except Exception:
                        content = str(content)
                f.write(f"--- step {i} | role={role} ---\n{content}\n\n")

    out_files["ALL_AGENTS"] = str(combined_fp)
    return {"log_dir": str(log_dir), "files": out_files}



# High-level safe wrapper


def safe_run_one(
    *,
    question: str,
    input_path: str,
    save_mri_pdf: bool = True,
    verbose: bool = True,
) -> dict:
    t0 = time.perf_counter()

    try:
        raw_state = run_case_agent(user_question=question, input_path=input_path)
        parsed = unpack_events(raw_state)

        log_info = write_agent_logs(parsed, input_path=input_path)

        pdf_info = {"created": False, "reason": "Not attempted", "pdf_result": None}

        if save_mri_pdf:
            agent_traces = parsed.get("agent_traces", {}) or {}
            keys = set(agent_traces.keys())

    
            if ("oct_segmentation_agent" in keys) or ("oct_thickness_agent" in keys):
                modality = "OCT"
                model_path = os.getenv("OCT_MODEL_PATH", "").strip()
                orientation = None

            elif ("pathology_segmentation_agent" in keys) or ("cell_count_agent" in keys):
                modality = "PATHOLOGY"
                model_path = os.getenv("PATH_MODEL_PATH", "").strip()
                orientation = None

            elif "segmentation_agent" in keys:
                modality = "MRI"
                model_path = os.getenv("MRI_MODEL_PATH", "").strip()
                orientation = os.getenv("MRI_ORIENTATION", "COR").strip().upper()

            else:
                pdf_info = {
                    "created": False,
                    "reason": "Could not infer modality from executed agents (no PDF).",
                    "pdf_result": None,
                }
                model_path = ""
                orientation = None
                modality = "UNKNOWN"

            if modality != "UNKNOWN":
                if model_path and os.path.exists(model_path):
                    pdf_info = create_pdf_if_complete(
                        parsed=parsed,
                        modality=modality,
                        image_path=input_path,
                        model_path=model_path,
                        orientation=orientation,
                    )
                else:
                    pdf_info = {
                        "created": False,
                        "reason": f"{modality}_MODEL_PATH missing; cannot build PDF.",
                        "pdf_result": None,
                    }

        t1 = time.perf_counter()
        return {
            "ok": True,
            "input_path": input_path,
            "elapsed_s": t1 - t0,
            "final_answer": parsed.get("final_answer", "") or "",
            "log_dir": (log_info or {}).get("log_dir", ""),
            "pdf_created": bool(pdf_info.get("created", False)),
            "pdf_path": ((pdf_info.get("pdf_result") or {}).get("pdf_path", "")),
            "pdf_reason": pdf_info.get("reason", ""),
        }

    except GraphRecursionError as e:
        t1 = time.perf_counter()
        return {
            "ok": False,
            "skipped_due_to_recursion": True,
            "input_path": input_path,
            "elapsed_s": t1 - t0,
            "error": f"GraphRecursionError: {e}",
            "traceback": traceback.format_exc(),
        }

    except Exception as e:
        t1 = time.perf_counter()
        return {
            "ok": False,
            "skipped_due_to_recursion": False,
            "input_path": input_path,
            "elapsed_s": t1 - t0,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }


# CLI entrypoint

def collect_input_files(input_path, exts=None):
    """
    Returns a list of files.
    Supports:
      - single file
      - folder of files
    """
    if exts is None:
        exts = [".nii", ".nii.gz", ".png", ".jpg", ".jpeg", ".bmp"]

    if os.path.isfile(input_path):
        return [input_path]

    if os.path.isdir(input_path):
        files = []
        for ext in exts:
            files.extend(glob(os.path.join(input_path, f"*{ext}")))
        return sorted(files)

    raise ValueError(f"Invalid input_path: {input_path}")

if __name__ == "__main__":

    # Pathology
    # question = "Count the cells found in the provided image and give me a report if the patient is healthy or not."
    # input_path = "/content/drive/MyDrive/Colab Notebooks/Agents/im_test"

    # OCT
    # question = "Calculate the retinal layes thickness and then compare them normal range from literature before generate a clinical report."
    # input_path = "/content/drive/MyDrive/Colab Notebooks/Agents/oct_dataset/oct_dataset_opened/oct_dataset/test/img"

    # MRI
    question = "Compute volumetric measurements for Right Amygdala and Left Hippocampus, then compare it with normal range from literature before generate a clinical report."
    input_path = "/content/drive/MyDrive/Colab Notebooks/Agents/MRI dataset_20 example"


    
  
    
    files = collect_input_files(input_path)

    print(f"\nFound {len(files)} files to process\n")

    total_start = time.perf_counter()
    success, failed = 0, 0

    

    
    for i, file_path in enumerate(files, 1):

        print("=" * 60)
        print(f"[{i}/{len(files)}] Processing: {file_path}")

        start_time = time.perf_counter()

        try:
            result = safe_run_one(
                question=question,
                input_path=file_path,
                save_mri_pdf=True,
                verbose=True
            )

            elapsed = time.perf_counter() - start_time

            print(f"\nTime: {elapsed:.2f} sec")

            if result.get("ok", False):
                success += 1

                print(f"Logs: {result.get('log_dir','')}")

                if result.get("pdf_created", False):
                    print(f"PDF: {result.get('pdf_path','')}")
                else:
                    print(f"No PDF: {result.get('pdf_reason','')}")

                print("\nFINAL OUTPUT:")
                print(result.get("final_answer", "") or "")

            else:
                failed += 1
                print("FAILED")
                print(result.get("error", "Unknown error"))

        except Exception as e:
            failed += 1
            print("CRASHED")
            print(str(e))

    
   
    total_time = time.perf_counter() - total_start

    print("\n" + "=" * 60)
    print("BATCH SUMMARY")
    print("=" * 60)
    print(f"Total files : {len(files)}")
    print(f"Success     : {success}")
    print(f"Failed      : {failed}")
    print(f"Total time  : {total_time:.2f} sec")
    print(f"Avg / case  : {total_time / max(len(files),1):.2f} sec")


    
# if __name__ == "__main__":
#     #Pathology exmaple
#     # question = "Perform quantitative cell enumeration on the provided image and report the total cell count."
#     # input_path = "/acfs-home/hoh4002/serag_AI_lab/users/hoh4002/eICU/Agentic_BrAIn/pathology_images/im/Im008_1.jpg"

#     # #OCT Example 
#     # question = "Segment the retina image and calculate the thickness."
#     # input_path ="/acfs-home/hoh4002/serag_AI_lab/users/hoh4002/eICU/Agentic_BrAIn/oct_dataset/oct_dataset_opened/oct_dataset/test/img/3_R_00_flip.bmp"

#     #MRI Example
#     question = "Compute volumetric measurements for Left Hippocampus, then compare it with normal range from literature before generate a clinical report."
#     input_path ="/content/drive/MyDrive/Colab Notebooks/Agents/oct_dataset/oct_dataset_opened/oct_dataset/test/img/11_L_07_flip.bmp"

#     start_time = time.perf_counter()
#     result = safe_run_one(question=question, input_path=input_path, save_mri_pdf=True, verbose=True)
#     end_time = time.perf_counter()

#     print(f"\nTotal agent run time: {end_time - start_time:.2f} seconds\n")
#     if result.get("ok", False):
#         print(f"Logs saved to: {result.get('log_dir','')}")
#         if result.get("pdf_created", False):
#             print("\nPDF REPORT\n")
#             print(f"PDF created: {result.get('pdf_path','')}")
#         else:
#             print("\nPDF REPORT\n")
#             print(f"No PDF generated: {result.get('pdf_reason','')}")
#         print("\nFINAL OUTPUT\n")
#         print(result.get("final_answer", "") or "")
#     else:
#         print("RUN FAILED")
#         print(result.get("error", "Unknown error"))
#         print(result.get("traceback", ""))