import os
import json
import re
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from pipeline.cv_module import detect_components
from pipeline.topology_module import build_topology, build_jianzi_sequence
from pipeline.llm_module import infer_pitch_duration
from pipeline.musicxml_encoder import generate_musicxml
from pipeline.score_model_transformer import transform_llm_result_to_score_model

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 配置上传目录
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 强制输出 utf-8 编码
app.config['JSON_AS_ASCII'] = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_IMAGE_PATH = os.path.join(PROJECT_ROOT, 'test', 'testpicture-1.jpg')

@app.route('/static/uploads/<filename>')
def serve_upload(filename):
    """提供上传图片的静态访问路径"""
    return send_from_directory(UPLOAD_FOLDER, filename)


def _run_pipeline_and_save_result(save_path: str):
    """
    执行完整流水线并将结果落盘为 *_result.json。
    """
    # 1. 视觉感知
    yolo_boxes = detect_components(save_path)

    # 2. 空间拓扑解析
    topology_json = build_topology(yolo_boxes)

    # 2.1 减字序列化（右到左列序，供 LLM 使用的轻量结构）
    jianzi_sequence = build_jianzi_sequence(topology_json)

    # 3. 大模型打谱推理 (音高与节奏)
    llm_result = infer_pitch_duration(jianzi_sequence)

    # 3.1 统一前端渲染模型（JSON 规范化）
    score_model = transform_llm_result_to_score_model(llm_result, strict=False)

    # 4. XML 编码层
    music_xml = generate_musicxml(llm_result)

    result_payload = {
        "yolo_boxes": yolo_boxes,
        "topology_json": topology_json,
        "jianzi_sequence": jianzi_sequence,
        "llm_result": llm_result,
        "score_model": score_model,
        "music_xml": music_xml,
    }

    result_json_path = save_path + "_result.json"
    with open(result_json_path, 'w', encoding='utf-8-sig') as f:
        json.dump(result_payload, f, ensure_ascii=False, indent=2)

    return result_payload


def _group_sort_key(group_id: str):
    match = re.search(r"(\d+)", str(group_id))
    if not match:
        return (10**9, str(group_id))
    return (int(match.group(1)), str(group_id))


def _sanitize_topology_for_reflow(raw_topology):
    """
    清洗前端回传的 topology_json：
    - 过滤非对象组与 __deleted 软删除组
    - 统一关键字段，兼容新旧字段名
    - 重排 sequence_index，确保后续链路顺序稳定
    """
    if not isinstance(raw_topology, dict):
        return {}

    sanitized = {}
    sequence_index = 1

    for group_id, payload in sorted(raw_topology.items(), key=lambda item: _group_sort_key(item[0])):
        if not isinstance(payload, dict):
            continue
        if bool(payload.get("__deleted")):
            continue

        group = dict(payload)
        group.pop("__deleted", None)
        group["sequence_index"] = sequence_index
        sequence_index += 1

        right_fingering = str(group.get("right_fingering") or "").strip()
        left_fingering = str(group.get("left_fingering") or "").strip()
        left_finger = str(group.get("left_finger") or group.get("finger") or "").strip()
        hui = str(group.get("hui") or group.get("position") or "").strip()
        xian = str(group.get("xian") or group.get("string") or group.get("xian_digit") or "").strip()

        group["right_fingering"] = right_fingering
        group["left_fingering"] = left_fingering
        group["left_finger"] = left_finger
        group["hui"] = hui
        group["xian"] = xian

        # 兼容旧下游字段
        group["fingering"] = str(group.get("fingering") or right_fingering or left_fingering or "").strip()
        group["finger"] = str(group.get("finger") or left_finger or "").strip()
        group["position"] = str(group.get("position") or hui or "").strip()
        group["string"] = str(group.get("string") or xian or "").strip()

        sanitized[str(group_id)] = group

    return sanitized

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    处理前端图片上传。
    优化：上传阶段采用原始字节直写，避免 imdecode+imencode 带来的额外耗时。
    """
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400

    if file:
        filename = file.filename
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        
        raw_bytes = file.read()
        if not raw_bytes:
            return jsonify({'status': 'error', 'message': 'Invalid image file'}), 400

        # 直接写入上传字节，避免上传阶段额外解码/编码开销
        try:
            with open(save_path, 'wb') as f:
                f.write(raw_bytes)
        except Exception:
            return jsonify({'status': 'error', 'message': 'Failed to save image'}), 500

        try:
            result_payload = _run_pipeline_and_save_result(save_path)
        except Exception as exc:
            return jsonify({
                'status': 'error',
                'message': f'Pipeline execution failed: {exc}'
            }), 500

        return jsonify({
            'status': 'success',
            'data': {
                'original_image_url': f'/static/uploads/{filename}',
                **result_payload
            }
        })


@app.route('/api/run_testpicture_pipeline', methods=['GET'])
def run_testpicture_pipeline():
    """
    直接使用 test/testpicture-1.jpg 走完整流水线，便于前端一键端到端验证。
    """
    if not os.path.exists(TEST_IMAGE_PATH):
        return jsonify({
            'status': 'error',
            'message': f'test image not found: {TEST_IMAGE_PATH}'
        }), 404

    filename = 'testpicture-1.jpg'
    save_path = os.path.join(UPLOAD_FOLDER, filename)

    file_bytes = np.fromfile(TEST_IMAGE_PATH, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'status': 'error', 'message': 'Invalid test image file'}), 400

    extension = os.path.splitext(filename)[1].lower() or '.jpg'
    is_success, buffer = cv2.imencode(extension, img)
    if not is_success:
        return jsonify({'status': 'error', 'message': 'Failed to save test image'}), 500
    buffer.tofile(save_path)

    try:
        result_payload = _run_pipeline_and_save_result(save_path)
    except Exception as exc:
        return jsonify({
            'status': 'error',
            'message': f'Pipeline execution failed: {exc}'
        }), 500

    return jsonify({
        'status': 'success',
        'data': {
            'original_image_url': f'/static/uploads/{filename}',
            **result_payload
        }
    })


@app.route('/api/reflow_from_topology', methods=['POST'])
def reflow_from_topology():
    """
    基于前端修正后的 topology_json，仅重跑后续链路：
    topology -> jianzi_sequence -> llm -> score_model -> musicxml
    """
    body = request.get_json(silent=True) or {}
    topology_payload = body.get("topology_json")
    sanitized_topology = _sanitize_topology_for_reflow(topology_payload)

    if not sanitized_topology:
        return jsonify({
            'status': 'error',
            'message': 'topology_json 为空，或全部被标记删除。请至少保留一个有效减字组。'
        }), 400

    try:
        jianzi_sequence = build_jianzi_sequence(sanitized_topology)
        llm_result = infer_pitch_duration(sanitized_topology)
        score_model = transform_llm_result_to_score_model(llm_result, strict=False)
        music_xml = generate_musicxml(llm_result)
    except Exception as exc:
        return jsonify({
            'status': 'error',
            'message': f'Reflow failed: {exc}'
        }), 500

    return jsonify({
        'status': 'success',
        'data': {
            'topology_json': sanitized_topology,
            'jianzi_sequence': jianzi_sequence,
            'llm_result': llm_result,
            'score_model': score_model,
            'music_xml': music_xml
        }
    })

@app.route('/api/mock_pipeline', methods=['GET'])
def mock_pipeline():
    """
    返回特定假数据以便前端开发 XML 解析与渲染功能。
    使用完整的一首曲子片段（包含高低音、增时线、小节等内容）作为 Mock 数据。
    """
    mock_llm_result = [
        # 第一小节：包含四分音符、八分音符、十六分音符（减时线）以及高音点 (4拍)
        {"pitch": "1", "octave": "4", "duration": "4", "action": "勾", "string": "一", "position": "九", "finger": "大"},
        {"pitch": "2", "octave": "4", "duration": "8", "action": "抹", "string": "二", "position": "十", "finger": "中"},
        {"pitch": "3", "octave": "4", "duration": "8", "action": "历", "string": "三", "position": " ", "finger": "名"},
        {"pitch": "5", "octave": "5", "duration": "16", "action": "挑", "string": "七", "position": "七", "finger": "跪"},
        {"pitch": "6", "octave": "5", "duration": "16", "action": "轮", "string": "七", "position": "六", "finger": "名"},
        {"pitch": "1", "octave": "5", "duration": "8", "action": "勾", "string": "六", "position": "七", "finger": "大"},
        {"pitch": "2", "octave": "4", "duration": "4", "action": "托", "string": "三", "position": " ", "finger": "中"},
        
        # 第二小节：包含二分音符（增时线）和低音点 (4拍)
        {"new_measure": True, "pitch": "6", "octave": "3", "duration": "4", "action": "打", "string": "一", "position": "徽外", "finger": "大"},
        {"pitch": "1", "octave": "4", "duration": "4", "action": "摘", "string": "二", "position": " ", "finger": "中"},
        {"pitch": "5", "octave": "4", "duration": "2", "action": "托", "string": "六", "position": "八", "finger": "名"}
    ]
    
    score_model = transform_llm_result_to_score_model(mock_llm_result, strict=False)
    music_xml = generate_musicxml(mock_llm_result)

    return jsonify({
      "status": "success",
      "data": {
        "original_image_url": "/static/uploads/temp.jpg",
        "yolo_boxes": [
          {"class": "大", "bbox": [10, 20, 50, 60]}
        ],
        "topology_json": {
          "group_1": {"fingering": "勾", "finger": "大", "position": "九", "string": "一"}
        },
        "jianzi_sequence": [
          {"group_id": "group_1", "action": "勾", "finger": "大", "position": "九", "string": "一"}
        ],
        "llm_result": mock_llm_result,
        "score_model": score_model,
        "music_xml": music_xml
      }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
