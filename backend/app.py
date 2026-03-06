import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from pipeline.cv_module import detect_components
from pipeline.topology_module import build_topology
from pipeline.llm_module import infer_pitch_duration
from pipeline.musicxml_encoder import generate_musicxml

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 配置上传目录
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 强制输出 utf-8 编码
app.config['JSON_AS_ASCII'] = False

@app.route('/static/uploads/<filename>')
def serve_upload(filename):
    """提供上传图片的静态访问路径"""
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    处理前端图片上传。
    规约：必须采用 cv2.imdecode 配合 numpy.fromfile 处理可能包含中文字符的路径，绝不使用直接 read 导致 GBK 乱码。
    """
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400

    if file:
        filename = file.filename
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        
        # 为了防止包含中文路径时 cv2 报错或乱码，我们直接将 request 中的流转为 numpy 数组并 imdecode
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
             return jsonify({'status': 'error', 'message': 'Invalid image file'}), 400
        
        # 安全落盘: 强制 utf-8 并且支持多语言路径
        # cv2.imencode('.jpg', img)[1].tofile(save_path) 是处理中文路径保存的标准做法
        extension = os.path.splitext(filename)[1].lower() or '.jpg'
        is_success, buffer = cv2.imencode(extension, img)
        if is_success:
            buffer.tofile(save_path)
        else:
            return jsonify({'status': 'error', 'message': 'Failed to save image'}), 500

        try:
            # 流水线执行
            # 1. 视觉感知
            yolo_boxes = detect_components(save_path)
            
            # 2. 空间拓扑解析
            topology_json = build_topology(yolo_boxes)
            
            # 3. 大模型打谱推理 (音高与节奏)
            llm_result = infer_pitch_duration(topology_json)
            
            # 4. XML 编码层
            music_xml = generate_musicxml(llm_result)
        except Exception as exc:
            return jsonify({
                'status': 'error',
                'message': f'Pipeline execution failed: {exc}'
            }), 500

        # 我们可以将最终结果落盘为 JSON 文件 (强制 utf-8-sig)
        result_json_path = save_path + "_result.json"
        import json
        with open(result_json_path, 'w', encoding='utf-8-sig') as f:
            json.dump({
                "yolo_boxes": yolo_boxes,
                "topology_json": topology_json,
                "llm_result": llm_result,
                "music_xml": music_xml
            }, f, ensure_ascii=False, indent=2)

        return jsonify({
            'status': 'success',
            'data': {
                'original_image_url': f'/static/uploads/{filename}',
                'yolo_boxes': yolo_boxes,
                'topology_json': topology_json,
                'llm_result': llm_result,
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
        "llm_result": mock_llm_result,
        "music_xml": music_xml
      }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
