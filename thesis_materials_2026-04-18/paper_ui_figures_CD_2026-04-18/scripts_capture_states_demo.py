from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from pathlib import Path
import time, json

base = Path(r'F:\AIcharacter\End\thesis_materials_2026-04-18\paper_ui_figures_CD_2026-04-18')
raw = base / 'raw_screenshots' / 'paper_states'
raw.mkdir(parents=True, exist_ok=True)
log=[]
img = Path(r'F:\AIcharacter\End\test\testpicture-1.jpg').resolve()

opts = Options()
opts.add_argument('--headless=new')
opts.add_argument('--window-size=1920,1080')
opts.add_argument('--disable-gpu')
opts.add_argument('--no-sandbox')
opts.add_argument('--disable-dev-shm-usage')

d = webdriver.Chrome(options=opts)

def shot(name, note=''):
    path = raw / name
    d.save_screenshot(str(path))
    body = d.find_element(By.TAG_NAME,'body').text
    buttons = [b.text.strip().replace('\n',' | ') for b in d.find_elements(By.TAG_NAME,'button') if b.text.strip()]
    log.append({'file':str(path),'note':note,'buttons':buttons,'body_head':body[:1200]})


def click_button_contains(text):
    for b in d.find_elements(By.TAG_NAME,'button'):
        if text in (b.text or ''):
            try:
                b.click()
                return True
            except Exception:
                pass
    return False


def click_text_node(text):
    for el in d.find_elements(By.XPATH, f"//*[contains(text(),'{text}')]"):
        if el.is_displayed():
            try:
                el.click()
                return True
            except Exception:
                pass
    return False

try:
    d.get('http://localhost:3001')
    d.implicitly_wait(10)

    # state 1: upload preview + trigger
    d.find_element(By.CSS_SELECTOR,'input[type="file"]').send_keys(str(img))
    time.sleep(1.5)
    shot('S01_upload_preview_start.png', '上传后预览 + 开始打谱按钮')

    # use demo for fast full-flow
    click_button_contains('Demo')
    time.sleep(4)
    shot('S08_final_render_initial.png', 'Demo完成后默认终态（打谱渲染）')

    # stage 1: feature extraction + group overlays
    click_text_node('特征提取')
    time.sleep(1.5)
    shot('S02_feature_extraction.png', '特征提取状态：原图+交互校对视图')

    # hover tooltip on group
    group_btn = None
    for b in d.find_elements(By.TAG_NAME,'button'):
        t = (b.text or '').strip()
        if t.startswith('group_'):
            group_btn = b
            break
    if group_btn is not None:
        webdriver.ActionChains(d).move_to_element(group_btn).perform()
        time.sleep(1.0)
        shot('S03_hover_explain.png', '悬停字段提示框')

        # click open edit panel
        group_btn.click()
        time.sleep(1.2)
        shot('S04_edit_panel_open.png', '编辑面板打开')

        # change first visible input a little then save
        edited=False
        for inp in d.find_elements(By.CSS_SELECTOR, 'input'):
            if not inp.is_displayed():
                continue
            t = (inp.get_attribute('type') or '').lower()
            if t in ('text','search',''):
                try:
                    old = inp.get_attribute('value') or ''
                    inp.clear()
                    inp.send_keys(old if old else '-')
                    edited=True
                    break
                except Exception:
                    pass

        click_button_contains('保存字段')
        time.sleep(0.9)
        shot('S05_apply_feedback.png', '保存字段后反馈')

        # close panel if visible
        click_button_contains('关闭')
        time.sleep(0.6)

    # stage 2 topology sequence
    click_text_node('拓扑序列')
    time.sleep(1.3)
    shot('S06_topology_sequence.png', '拓扑序列状态')

    # stage 3 llm inference cards
    click_text_node('乐理推理')
    time.sleep(1.3)
    shot('S07_llm_inference.png', '乐理推理结果状态')

    # stage 4 final render
    click_text_node('打谱渲染')
    time.sleep(1.3)
    shot('S08_final_render.png', '最终渲染状态')

finally:
    d.quit()

(base/'logs'/'paper_states_capture_log.json').write_text(json.dumps(log,ensure_ascii=False,indent=2),encoding='utf-8')
print('captured', len(log), 'screens')
