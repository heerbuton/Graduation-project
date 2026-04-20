from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from pathlib import Path
import time, json

base = Path(r'F:\AIcharacter\End\thesis_materials_2026-04-18\paper_ui_figures_CD_2026-04-18')
raw = base / 'raw_screenshots' / 'timeline_probe'
raw.mkdir(parents=True, exist_ok=True)
img = Path(r'F:\AIcharacter\End\test\testpicture-1.jpg').resolve()

opts = Options()
opts.add_argument('--headless=new')
opts.add_argument('--window-size=1920,1080')
opts.add_argument('--disable-gpu')
opts.add_argument('--no-sandbox')
opts.add_argument('--disable-dev-shm-usage')

d = webdriver.Chrome(options=opts)
records=[]
try:
    d.get('http://localhost:3001')
    d.implicitly_wait(10)
    d.find_element(By.CSS_SELECTOR,'input[type="file"]').send_keys(str(img))
    time.sleep(1.5)
    d.save_screenshot(str(raw/'t00_upload_preview.png'))

    # click start
    start_btn = None
    for b in d.find_elements(By.TAG_NAME,'button'):
        if '开始打谱' in (b.text or ''):
            start_btn=b
            break
    if not start_btn:
        raise RuntimeError('start button not found')
    start_btn.click()

    t0 = time.time()
    checkpoints = [2,5,8,12,16,20,25,30,35,40,45,50,55,60,70,80,90,100,110,120]
    for sec in checkpoints:
        while time.time()-t0 < sec:
            time.sleep(0.2)
        png = raw / f't{sec:03d}.png'
        d.save_screenshot(str(png))
        txt = d.find_element(By.TAG_NAME,'body').text
        btns = [b.text.strip().replace('\n',' | ') for b in d.find_elements(By.TAG_NAME,'button') if b.text.strip()]
        records.append({'t':sec,'buttons':btns,'text_head':txt[:1000]})

    # if has next-step button click once then keep capture
    next_clicked=False
    for b in d.find_elements(By.TAG_NAME,'button'):
        if '下一步' in (b.text or ''):
            b.click(); next_clicked=True; break
    if next_clicked:
        time.sleep(2)
        d.save_screenshot(str(raw/'t_after_next1.png'))
        btns = [b.text.strip().replace('\n',' | ') for b in d.find_elements(By.TAG_NAME,'button') if b.text.strip()]
        txt = d.find_element(By.TAG_NAME,'body').text
        records.append({'t':'after_next1','buttons':btns,'text_head':txt[:1200]})

    for b in d.find_elements(By.TAG_NAME,'button'):
        if '下一步' in (b.text or ''):
            b.click(); time.sleep(3); d.save_screenshot(str(raw/'t_after_next2.png')); 
            btns = [bb.text.strip().replace('\n',' | ') for bb in d.find_elements(By.TAG_NAME,'button') if bb.text.strip()]
            txt = d.find_element(By.TAG_NAME,'body').text
            records.append({'t':'after_next2','buttons':btns,'text_head':txt[:1200]})
            break

finally:
    d.quit()

(base/'logs'/'timeline_probe.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
print('wrote', base/'logs'/'timeline_probe.json')
