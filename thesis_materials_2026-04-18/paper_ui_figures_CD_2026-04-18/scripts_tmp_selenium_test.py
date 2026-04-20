from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from pathlib import Path

out = Path(r"F:\AIcharacter\End\thesis_materials_2026-04-18\paper_ui_figures_CD_2026-04-18\raw_screenshots\sanity_home.png")
out.parent.mkdir(parents=True, exist_ok=True)

opts = Options()
opts.add_argument('--headless=new')
opts.add_argument('--window-size=1920,1080')
opts.add_argument('--disable-gpu')
opts.add_argument('--no-sandbox')
opts.add_argument('--disable-dev-shm-usage')

driver = webdriver.Chrome(options=opts)
try:
    driver.get('http://localhost:3001')
    driver.implicitly_wait(10)
    driver.save_screenshot(str(out))
    print('OK', out)
finally:
    driver.quit()
