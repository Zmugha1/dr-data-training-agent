from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from PIL import Image
import time
import os

BASE_URL = "http://localhost:8501"
OUTPUT_DIR = "paper"
os.makedirs(OUTPUT_DIR, exist_ok=True)

opts = Options()
opts.add_argument("--headless")
opts.add_argument("--no-sandbox")
opts.add_argument("--disable-dev-shm-usage")
opts.add_argument("--window-size=1600,900")
opts.add_argument("--force-device-scale-factor=1")

driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=opts
)

wait = WebDriverWait(driver, 30)

def wait_for_streamlit():
    time.sleep(6)

def scroll_to_element(driver, element):
    driver.execute_script(
        "arguments[0].scrollIntoView(true);", element)
    time.sleep(1)

try:
    print("Loading Streamlit app...")
    driver.get(BASE_URL)
    wait_for_streamlit()

    print("Triggering pipeline run with 60 analysts...")
    driver.get(BASE_URL)
    wait_for_streamlit()

    print("Capturing fig_A2a: Summary Visualization...")
    driver.set_window_size(1600, 1200)
    time.sleep(2)

    try:
        summary_header = driver.find_element(
            By.XPATH,
            "//*[contains(text(), 'Summary visualization')]"
        )
        scroll_to_element(driver, summary_header)
        time.sleep(2)
    except:
        pass

    driver.save_screenshot(
        f"{OUTPUT_DIR}/fig_A2a_summary_viz_full.png")

    img = Image.open(
        f"{OUTPUT_DIR}/fig_A2a_summary_viz_full.png")
    w, h = img.size
    cropped = img.crop((280, 60, w, min(900, h)))
    cropped.save(f"{OUTPUT_DIR}/fig_A2a_summary_viz.png",
                 dpi=(150, 150))
    print("  fig_A2a_summary_viz.png saved")

    print("Scrolling down for Incident Risk drivers...")
    driver.execute_script("window.scrollBy(0, 900)")
    time.sleep(2)

    driver.save_screenshot(
        f"{OUTPUT_DIR}/fig_A2b_incident_full.png")
    img = Image.open(
        f"{OUTPUT_DIR}/fig_A2b_incident_full.png")
    w, h = img.size
    cropped = img.crop((280, 0, w, min(500, h)))
    cropped.save(f"{OUTPUT_DIR}/fig_A2b_incident_drivers.png",
                 dpi=(150, 150))
    print("  fig_A2b_incident_drivers.png saved")

    print("Scrolling down for intervention table...")
    driver.execute_script("window.scrollBy(0, 1200)")
    time.sleep(2)
    driver.save_screenshot(
        f"{OUTPUT_DIR}/fig_A2c_table_full.png")
    img = Image.open(
        f"{OUTPUT_DIR}/fig_A2c_table_full.png")
    w, h = img.size
    cropped = img.crop((280, 0, w, min(600, h)))
    cropped.save(f"{OUTPUT_DIR}/fig_A2c_intervention_table.png",
                 dpi=(150, 150))
    print("  fig_A2c_intervention_table.png saved")

    print("Scrolling down for targeted plan output...")
    driver.execute_script("window.scrollBy(0, 1500)")
    time.sleep(2)
    driver.save_screenshot(
        f"{OUTPUT_DIR}/fig_A2d_plan_full.png")
    img = Image.open(
        f"{OUTPUT_DIR}/fig_A2d_plan_full.png")
    w, h = img.size
    cropped = img.crop((280, 0, w, min(800, h)))
    cropped.save(f"{OUTPUT_DIR}/fig_A2d_plan_output.png",
                 dpi=(150, 150))
    print("  fig_A2d_plan_output.png saved")

    print("\nAll screenshots captured successfully.")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    driver.quit()

    for tmp in ["fig_A2a_summary_viz_full.png",
                "fig_A2b_incident_full.png",
                "fig_A2c_table_full.png",
                "fig_A2d_plan_full.png"]:
        try:
            os.remove(f"{OUTPUT_DIR}/{tmp}")
        except:
            pass

print("\nFinal figure inventory:")
import os
for f in sorted(os.listdir("paper")):
    if f.startswith("fig_"):
        size = os.path.getsize(f"paper/{f}")
        print(f"  {f}: {size:,} bytes")
