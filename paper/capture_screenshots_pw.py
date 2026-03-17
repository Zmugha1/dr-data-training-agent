from playwright.sync_api import sync_playwright
import time

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={
        "width": 1600, "height": 900})
    page.goto("http://localhost:8501")
    page.wait_for_timeout(8000)

    page.screenshot(
        path="paper/fig_A2a_summary_viz.png",
        full_page=False)
    print("fig_A2a saved")

    page.evaluate("window.scrollBy(0, 900)")
    page.wait_for_timeout(2000)
    page.screenshot(
        path="paper/fig_A2b_incident_drivers.png")
    print("fig_A2b saved")

    page.evaluate("window.scrollBy(0, 1200)")
    page.wait_for_timeout(2000)
    page.screenshot(
        path="paper/fig_A2c_intervention_table.png")
    print("fig_A2c saved")

    page.evaluate("window.scrollBy(0, 1500)")
    page.wait_for_timeout(2000)
    page.screenshot(
        path="paper/fig_A2d_plan_output.png")
    print("fig_A2d saved")

    browser.close()
