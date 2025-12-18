#!/usr/bin/env python3
"""Automated UI Demo - controls browser automatically."""

import time
from playwright.sync_api import sync_playwright

def main():
    with sync_playwright() as p:
        # Launch browser (visible for recording)
        browser = p.chromium.launch(headless=False, slow_mo=100)
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        
        print("Opening Streamlit...")
        page.goto("http://localhost:8501")
        time.sleep(3)
        
        # Step 1: Load SEC Samples
        print("Step 1: Loading SEC samples...")
        page.click("text=📚 Load SEC Samples")
        time.sleep(5)  # Wait for ingestion
        
        # Step 2: First query
        print("Step 2: First query...")
        page.fill("textarea", "What are the main risk factors disclosed by these companies?")
        time.sleep(1)
        page.click("button:has-text('Ask')")
        time.sleep(8)  # Wait for response
        
        # Step 3: Expand sources
        print("Step 3: Expanding sources...")
        try:
            page.click("text=📚 Sources")
            time.sleep(2)
        except:
            pass
        
        # Step 4: Filter to Tesla
        print("Step 4: Filter to Tesla...")
        page.click("text=Filter by Company")
        time.sleep(1)
        page.click("text=Tesla")
        time.sleep(1)
        
        # Step 5: Tesla query
        print("Step 5: Tesla query...")
        page.fill("textarea", "What are manufacturing and supply chain risks?")
        time.sleep(1)
        page.click("button:has-text('Ask')")
        time.sleep(8)
        
        # Step 6: Clear filter, go to Compare tab
        print("Step 6: Compare tab...")
        page.click("text=📊 Compare Companies")
        time.sleep(2)
        
        # Step 7: Select all companies
        print("Step 7: Selecting companies...")
        page.click("text=Select companies to compare")
        time.sleep(1)
        page.click("text=Meta")
        page.click("text=Tesla")  
        page.click("text=NVIDIA")
        time.sleep(1)
        
        # Step 8: Compare query
        print("Step 8: Compare query...")
        compare_input = page.locator("textarea").nth(0)
        compare_input.fill("Compare AI and machine learning strategies across these companies")
        time.sleep(1)
        page.click("button:has-text('Compare')")
        time.sleep(10)
        
        # Step 9: Show registry
        print("Step 9: Registry tab...")
        page.click("text=📋 Registry")
        time.sleep(3)
        
        print("\n✅ UI Demo complete!")
        print("Press Enter to close browser...")
        input()
        
        browser.close()

if __name__ == "__main__":
    main()
