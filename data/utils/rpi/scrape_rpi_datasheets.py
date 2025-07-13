import json
import os
import time

import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

__author__ = "Vojtěch Sýkora"
__email__ = "sykoravojtech01@gmail.com"


def scrape_rpi_pdfs(
    save_path="data/rpi_datasheets/rpi_datasheets.json", filter_phrase=None
):
    BASE_URL = "https://datasheets.raspberrypi.com/"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Set up Selenium WebDriver
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(service=Service(), options=chrome_options)
    driver.get(BASE_URL)

    time.sleep(3)  # Wait for JavaScript to load

    pdf_links = []

    # Locate the dynamically loaded PDF links
    pdf_elements = driver.find_elements(By.CSS_SELECTOR, "ul.datasheets a")

    for element in pdf_elements:
        href = element.get_attribute("href").strip()
        if href.endswith(".pdf"):  # Only save .pdf files
            pdf_name = href.replace(BASE_URL, "")

            # Apply filter if a phrase is provided
            if filter_phrase and filter_phrase.lower() not in pdf_name.lower():
                continue

            pdf_links.append({"name": pdf_name, "url": href})

    driver.quit()

    with open(save_path, "w", encoding="utf-8") as json_file:
        json.dump(pdf_links, json_file, indent=4)

    print(f"Scraped {len(pdf_links)} PDFs. Data saved to {save_path}")


def download_pdfs(
    json_path="data/rpi_datasheets/rpi_datasheets.json",
    download_dir="data/rpi_datasheets/pdf",
):
    os.makedirs(download_dir, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as json_file:
        pdf_links = json.load(json_file)

    for pdf in pdf_links:
        pdf_name = pdf["name"].replace("/", "_")  # Replace slashes for valid file names
        pdf_url = pdf["url"]
        pdf_save_path = os.path.join(download_dir, pdf_name)

        if os.path.exists(pdf_save_path):
            print(f"Skipping {pdf_name}, already downloaded.")
            continue

        print(f"Downloading {pdf_name}...")
        response = requests.get(pdf_url, stream=True)
        if response.status_code == 200:
            with open(pdf_save_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
            print(f"Saved {pdf_name}.")
        else:
            print(f"Failed to download {pdf_name}.")


if __name__ == "__main__":
    filter_phrase = "pico"  # Change this to filter results (e.g., "pico") or None
    save_path = f"data/rpi_datasheets/rpi_datasheets{'_'+filter_phrase if filter_phrase else ''}.json"
    scrape_rpi_pdfs(save_path=save_path, filter_phrase=filter_phrase)
    download_pdfs(
        json_path=save_path,
        download_dir=os.path.join(
            os.path.dirname(save_path),
            f"pdf{'_'+filter_phrase if filter_phrase else ''}",
        ),
    )
