
"""
Web scraping service for collecting product images.

This module provides functionalities to scrape product images from websites like
Amazon using Selenium. It includes features for searching product categories,
filtering out irrelevant images (e.g., icons, logos), and saving the
downloaded images to a specified directory.
"""

import os
import re
import time
from typing import List, Optional, Union

import pandas as pd
import requests
from selenium import webdriver
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement

def create_directory(directory: str) -> None:
    """Creates a directory if it doesn't already exist.

    Args:
        directory: The path of the directory to create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def download_image(url: str, save_path: str) -> bool:
    """Downloads and saves an image from a URL.

    Args:
        url: The URL of the image to download.
        save_path: The local path where the image will be saved.

    Returns:
        True if the download was successful, False otherwise.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"✓ Downloaded: {save_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {url}: {e}")
        return False

def is_product_image(img_url: str, img_element: Optional[WebElement] = None) -> bool:
    """Determines if an image is likely a product image and not an icon or logo.

    This function uses a series of heuristics, including URL patterns and image
    dimensions, to filter out irrelevant images.

    Args:
        img_url: The URL of the image.
        img_element: The Selenium WebElement of the image (optional).

    Returns:
        True if the image is likely a product image, False otherwise.
    """
    if not img_url or not img_url.startswith('http'):
        return False

    # Convert to lowercase for case-insensitive matching
    url_lower = img_url.lower()

    # Exclude common icon/logo patterns
    icon_patterns = [
        'icon', 'logo', 'sprite', 'button', 'arrow', 'star', 'rating',
        'badge', 'flag', 'banner', 'header', 'footer', 'nav', 'menu',
        'social', 'facebook', 'twitter', 'instagram', 'youtube',
        'amazon-logo', 'prime-logo', 'ssl-seal', 'security',
        'checkout', 'cart', 'wishlist', 'compare', 'share',
        'dropdown', 'expand', 'collapse', 'close', 'search-icon',
        'filter', 'sort', 'view', 'grid', 'list', 'thumbnail'
    ]

    # Check if URL contains any icon patterns
    for pattern in icon_patterns:
        if pattern in url_lower:
            return False

    # Exclude very small images (likely icons)
    # Look for dimension indicators in URL
    import re
    size_match = re.search(r'(\d+)x(\d+)', url_lower)
    if size_match:
        width, height = int(size_match.group(1)), int(size_match.group(2))
        # Exclude images smaller than 100x100 (likely icons)
        if width < 100 or height < 100:
            return False

    # Look for other size indicators
    small_size_patterns = ['16x16', '24x24', '32x32', '48x48', '64x64', '80x80']
    for pattern in small_size_patterns:
        if pattern in url_lower:
            return False

    # Include images that are likely product images
    product_indicators = [
        'images-na.ssl-images-amazon.com',
        'images-amazon.com',
        'm.media-amazon.com',
        'product', 'item', 'listing'
    ]

    # Must contain at least one product indicator
    has_product_indicator = any(indicator in url_lower for indicator in product_indicators)
    if not has_product_indicator:
        return False

    # Additional checks if we have the element
    if img_element:
        try:
            # Check image dimensions from element attributes
            width = img_element.get_attribute('width')
            height = img_element.get_attribute('height')

            if width and height:
                w, h = int(width), int(height)
                # Exclude very small images
                if w < 100 or h < 100:
                    return False

            # Check alt text for icon indicators
            alt_text = img_element.get_attribute('alt') or ''
            alt_lower = alt_text.lower()

            for pattern in icon_patterns:
                if pattern in alt_lower:
                    return False

        except (ValueError, TypeError):
            pass  # Ignore errors in attribute parsing

    return True

def scrape_images_for_category(
    search_term: str, 
    driver: webdriver.Chrome, 
    num_images_needed: int = 10, 
    max_images_per_page: int = 50
) -> List[str]:
    """Scrapes image URLs for a specific product category from Amazon.

    Args:
        search_term: The product category to search for.
        driver: The Selenium WebDriver instance.
        num_images_needed: The number of images to collect for the category.
        max_images_per_page: The maximum number of images to collect from a single page.

    Returns:
        A list of image URLs found for the category.
    """
    print(f"\n--- Scraping images for category: {search_term} ---")

    try:
        # Navigate to search results for this category
        search_url = f"https://www.amazon.com/s?k={search_term.replace(' ', '+')}"
        driver.get(search_url)
        print(f"✓ Navigated to search results: {search_url}")
        time.sleep(5)  # Wait for results to load

        # Find all product images using our improved selectors
        image_selectors = [
            "//img[@class='s-image']",  # Amazon's main product image class
            "//img[contains(@class, 'product-image')]",
            "//img[contains(@data-image-latency, 's-product-image')]",
            "//img[contains(@class, 's-image') and not(contains(@class, 'icon'))]",  # Exclude icon classes
            "//img[contains(@src, 'images-na.ssl-images-amazon.com') and not(contains(@src, 'icon')) and not(contains(@src, 'logo'))]",  # Exclude icon/logo URLs
            "//img[contains(@alt, 'product') and not(contains(@alt, 'icon')) and not(contains(@alt, 'logo'))]"  # Product alt text but not icons
        ]

        image_elements = []
        for selector in image_selectors:
            try:
                elements = driver.find_elements(By.XPATH, selector)
                if elements:
                    # Apply additional filtering to remove icons
                    filtered_elements = []
                    for element in elements:
                        img_url = element.get_attribute('src')
                        if is_product_image(img_url, element):
                            filtered_elements.append(element)

                    if filtered_elements:
                        image_elements = filtered_elements
                        print(f"✓ Found {len(filtered_elements)} product images with selector: {selector}")
                        break
            except Exception:
                continue

        if not image_elements:
            # Fallback: get all images and filter
            all_images = driver.find_elements(By.TAG_NAME, "img")
            # Apply filter for Amazon product images
            amazon_images = [img for img in all_images if img.get_attribute('src') and 'amazon' in img.get_attribute('src').lower()]

            # Filter to exclude icons using our filtering function
            image_elements = []
            for img in amazon_images:
                img_url = img.get_attribute('src')
                if is_product_image(img_url, img):
                    image_elements.append(img)

            print(f"✓ Found {len(image_elements)} product images using fallback method")

        # Extract image URLs with improved filtering
        image_urls = []
        for element in image_elements:
            try:
                img_url = element.get_attribute('src')
                # Use our improved filtering function
                if img_url and is_product_image(img_url, element):
                    image_urls.append(img_url)

                    # Stop when we have enough images for this category
                    if len(image_urls) >= num_images_needed:
                        break
            except StaleElementReferenceException:
                continue

        print(f"✓ Extracted {len(image_urls)} valid image URLs for {search_term}")
        return image_urls

    except Exception as e:
        print(f"✗ Error scraping images for {search_term}: {e}")
        return []

def scrape_images_with_selenium(
    search_terms: Union[str, List[str], None] = None, 
    output_dir: str = "data/scraped_images", 
    start_stock_code: int = 20000, 
    num_products: int = 10
) -> int:
    """Scrapes images from Amazon for multiple product categories using Selenium.

    This is the main function that orchestrates the web scraping process. It sets
    up the WebDriver, iterates through the search terms, downloads the images,
    and saves the metadata to a CSV file.

    Args:
        search_terms: A list of product categories to search for.
        output_dir: The directory where the scraped images will be saved.
        start_stock_code: The starting stock code for the products.
        num_products: The total number of products to scrape.

    Returns:
        The total number of images that were successfully downloaded.
    """
    # Handle both single search term and multiple search terms
    if search_terms is None:
        search_terms = ["laptop", "smartphone", "headphones", "tablet", "camera"]
    elif isinstance(search_terms, str):
        search_terms = [search_terms]

    print(f"Scraping images for categories: {', '.join(search_terms)}")

    # Calculate how many images to get per category for better distribution
    images_per_category = max(1, num_products // len(search_terms))
    remaining_images = num_products % len(search_terms)

    print(f"Distribution: {images_per_category} images per category")
    print(f"Extra images: {remaining_images} (distributed to first {remaining_images} categories)")
    print(f"Final range: {images_per_category} to {images_per_category + 1} images per category")

    # Create output directory
    create_directory(output_dir)
    create_directory("data/dataset")
    
    # Setup Chrome WebDriver
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in background
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    try:
        # Initialize WebDriver
        driver = webdriver.Chrome(options=chrome_options)
        print("✓ Chrome WebDriver initialized")

        # Navigate to Amazon
        url = "https://www.amazon.com/"
        driver.get(url)
        print(f"✓ Navigated to: {url}")
        time.sleep(3)

        # Collect images from all categories
        all_image_urls = []
        category_image_counts = {}

        for i, search_term in enumerate(search_terms):
            # Calculate how many images to get for this category
            target_images = images_per_category
            if i < remaining_images:  # Distribute remaining images to first few categories
                target_images += 1

            print(f"\n=== Processing category {i+1}/{len(search_terms)}: {search_term} (target: {target_images} images) ===")

            # Get images for this category
            category_images = scrape_images_for_category(search_term, driver, target_images)

            # Add category info to each image URL
            for img_url in category_images:
                all_image_urls.append({
                    'url': img_url,
                    'category': search_term
                })

            category_image_counts[search_term] = len(category_images)
            print(f"✓ Collected {len(category_images)} images for {search_term}")

            # Small delay between categories
            time.sleep(2)

        # Print summary of collected images
        print(f"\n=== IMAGE COLLECTION SUMMARY ===")
        total_collected = len(all_image_urls)
        print(f"Total images collected: {total_collected}")
        for category, count in category_image_counts.items():
            print(f"  {category}: {count} images")

        # Close WebDriver
        driver.quit()
        print("✓ WebDriver closed")

        # Check if we have enough images
        total_collected = len(all_image_urls)
        print(f"Total images collected from scraping: {total_collected}")

        if total_collected < num_products:
            print(f"⚠️  Only collected {total_collected} images, but need {num_products}")
            print("Consider:")
            print("1. Running the script multiple times to collect more images")
            print("2. Adding more product categories")
            print("3. Reducing the target number of products")
            print(f"Proceeding with {total_collected} images...")
        
        # Prepare CSV data
        csv_data = []
        total_downloaded = 0
        start_time = time.time()
        
        # Track images per category for naming
        category_counters = {}

        # Download images with new folder structure
        for i in range(min(num_products, len(all_image_urls))):
            stock_code = start_stock_code + i

            # Get image URL and category info
            img_data = all_image_urls[i]
            img_url = img_data['url']
            category = img_data['category']

            # Initialize counter for this category if not exists
            if category not in category_counters:
                category_counters[category] = 0

            category_counters[category] += 1

            print(f"Processing stock code {stock_code} with {category} image #{category_counters[category]}")

            # Create directory for this category (new format)
            safe_category_dir = category.replace(' ', '_').replace('/', '_').replace('&', 'and')
            category_dir = os.path.join(output_dir, safe_category_dir)
            create_directory(category_dir)

            # Create filename in new format: category_name_001.jpg
            safe_category_name = category.replace(' ', '_').replace('/', '_').replace('&', 'and')
            filename = f"{safe_category_name}_{category_counters[category]:03d}.jpg"
            save_path = os.path.join(category_dir, filename)

            # Download image
            if download_image(img_url, save_path):
                relative_path = os.path.relpath(save_path, start=os.getcwd())
                csv_data.append({
                    'stockcode': str(stock_code),
                    'image': relative_path,
                    'description': category
                })
                total_downloaded += 1
                print(f"✓ Downloaded: {relative_path}")
            else:
                # Add empty entry if download failed
                csv_data.append({
                    'stockcode': str(stock_code),
                    'image': '',
                    'description': category
                })
                print(f"✗ Failed to download image for {stock_code}")

            # Small delay to be respectful
            time.sleep(0.5)

            # Progress update every 25 products for 1K dataset
            if (i + 1) % 25 == 0:
                progress_pct = ((i + 1)/min(num_products, len(all_image_urls)))*100
                print(f"Progress: {i + 1}/{min(num_products, len(all_image_urls))} products processed ({progress_pct:.1f}%)")

            # Save progress every 200 products to avoid data loss
            if (i + 1) % 200 == 0:
                temp_df = pd.DataFrame(csv_data)
                temp_csv_path = f"data/dataset/CNN_Model_Train_Data_backup_{i+1}.csv"
                temp_df.to_csv(temp_csv_path, index=False)
                print(f"✓ Backup saved: {temp_csv_path}")
        
        # Save to CSV
        df = pd.DataFrame(csv_data)
        csv_path = "data/dataset/cnn_model_train.csv"
        df.to_csv(csv_path, index=False)
        
        actual_products_processed = len(csv_data)
        end_time = time.time()
        total_time = end_time - start_time

        print(f"\n=== SCRAPING COMPLETED ===")
        print(f"Categories processed: {len(search_terms)} categories")
        print(f"Total products processed: {actual_products_processed:,}")
        print(f"Successful downloads: {total_downloaded:,}")
        print(f"Failed downloads: {actual_products_processed - total_downloaded:,}")
        if actual_products_processed > 0:
            print(f"Success rate: {(total_downloaded/actual_products_processed)*100:.1f}%")
        print(f"Total time: {total_time/3600:.1f} hours ({total_time/60:.1f} minutes)")
        if total_downloaded > 0:
            print(f"Average time per image: {total_time/total_downloaded:.1f} seconds")
        print(f"CSV saved to: {csv_path}")
        print(f"Images saved to: {output_dir}")

        # Show folder structure
        print(f"\n📁 Folder Structure:")
        print(f"data/scraped_images/")
        if 'description' in df.columns:
            category_counts = df['description'].value_counts()
            for category, count in category_counts.items():
                safe_category = category.replace(' ', '_').replace('/', '_').replace('&', 'and')
                print(f"├── {safe_category}/ ({count} images)")
                if count > 0:
                    print(f"│   ├── {safe_category}_001.jpg")
                    print(f"│   ├── {safe_category}_002.jpg")
                    if count > 2:
                        print(f"│   └── ... ({count} total)")

        # Show sample data
        print(f"\nSample CSV data:")
        print(df.head().to_string(index=False))
        
        return total_downloaded
        
    except Exception as e:
        print(f"✗ Error with WebDriver: {e}")
        return []

if __name__ == "__main__":
    # Configuration optimized for assessment timeline
    START_STOCK_CODE = 20000
    NUM_PRODUCTS = 1500  # 1,500 images - balanced for assessment timeline

    # Dataset-aligned categories: Only categories that exist in your actual dataset
    # Reduced to 14 categories - all verified to be present in cleaned_dataset.csv
    PRODUCT_CATEGORIES = [
        # === CORE DATASET CATEGORIES (High visual distinction) ===
        "candle", "heart decoration", "flower", "mug", "clock",
        "bottle", "toy", "bag", "hat",

        # === ELECTRONICS (From dataset) ===
        "headphones",

        # === HOME & KITCHEN (From dataset) ===
        "coffee maker", "microwave",

        # === SPORTS & HOME (From dataset) ===
        "yoga mat",

        # === SEASONAL (From dataset) ===
        "christmas decoration"
    ]

    print("=" * 80)
    print("🚀 DATASET-ALIGNED WEB SCRAPING FOR CNN TRAINING")
    print("=" * 80)
    print(f"Target images: {NUM_PRODUCTS:,} (optimized for 5-hour timeline)")
    print(f"Stock codes: {START_STOCK_CODE:,} - {START_STOCK_CODE + NUM_PRODUCTS - 1:,}")
    print(f"Categories: {len(PRODUCT_CATEGORIES)} categories (100% dataset-aligned)")

    # Calculate distribution
    base_per_category = NUM_PRODUCTS // len(PRODUCT_CATEGORIES)
    extra_images = NUM_PRODUCTS % len(PRODUCT_CATEGORIES)

    print(f"Distribution: {base_per_category}-{base_per_category + 1} images per category")
    print(f"  • {len(PRODUCT_CATEGORIES) - extra_images} categories get {base_per_category} images")
    print(f"  • {extra_images} categories get {base_per_category + 1} images")
    print("=" * 80)
    print("📊 Dataset-Aligned Categories (100% match):")
    print("   � High visual distinction for effective CNN training")
    print("   ⚡ Optimized for assessment timeline")
    print("   📈 ~50 images per category for balanced training")
    print("=" * 80)
    print("Selected categories:")
    for i, category in enumerate(PRODUCT_CATEGORIES, 1):
        target_for_this = base_per_category + (1 if i <= extra_images else 0)
        print(f"  {i:2d}. {category} ({target_for_this} images)")
    print("=" * 80)

    # Confirm before starting scraping operation
    print("⚡ DATASET-ALIGNED: This will scrape 1,500 images in ~45-60 minutes.")
    print("🎯 100% dataset match: Perfect alignment with your actual product data.")
    print("🔄 Progress tracking every 25 images, backups every 200 images.")
    print(f"📊 Each category gets ~{base_per_category} images - excellent for CNN training.")
    print("📁 Folder structure: data/scraped_images/category_name/category_name_001.jpg")
    print("📋 CSV format: stockcode,image,description")
    print("=" * 80)

    # Run the scraping with multiple categories
    scrape_images_with_selenium(
        search_terms=PRODUCT_CATEGORIES,
        output_dir="data/scraped_images",
        start_stock_code=START_STOCK_CODE,
        num_products=NUM_PRODUCTS
    )