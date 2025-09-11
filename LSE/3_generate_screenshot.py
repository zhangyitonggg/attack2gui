import json
import random
from pathlib import Path
from playwright.sync_api import sync_playwright
from datetime import datetime, timedelta
import os

# 生成的商品数量
NUM_PRODUCTS = 6  # 6个需要替换的商品图片位置

# HTML页面模板（完全复制自amazon.html）
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Amazon.com: Small space solutions: living</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: "Amazon Ember", Arial, sans-serif;
            background-color: #fff;
            color: #0F1111;
        }}

        /* Header Styles */
        .header {{
            background-color: #232F3E;
            padding: 8px 0;
            position: relative;
        }}

        .header-content {{
            max-width: 1500px;
            margin: 0 auto;
            display: flex;
            align-items: center;
            padding: 0 15px;
        }}

        .logo {{
            color: white;
            font-size: 24px;
            font-weight: bold;
            margin-right: 20px;
            text-decoration: none;
        }}

        .deliver-to {{
            color: #ccc;
            font-size: 12px;
            margin-right: 20px;
        }}

        .deliver-to .location {{
            color: white;
            font-weight: bold;
        }}

        .search-container {{
            flex: 1;
            display: flex;
            max-width: 600px;
            margin: 0 20px;
        }}

        .search-dropdown {{
            background-color: #f3f3f3;
            border: none;
            padding: 10px;
            border-radius: 4px 0 0 4px;
            font-size: 12px;
        }}

        .search-input {{
            flex: 1;
            padding: 10px;
            border: none;
            font-size: 16px;
        }}

        .search-button {{
            background-color: #febd69;
            border: none;
            padding: 10px 15px;
            border-radius: 0 4px 4px 0;
            cursor: pointer;
        }}

        .header-right {{
            display: flex;
            align-items: center;
            gap: 20px;
        }}

        .header-link {{
            color: white;
            text-decoration: none;
            font-size: 14px;
        }}

        .cart {{
            position: relative;
        }}

        .cart-count {{
            position: absolute;
            top: -8px;
            right: -8px;
            background-color: #ff9900;
            color: white;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            font-weight: bold;
        }}

        /* Navigation Bar */
        .nav-bar {{
            background-color: #37475A;
            padding: 8px 0;
        }}

        .nav-content {{
            max-width: 1500px;
            margin: 0 auto;
            display: flex;
            align-items: center;
            padding: 0 15px;
        }}

        .nav-link {{
            color: white;
            text-decoration: none;
            padding: 8px 12px;
            font-size: 14px;
            border-radius: 2px;
        }}

        .nav-link:hover {{
            background-color: rgba(255,255,255,0.1);
        }}

        /* Main Content */
        .main-container {{
            max-width: 1500px;
            margin: 0 auto;
            display: flex;
            padding: 20px 15px;
            gap: 20px;
        }}

        /* Sidebar */
        .sidebar {{
            width: 240px;
            flex-shrink: 0;
        }}

        .filter-section {{
            margin-bottom: 20px;
        }}

        .filter-title {{
            font-weight: bold;
            font-size: 16px;
            margin-bottom: 10px;
            color: #0F1111;
        }}

        .filter-item {{
            margin-bottom: 8px;
            font-size: 13px;
        }}

        .filter-item a {{
            color: #007185;
            text-decoration: none;
        }}

        .filter-item a:hover {{
            text-decoration: underline;
        }}

        .checkbox-item {{
            display: flex;
            align-items: center;
            margin-bottom: 8px;
        }}

        .checkbox-item input {{
            margin-right: 8px;
        }}

        .stars {{
            color: #ffa500;
            margin-right: 5px;
        }}

        /* Main Content Area */
        .content {{
            flex: 1;
        }}

        .results-header {{
            margin-bottom: 20px;
            font-size: 16px;
        }}

        .results-count {{
            color: #565959;
        }}

        .search-term {{
            color: #C7511F;
        }}

        /* Product Grid */
        .product-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}

        .product-card {{
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            background-color: white;
            position: relative;
        }}

        .best-seller {{
            position: absolute;
            top: 10px;
            left: 10px;
            background-color: #ff9900;
            color: white;
            padding: 4px 8px;
            font-size: 12px;
            border-radius: 3px;
            font-weight: bold;
        }}

        .product-image {{
            width: 100%;
            height: 200px;
            object-fit: cover;
            margin-bottom: 10px;
            border-radius: 4px;
        }}

        .product-title {{
            font-size: 16px;
            font-weight: normal;
            margin-bottom: 8px;
            line-height: 1.3;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        .product-title a {{
            color: #0F1111;
            text-decoration: none;
        }}

        .product-title a:hover {{
            color: #C7511F;
        }}

        .product-rating {{
            display: flex;
            align-items: center;
            margin-bottom: 8px;
        }}

        .rating-stars {{
            color: #ffa500;
            margin-right: 5px;
        }}

        .rating-count {{
            color: #007185;
            font-size: 13px;
        }}

        .purchase-info {{
            font-size: 13px;
            color: #565959;
            margin-bottom: 8px;
        }}

        .product-price {{
            font-size: 18px;
            font-weight: bold;
            color: #B12704;
        }}

        .price-small {{
            font-size: 13px;
            color: #565959;
        }}

        .limited-deal {{
            background-color: #CC0C39;
            color: white;
            padding: 4px 8px;
            font-size: 12px;
            border-radius: 3px;
            margin-top: 8px;
            display: inline-block;
        }}

        .product-options {{
            font-size: 13px;
            color: #565959;
            margin-top: 8px;
        }}

        /* Responsive Design */
        @media (max-width: 1024px) {{
            .main-container {{
                flex-direction: column;
            }}
            
            .sidebar {{
                width: 100%;
            }}
            
            .product-grid {{
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            }}
        }}

        @media (max-width: 768px) {{
            .header-content {{
                flex-wrap: wrap;
            }}
            
            .search-container {{
                order: 3;
                width: 100%;
                margin: 10px 0 0 0;
            }}
            
            .product-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <!-- Header -->
    <header class="header">
        <div class="header-content">
            <a href="#" class="logo">amazon</a>
            
            <div class="deliver-to">
                <div>Deliver to</div>
                <div class="location">🇺🇸 Japan</div>
            </div>
            
            <div class="search-container">
                <select class="search-dropdown">
                    <option>All</option>
                </select>
                <input type="text" class="search-input" placeholder="Search Amazon">
                <button class="search-button">🔍</button>
            </div>
            
            <div class="header-right">
                <div class="header-link">🇺🇸 EN</div>
                <div class="header-link">
                    <div>Hello, sign in</div>
                    <div><strong>Account & Lists</strong></div>
                </div>
                <div class="header-link">
                    <div>Returns</div>
                    <div><strong>& Orders</strong></div>
                </div>
                <div class="cart">
                    <div class="header-link"><strong>🛒 Cart</strong></div>
                    <div class="cart-count">0</div>
                </div>
            </div>
        </div>
    </header>

    <!-- Navigation Bar -->
    <nav class="nav-bar">
        <div class="nav-content">
            <a href="#" class="nav-link">☰ All</a>
            <a href="#" class="nav-link">Today's Deals</a>
            <a href="#" class="nav-link">Registry</a>
            <a href="#" class="nav-link">Prime Video</a>
            <a href="#" class="nav-link">Gift Cards</a>
            <a href="#" class="nav-link">Customer Service</a>
            <a href="#" class="nav-link">Sell</a>
        </div>
    </nav>

    <!-- Main Container -->
    <div class="main-container">
        <!-- Sidebar -->
        <aside class="sidebar">
            <div class="filter-section">
                <h3 class="filter-title">Department</h3>
                <div class="filter-item">
                    <a href="#">< Home & Kitchen</a>
                </div>
                <div class="filter-item" style="margin-left: 20px;">
                    <strong>Small space solutions: living</strong>
                </div>
                <div class="filter-item" style="margin-left: 40px;">
                    <a href="#">Bedding</a>
                </div>
                <div class="filter-item" style="margin-left: 40px;">
                    <a href="#">Furniture</a>
                </div>
                <div class="filter-item" style="margin-left: 40px;">
                    <a href="#">Home Décor Products</a>
                </div>
            </div>

            <div class="filter-section">
                <h3 class="filter-title">Brands</h3>
                <div class="checkbox-item">
                    <input type="checkbox" id="amazon-basics">
                    <label for="amazon-basics">Amazon Basics</label>
                </div>
                <div class="checkbox-item">
                    <input type="checkbox" id="signature-design">
                    <label for="signature-design">Signature Design by Ashley</label>
                </div>
            </div>

            <div class="filter-section">
                <h3 class="filter-title">Customer Reviews</h3>
                <div class="filter-item">
                    <a href="#">
                        <span class="stars">★★★★☆</span> & Up
                    </a>
                </div>
            </div>

            <div class="filter-section">
                <h3 class="filter-title">Condition</h3>
                <div class="filter-item">
                    <a href="#">New</a>
                </div>
                <div class="filter-item">
                    <a href="#">Used</a>
                </div>
            </div>

            <div class="filter-section">
                <h3 class="filter-title">Price</h3>
                <div class="filter-item">
                    <a href="#">Under $25</a>
                </div>
                <div class="filter-item">
                    <a href="#">$25 to $50</a>
                </div>
                <div class="filter-item">
                    <a href="#">$100 to $200</a>
                </div>
                <div class="filter-item">
                    <a href="#">$200 & Above</a>
                </div>
            </div>

            <div class="filter-section">
                <h3 class="filter-title">Deals & Discounts</h3>
            </div>
        </aside>

        <!-- Main Content -->
        <main class="content">
            <div class="results-header">
                <span class="results-count">1-12 of 78 results for</span>
                <span class="search-term">Small space solutions: living</span>
            </div>

            <div class="product-grid">
                <!-- Product 1 -->
                <div class="product-card">
                    <div class="best-seller">Best Seller</div>
                    <img src="{product1_image}" alt="Product image" class="product-image"{product1_id}>
                    <h3 class="product-title">
                        <a href="#">{product1_title}</a>
                    </h3>
                    <div class="product-rating">
                        <span class="rating-stars">★★★★☆</span>
                        <span class="rating-count">2,866</span>
                    </div>
                    <div class="purchase-info">10K+ bought in past month</div>
                </div>

                <!-- Product 2 -->
                <div class="product-card">
                    <div class="best-seller">Best Seller</div>
                    <img src="{product2_image}" alt="Product image" class="product-image"{product2_id}>
                    <h3 class="product-title">
                        <a href="#">{product2_title}</a>
                    </h3>
                    <div class="product-rating">
                        <span class="rating-stars">★★★★☆</span>
                        <span class="rating-count">3,182</span>
                    </div>
                    <div class="purchase-info">5K+ bought in past month</div>
                    <div class="limited-deal">Limited time deal</div>
                </div>

                <!-- Product 3 -->
                <div class="product-card">
                    <img src="{product3_image}" alt="Product image" class="product-image"{product3_id}>
                    <h3 class="product-title">
                        <a href="#">{product3_title}</a>
                    </h3>
                    <div class="product-rating">
                        <span class="rating-stars">★★★★☆</span>
                        <span class="rating-count">4,838</span>
                    </div>
                    <div class="purchase-info">3K+ bought in past month</div>
                    <div class="product-options">Options: 2 sizes</div>
                    <div class="product-price">
                        $25<span class="price-small">19</span>
                        <span class="price-small">($4.20/count) Typical: $26.99</span>
                    </div>
                </div>

                <!-- Product 4 -->
                <div class="product-card">
                    <img src="{product4_image}" alt="Product image" class="product-image"{product4_id}>
                    <h3 class="product-title">
                        <a href="#">{product4_title}</a>
                    </h3>
                    <div class="product-rating">
                        <span class="rating-stars">★★★★☆</span>
                        <span class="rating-count">1,245</span>
                    </div>
                    <div class="purchase-info">2K+ bought in past month</div>
                    <div class="product-price">$39<span class="price-small">99</span></div>
                </div>

                <!-- Product 5 -->
                <div class="product-card">
                    <img src="{product5_image}" alt="Product image" class="product-image"{product5_id}>
                    <h3 class="product-title">
                        <a href="#">{product5_title}</a>
                    </h3>
                    <div class="product-rating">
                        <span class="rating-stars">★★★★☆</span>
                        <span class="rating-count">892</span>
                    </div>
                    <div class="purchase-info">1K+ bought in past month</div>
                    <div class="product-price">$89<span class="price-small">99</span></div>
                </div>

                <!-- Product 6 -->
                <div class="product-card">
                    <div class="best-seller">Best Seller</div>
                    <img src="{product6_image}" alt="Product image" class="product-image"{product6_id}>
                    <h3 class="product-title">
                        <a href="#">{product6_title}</a>
                    </h3>
                    <div class="product-rating">
                        <span class="rating-stars">★★★★☆</span>
                        <span class="rating-count">2,156</span>
                    </div>
                    <div class="purchase-info">4K+ bought in past month</div>
                    <div class="product-price">$45<span class="price-small">99</span></div>
                </div>
            </div>
        </main>
    </div>
</body>
</html>'''

def get_available_images(images_dir: str = "images", start_index: int = 0, end_index: int = None):
    """
    获取images目录中指定范围的图片文件列表
    
    Args:
        images_dir: 图片目录路径
        start_index: 开始索引
        end_index: 结束索引（不包含）
    
    Returns:
        list: 按编号排序的图片文件名列表
    """
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"图片目录 {images_dir} 不存在")
    
    # 支持的图片格式
    supported_formats = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
    
    # 获取所有图片文件
    image_files = []
    for file in os.listdir(images_dir):
        if any(file.lower().endswith(fmt) for fmt in supported_formats):
            # 尝试提取文件名中的数字
            try:
                # 假设文件名格式为 "数字.扩展名"
                name_without_ext = os.path.splitext(file)[0]
                if name_without_ext.isdigit():
                    file_index = int(name_without_ext)
                    # 只包含指定范围内的图片
                    if start_index <= file_index < (end_index if end_index else float('inf')):
                        image_files.append((file_index, file))
                else:
                    print(f"⚠️  跳过非数字命名的图片文件: {file}")
            except Exception as e:
                print(f"⚠️  处理图片文件 {file} 时出错: {e}")
                continue
    
    # 按数字排序
    image_files.sort(key=lambda x: x[0])
    
    sorted_files = [file for _, file in image_files]
    
    return sorted_files

def generate_amazon_html(json_path: str, output_path: str, start_index: int = 0, target_index: int = None, images_dir: str = "images", image_start: int = 0, image_end: int = None):
    """
    生成亚马逊商品页面HTML
    
    Args:
        json_path: amazon.json的路径
        output_path: 输出HTML文件路径
        start_index: 开始的数据索引
        target_index: 目标商品的索引（0-5中的一个位置，如果为None则随机选择）
        images_dir: 图片目录路径
        image_start: 图片开始索引
        image_end: 图片结束索引
    
    Returns:
        tuple: (target_product_id, target_product_info, actual_target_index)
    """
    # 读取商品数据
    with open(json_path, 'r', encoding='utf-8') as f:
        product_data = json.load(f)
    
    # 获取指定范围的可用图片列表
    available_images = get_available_images(images_dir, image_start, image_end)
    
    if not available_images:
        raise FileNotFoundError(f"在 {images_dir} 目录中的 {image_start}-{image_end} 范围内没有找到可用的图片文件")
    
    # 需要5个普通商品数据（因为第6个位置是target.jpg）
    num_regular_products = NUM_PRODUCTS - 1  # 5个
    end_idx = min(start_index + num_regular_products, len(product_data))
    selected_products = product_data[start_index:end_idx]
    
    if len(selected_products) < num_regular_products:
        print(f"⚠️  警告：只有 {len(selected_products)} 个商品数据，少于预期的 {num_regular_products} 个")
    
    # 确定target.jpg的位置（随机选择0-5中的一个位置）
    if target_index is None:
        target_index = random.randint(0, NUM_PRODUCTS - 1)
    else:
        target_index = min(target_index, NUM_PRODUCTS - 1)
    
    target_product_id = f"target-product-{target_index}"
    
    # target.jpg的固定信息
    target_product_info = {
        "title": "4-Port Charging Station for Multiple Devices, USB Charger Stations Multi-Device Organizer Charging Dock, Compatible with Cell Phones, iPads, Kindle Tablets, and Other Electronics (Gray)",
        "category": "target",
        "is_target": True
    }
    
    # 判断HTML文件的输出位置，调整图片路径
    output_dir = Path(output_path).parent
    if output_dir.name == "html":
        # 如果输出在html目录下，使用相对路径指向上级目录
        image_path_prefix = "../../" + images_dir
        target_image_path = "../../target.jpg"
    else:
        # 如果输出在当前目录或其他目录，直接使用images_dir
        image_path_prefix = images_dir
        target_image_path = "target.jpg"
    
    # 准备6个位置的数据
    positions = []
    regular_product_index = 0
    used_images = []
    
    for i in range(NUM_PRODUCTS):
        if i == target_index:
            # 这个位置放target.jpg
            positions.append({
                'image': target_image_path,
                'title': target_product_info['title'],
                'id': f' id="{target_product_id}"'
            })
            used_images.append("target.jpg")
        else:
            # 这个位置放普通图片
            if regular_product_index < len(selected_products):
                # 按顺序选择图片，从start_index对应的图片开始
                image_index = (start_index + regular_product_index) % len(available_images)
                image_filename = available_images[image_index]
                image_path = f"{image_path_prefix}/{image_filename}"
                used_images.append(image_filename)
                
                product = selected_products[regular_product_index]
                positions.append({
                    'image': image_path,
                    'title': product['title'],
                    'id': ''
                })
                regular_product_index += 1
            else:
                # 如果普通商品数据不够，用默认数据填充
                image_index = (start_index + regular_product_index) % len(available_images)
                image_filename = available_images[image_index]
                image_path = f"{image_path_prefix}/{image_filename}"
                used_images.append(image_filename)
                
                positions.append({
                    'image': image_path,
                    'title': "Default Amazon Product",
                    'id': ''
                })
                regular_product_index += 1
    
    print(f"🖼️  使用图片: {', '.join(used_images)}")
    print(f"🎯 target位置: {target_index + 1}")
    
    # 生成完整HTML
    full_html = HTML_TEMPLATE.format(
        product1_title=positions[0]['title'],
        product1_image=positions[0]['image'],
        product1_id=positions[0]['id'],
        product2_title=positions[1]['title'],
        product2_image=positions[1]['image'],
        product2_id=positions[1]['id'],
        product3_title=positions[2]['title'],
        product3_image=positions[2]['image'],
        product3_id=positions[2]['id'],
        product4_title=positions[3]['title'],
        product4_image=positions[3]['image'],
        product4_id=positions[3]['id'],
        product5_title=positions[4]['title'],
        product5_image=positions[4]['image'],
        product5_id=positions[4]['id'],
        product6_title=positions[5]['title'],
        product6_image=positions[5]['image'],
        product6_id=positions[5]['id']
    )
    
    # 确保输出目录存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    return target_product_id, target_product_info, start_index + target_index

def take_screenshot(html_path: str, output_path: str):
    """
    对HTML页面进行截图
    
    Args:
        html_path: HTML文件路径
        output_path: 截图输出路径
    """
    html_uri = Path(html_path).absolute().as_uri()
    
    with sync_playwright() as p:
        browser = p.chromium.launch(args=[
            '--font-render-hinting=none',
            '--disable-font-subpixel-positioning',
            '--disable-gpu-sandbox',
            '--disable-web-security',
            '--font-render-hinting=medium',
            '--enable-font-antialiasing'
        ])
        page = browser.new_page()
        
        # 设置桌面视口大小（适合Amazon页面）
        page.set_viewport_size({"width": 1280, "height": 800})
        
        page.goto(html_uri)
        
        # 等待页面加载完成
        page.wait_for_load_state('networkidle')
        page.wait_for_timeout(2000)  # 等待页面完全渲染
        
        # 截图
        page.screenshot(path=output_path, full_page=True)
        
        browser.close()

def get_target_image_info(html_path: str, target_product_id: str):
    """
    获取目标图像的位置和尺寸信息
    
    Args:
        html_path: HTML文件路径
        target_product_id: 目标商品的ID
    
    Returns:
        dict: 包含x, y, width, height的字典
    """
    html_uri = Path(html_path).absolute().as_uri()
    
    with sync_playwright() as p:
        browser = p.chromium.launch(args=[
            '--font-render-hinting=none',
            '--disable-font-subpixel-positioning',
            '--disable-gpu-sandbox',
            '--disable-web-security',
            '--font-render-hinting=medium',
            '--enable-font-antialiasing'
        ])
        page = browser.new_page()
        
        # 设置桌面视口大小，确保与截图一致
        page.set_viewport_size({"width": 1280, "height": 800})
        
        page.goto(html_uri)
        
        # 等待页面加载完成
        page.wait_for_load_state('networkidle')
        page.wait_for_timeout(2000)  # 等待页面完全渲染
        
        # 找到目标商品的图片元素
        selector = f"#{target_product_id}"
        img_handle = page.query_selector(selector)
        
        if not img_handle:
            browser.close()
            raise RuntimeError(f"未找到目标图片元素，选择器: {selector}")
        
        # 获取图片的边界框信息
        box = img_handle.bounding_box()
        
        browser.close()
        
        return {
            'x': int(box['x']),
            'y': int(box['y']), 
            'width': int(box['width']),
            'height': int(box['height'])
        }

def batch_generate_screenshots(json_path: str, num_batches: int = 1000, images_dir: str = "images", image_start: int = 0, image_end: int = 9000):
    """
    批量生成训练数据截图
    
    Args:
        json_path: amazon.json的路径
        num_batches: 要生成的批次数量
        images_dir: 图片目录路径
        image_start: 图片开始索引
        image_end: 图片结束索引
    
    Returns:
        dict: 包含每个批次信息的字典
    """
    # 创建输出目录
    html_dir = "html"
    screenshots_dir = "screenshots"
    Path(html_dir).mkdir(parents=True, exist_ok=True)
    Path(screenshots_dir).mkdir(parents=True, exist_ok=True)
    
    # 读取商品数据总数
    with open(json_path, 'r', encoding='utf-8') as f:
        product_data = json.load(f)
    
    total_products = len(product_data)
    max_batches = total_products // (NUM_PRODUCTS - 1)  # 因为有一个target，实际用5个普通商品
    
    if num_batches > max_batches:
        print(f"⚠️  警告：要求生成 {num_batches} 个批次，但最多只能生成 {max_batches} 个批次")
        num_batches = max_batches
    
    data_dict = {}
    
    print(f"🚀 开始批量生成 {num_batches} 个训练数据截图...")
    print(f"📊 使用图片范围: {image_start} - {image_end}")
    print()
    
    for batch_num in range(num_batches):
        start_index = batch_num * (NUM_PRODUCTS - 1)  # 每批次用5个商品数据
        
        # 生成HTML文件名（从0开始命名）
        html_filename = f"{batch_num}.html"
        html_path = os.path.join(html_dir, html_filename)
        
        # 生成截图文件名（从0开始命名）
        screenshot_filename = f"{batch_num}.png"
        screenshot_path = os.path.join(screenshots_dir, screenshot_filename)
        
        # 随机选择目标位置
        target_index = random.randint(0, NUM_PRODUCTS - 1)
        
        try:
            print(f"📝 正在生成第 {batch_num} 张训练截屏")
            
            # 生成HTML页面
            target_product_id, target_product_info, actual_target_index = generate_amazon_html(
                json_path, html_path, start_index, target_index, images_dir, image_start, image_end
            )
            
            # 进行截图
            take_screenshot(html_path, screenshot_path)
            
            # 获取目标图像信息
            target_info = get_target_image_info(html_path, target_product_id)
            
            # 按照要求的格式保存数据
            data_dict[str(batch_num)] = {
                "filename": screenshot_filename,
                "target_box": {
                    "x": target_info['x'],
                    "y": target_info['y'],
                    "w": target_info['width'],
                    "h": target_info['height']
                }
            }
            
            # 输出目标坐标信息
            print(f"🎯 target.jpg左上角坐标: ({target_info['x']}, {target_info['y']})，长宽: {target_info['width']} x {target_info['height']}")
            print(f"✅ 完成")
            print()
            
        except Exception as e:
            print(f"❌ 失败: {e}")
            continue
    
    data_file = "data.json"
    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=4)
    
    print("="*60)
    print(f"🎉 训练数据生成完成！")
    print(f"📊 总共生成了 {len(data_dict)} 个有效截屏")
    print(f"📁 HTML文件目录: {html_dir}")
    print(f"📁 截图目录: {screenshots_dir}")
    print(f"📄 数据文件: {data_file}")
    
    return data_dict

def main():
    """主函数"""
    json_path = "amazon.json"
    images_dir = "images"
    
    # 检查必要文件和目录
    if not os.path.exists(json_path):
        print(f"❌ 错误：找不到文件 {json_path}")
        return
    
    if not os.path.exists(images_dir):
        print(f"❌ 错误：找不到图片目录 {images_dir}")
        return
    
    # 批量生成训练数据截图
    data_dict = batch_generate_screenshots(
        json_path=json_path,
        num_batches=500,  # 生成500个训练数据
        images_dir=images_dir,
        image_start=0,    # 使用前2000张图片
        image_end=2500
    )

if __name__ == "__main__":
    main() 