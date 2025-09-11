import json
import requests
import os
import time
from urllib.parse import urlparse
import sys


def create_images_folder():
    """创建images文件夹"""
    if not os.path.exists('images'):
        os.makedirs('images')
        print("创建images文件夹")


def get_file_extension(url):
    """从URL获取文件扩展名"""
    parsed_url = urlparse(url)
    path = parsed_url.path
    if '.' in path:
        return path.split('.')[-1].lower()
    return 'jpg'  # 默认为jpg


def download_image(url, filename, max_retries=3):
    """下载单张图片"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': 'https://www.bilibili.com/'
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            with open(filename, 'wb') as f:
                f.write(response.content)
            
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"下载失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)  # 等待2秒后重试
            
    return False


def main():
    json_file = 'amazon.json'
    
    # 检查JSON文件是否存在
    if not os.path.exists(json_file):
        print(f"错误: 找不到文件 {json_file}")
        sys.exit(1)
    
    # 创建images文件夹
    create_images_folder()
    
    # 读取JSON文件
    print(f"正在读取 {json_file}...")
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取JSON文件失败: {e}")
        sys.exit(1)
    
    # 提取所有图片URL
    image_urls = []
    
    # 递归遍历JSON数据，查找所有包含"url"字段的图片链接
    def extract_urls(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == 'image_url' and isinstance(value, str) and any(ext in value.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                    image_urls.append(value)
                else:
                    extract_urls(value)
        elif isinstance(obj, list):
            for item in obj:
                extract_urls(item)
    
    extract_urls(data)
    
    print(f"找到 {len(image_urls)} 张图片")
    
    if len(image_urls) == 0:
        print("没有找到图片URL")
        return
    
    # 下载图片
    successful_downloads = 0
    failed_downloads = 0
    
    for i, url in enumerate(image_urls):
        extension = get_file_extension(url)
        filename = f'images/{i}.{extension}'
        
        print(f"正在下载第 {i + 1}/{len(image_urls)} 张图片")
        
        if download_image(url, filename):
            successful_downloads += 1
            print(f"✓ 下载成功: {filename}")
        else:
            failed_downloads += 1
            print(f"✗ 下载失败: {url}")
        
        # 添加延迟，避免请求过于频繁
        time.sleep(0.5)
    
    print(f"\n下载完成！")
    print(f"成功下载: {successful_downloads} 张")
    print(f"下载失败: {failed_downloads} 张")
    print(f"总计: {len(image_urls)} 张")


if __name__ == '__main__':
    main() 