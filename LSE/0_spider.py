'''
Amazon商品爬虫 - 爬取笔记本配件商品图片和标题
目标URL: https://www.amazon.com/s?i=specialty-aps&bbn=16225007011&rh=n%3A16225007011%2Cn%3A3011391011&ref=nav_em__nav_desktop_sa_intl_laptop_accessories_0_2_7_7
'''

import requests
import json
import time
import random
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse


class AmazonSpider(object):
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.base_url = "https://www.amazon.com/s"
        self.base_params = {
            'i': 'specialty-aps',
            'bbn': '16225007011',
            'rh': 'n:16225007011,n:3011391011',
            'ref': 'nav_em__nav_desktop_sa_intl_laptop_accessories_0_2_7_7'
        }
    
    def get_page_content(self, page=1):
        params = self.base_params.copy()
        if page > 1:
            params['page'] = page
        
        try:
            print(f'正在请求第 {page} 页...')
            response = self.session.get(self.base_url, params=params, timeout=15)
            response.raise_for_status()
            if 'robot_check' in response.url or 'captcha' in response.url.lower():
                print(f'第 {page} 页遇到验证码，跳过')
                return None
            
            response.encoding = 'utf-8'
            return response.text
            
        except requests.exceptions.RequestException as e:
            print(f'请求第 {page} 页时出错: {e}')
            return None
    
    def parse_products(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        products = []
        product_containers = soup.find_all('div', {'data-cy': 'asin-faceout-container'})
        if not product_containers:
            product_containers = soup.find_all('div', class_='puis-card-container')
        
        print(f'找到 {len(product_containers)} 个商品容器')
        
        for container in product_containers:
            try:
                product_data = {}
                img_tag = container.find('img', class_='s-image')
                if img_tag:
                    img_url = img_tag.get('src', '')
                    if not img_url:
                        img_url = img_tag.get('data-src', '')
                    if img_url:
                        if img_url.startswith('//'):
                            img_url = 'https:' + img_url
                        elif img_url.startswith('/'):
                            img_url = 'https://www.amazon.com' + img_url
                        product_data['image_url'] = img_url
                title_tag = container.find('h2', class_='a-size-base-plus')
                if not title_tag:
                    title_tag = container.find('h2', class_='s-size-mini')
                    if not title_tag:
                        title_tag = container.find('span', class_='a-size-base-plus')
                
                if title_tag:
                    title_span = title_tag.find('span')
                    if title_span:
                        title = title_span.get_text(strip=True)
                    else:
                        title = title_tag.get_text(strip=True)
                    
                    if title:
                        product_data['title'] = title
                price_container = container.find('span', class_='a-price')
                if price_container:
                    price_whole = price_container.find('span', class_='a-price-whole')
                    price_fraction = price_container.find('span', class_='a-price-fraction')
                    if price_whole and price_fraction:
                        price = f"${price_whole.get_text(strip=True)}.{price_fraction.get_text(strip=True)}"
                        product_data['price'] = price
                rating_tag = container.find('span', class_='a-icon-alt')
                if rating_tag:
                    rating_text = rating_tag.get_text(strip=True)
                    if 'out of 5 stars' in rating_text:
                        product_data['rating'] = rating_text
                if 'image_url' in product_data and 'title' in product_data:
                    products.append(product_data)
                    print(f'成功提取商品: {product_data["title"][:50]}...')
                
            except Exception as e:
                print(f'解析单个商品时出错: {e}')
                continue
        
        return products
    
    def crawl_products(self, max_pages=200):
        all_products = []
        consecutive_empty_pages = 0
        max_consecutive_empty = 5  
        
        for page in range(1, max_pages + 1):
            try:
                delay = random.uniform(2, 5)
                print(f'等待 {delay:.1f} 秒...')
                time.sleep(delay)
                
                html_content = self.get_page_content(page)
                if not html_content:
                    consecutive_empty_pages += 1
                    if consecutive_empty_pages >= max_consecutive_empty:
                        print(f'连续 {max_consecutive_empty} 页无法获取内容，停止爬取')
                        break
                    continue
                products = self.parse_products(html_content)
                
                if not products:
                    consecutive_empty_pages += 1
                    print(f'第 {page} 页没有找到商品数据')
                    if consecutive_empty_pages >= max_consecutive_empty:
                        print(f'连续 {max_consecutive_empty} 页没有商品数据，可能已到最后一页')
                        break
                    continue
                consecutive_empty_pages = 0  
                all_products.extend(products)
                print(f'第 {page} 页完成，获取到 {len(products)} 个商品，总计 {len(all_products)} 个商品')
                if page % 10 == 0:
                    print(f'已完成 {page} 页，共获取 {len(all_products)} 个商品')
                
            except KeyboardInterrupt:
                print('\n用户中断爬取')
                break
            except Exception as e:
                print(f'爬取第 {page} 页时出现未知错误: {e}')
                consecutive_empty_pages += 1
                if consecutive_empty_pages >= max_consecutive_empty:
                    break
                continue
        
        return all_products
    
    def save_to_json(self, products, filename='amazon.json'):
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(products, f, ensure_ascii=False, indent=4)
            print(f'数据已保存到 {filename}')
            return True
        except Exception as e:
            print(f'保存文件时出错: {e}')
            return False
    
    def get_statistics(self, products):
        total_products = len(products)
        products_with_price = len([p for p in products if 'price' in p])
        products_with_rating = len([p for p in products if 'rating' in p])
        
        print(f'\n=== 爬取统计 ===')
        print(f'总商品数量: {total_products}')
        print(f'包含价格信息: {products_with_price}')
        print(f'包含评分信息: {products_with_rating}')
        
        if total_products > 0:
            print(f'价格覆盖率: {products_with_price/total_products*100:.1f}%')
            print(f'评分覆盖率: {products_with_rating/total_products*100:.1f}%')


if __name__ == '__main__':
    spider = AmazonSpider()
    
    print('开始爬取Amazon笔记本配件商品...')
    print('目标页面: 笔记本配件分类')
    print('计划爬取页数: 200页')
    print('='*50)
    
    # 开始爬取
    products = spider.crawl_products(max_pages=200)
    
    if products:
        # 保存数据
        spider.save_to_json(products, 'amazon.json')
        
        # 显示统计信息
        spider.get_statistics(products)
        
        # 显示前几个商品示例
        print('\n=== 商品示例 ===')
        for i, product in enumerate(products[:3]):
            print(f'{i+1}. {product["title"][:60]}...')
            print(f'   图片: {product["image_url"]}')
            if 'price' in product:
                print(f'   价格: {product["price"]}')
            if 'rating' in product:
                print(f'   评分: {product["rating"]}')
            print()
    else:
        print('没有获取到任何商品数据')
    
    print('爬取完成!')
