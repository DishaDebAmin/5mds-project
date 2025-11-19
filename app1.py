from flask import Flask, render_template, request, jsonify, send_file
import os
import json
import random
import math
import requests
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import io
import base64
import re
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'campaigniq-secret-key-2024')
app.config['UPLOAD_FOLDER'] = 'static/generated'

# Ensure generated folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# API Configuration with proper error handling
class APIConfig:
    # OpenAI for text generation
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_BASE_URL = "https://api.openai.com/v1/chat/completions"
    
    # Leonardo AI for professional logo generation
    LEONARDO_API_KEY = os.getenv('LEONARDO_API_KEY')
    LEONARDO_API_URL = "https://cloud.leonardo.ai/api/rest/v1/generations"
    
    # Hugging Face as fallback
    HF_API_KEY = os.getenv('HF_API_KEY')
    HF_LOGO_API = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
    
    @classmethod
    def validate_keys(cls):
        """Validate that required API keys are present"""
        missing_keys = []
        if not cls.OPENAI_API_KEY:
            missing_keys.append("OPENAI_API_KEY")
        if not cls.LEONARDO_API_KEY:
            missing_keys.append("LEONARDO_API_KEY")
        
        if missing_keys:
            print(f"⚠️ WARNING: Missing API keys: {', '.join(missing_keys)}")
            print("ℹ️ Some features will use fallback methods")
        else:
            print("✅ All API keys are configured")
        
        return len(missing_keys) == 0

# Initialize and validate API config
api_config = APIConfig()
api_config.validate_keys()

class LLMTextGenerator:
    def __init__(self):
        self.api_key = api_config.OPENAI_API_KEY
        self.base_url = api_config.OPENAI_BASE_URL
        self.available = bool(self.api_key)
    
    def generate_with_llm(self, prompt, max_tokens=150, temperature=0.7):
        """Generate text using OpenAI GPT-4 with enhanced error handling"""
        if not self.available:
            return self._fallback_generation(prompt)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "system", 
                    "content": """You are a creative marketing expert specializing in brand development, 
                    advertising, and social media campaigns. Create compelling, concise, and engaging 
                    marketing content that resonates with target audiences."""
                },
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            elif response.status_code == 401:
                print("❌ OpenAI API: Invalid API key")
                self.available = False
            elif response.status_code == 429:
                print("⏳ OpenAI API: Rate limit exceeded")
            else:
                print(f"❌ OpenAI API Error: {response.status_code}")
            
            return self._fallback_generation(prompt)
            
        except requests.exceptions.Timeout:
            print("⏰ OpenAI API: Request timeout")
            return self._fallback_generation(prompt)
        except Exception as e:
            print(f"❌ OpenAI API Exception: {str(e)}")
            return self._fallback_generation(prompt)
    
    def _fallback_generation(self, prompt):
        """Sophisticated fallback generation when API fails"""
        prompt_lower = prompt.lower()
        
        if "tagline" in prompt_lower:
            tagline_templates = [
                "Elevate Your Experience",
                "Quality That Inspires",
                "Innovation Meets Excellence", 
                "Designed for Modern Life",
                "Where Quality Meets Passion",
                "Experience the Difference",
                "Crafting Exceptional Moments",
                "Your Vision, Our Mission"
            ]
            return random.choice(tagline_templates)
        
        elif "ad copy" in prompt_lower or "description" in prompt_lower:
            ad_copy_templates = [
                "Discover a new standard of excellence. Our commitment to quality and innovation ensures an unforgettable experience that transforms everyday moments into extraordinary memories.",
                "Experience the perfect blend of style and functionality. Designed with precision and crafted with care, our products elevate your daily routine with unparalleled quality.",
                "Join thousands of satisfied customers who have transformed their experience. Our innovative approach and attention to detail create solutions that truly make a difference."
            ]
            return random.choice(ad_copy_templates)
        
        return "Quality and innovation that transforms your experience. Discover the difference today."

class AILogoGenerator:
    def __init__(self):
        self.api_key = api_config.LEONARDO_API_KEY
        self.base_url = api_config.LEONARDO_API_URL
        self.available = bool(self.api_key)
        self.llm_generator = LLMTextGenerator()
    
    def generate_logo_with_ai(self, brand_name, prompt, style="creative", industry="general"):
        """Generate professional logo using Leonardo AI with enhanced prompt engineering"""
        if not self.available:
            return self._generate_intelligent_fallback_logo(brand_name, prompt, style, industry)
        
        # Enhanced prompt engineering with industry-specific guidance
        enhanced_prompt = self._build_enhanced_prompt(brand_name, prompt, style, industry)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "prompt": enhanced_prompt,
            "modelId": "6bef9f1b-29cb-40c7-b9df-32b51c1f67d3",
            "width": 512,
            "height": 512,
            "num_images": 1,
            "guidance_scale": 7.5,
            "steps": 40,
            "negative_prompt": "text, letters, words, signature, watermark, blurry, pixelated, low quality, distorted, amateur, cartoonish, childish, complex, busy, cluttered"
        }
        
        try:
            print(f"🔄 Generating logo with enhanced prompt: {enhanced_prompt}")
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                generation_data = response.json()
                generation_id = generation_data.get('sdGenerationJob', {}).get('generationId')
                
                if generation_id:
                    logo_url = self._wait_for_generation(generation_id)
                    if logo_url:
                        logo_data = self._download_and_save_logo(logo_url, brand_name)
                        if logo_data:
                            print("✅ Logo generated successfully via Leonardo AI")
                            # Generate design reason
                            design_reason = self._generate_design_reason(brand_name, prompt, style, industry, enhanced_prompt)
                            logo_data['design_reason'] = design_reason
                            return logo_data
            
            print("❌ Logo generation failed, using intelligent fallback")
            return self._generate_intelligent_fallback_logo(brand_name, prompt, style, industry)
                
        except Exception as e:
            print(f"❌ Logo Generation Error: {str(e)}")
            return self._generate_intelligent_fallback_logo(brand_name, prompt, style, industry)
    
    def _build_enhanced_prompt(self, brand_name, prompt, style, industry):
        """Build enhanced prompt with industry-specific guidance"""
        # Industry-specific guidance
        industry_prompts = {
            "technology": "tech logo, innovation, digital, futuristic, circuits, nodes, connectivity, modern technology, geometric patterns, data flow",
            "fashion": "fashion logo, elegant, stylish, modern, apparel, clothing, accessory, luxury, boutique, minimalist design, sophisticated",
            "food": "food logo, culinary, delicious, fresh, organic, ingredients, appetizing, restaurant, cafe, natural elements, warm colors",
            "health": "health logo, wellness, fitness, medical, clean, natural, vitality, healthcare, balance, growth, life",
            "finance": "finance logo, trust, security, growth, stability, professional, corporate, banking, investment, shield, upward trend",
            "education": "education logo, learning, knowledge, growth, academic, books, graduation, school, wisdom, light bulb, open book",
            "realestate": "real estate logo, property, home, building, architecture, trust, growth, foundation, structure, key",
            "automotive": "automotive logo, vehicles, speed, precision, engineering, mobility, innovation, wheels, motion, dynamic",
            "eco": "eco logo, sustainable, green, environment, nature, organic, renewable, eco-friendly, leaves, earth, growth"
        }
        
        industry_guidance = industry_prompts.get(industry.lower(), "professional, modern, clean business logo, corporate identity")
        
        # Style-specific guidance
        style_prompts = {
            "minimal": "minimalist, clean, simple, elegant, modern, vector logo, professional branding, flat design, negative space",
            "creative": "creative, abstract, artistic, unique, colorful, modern logo, innovative design, imaginative, dynamic composition",
            "elegant": "elegant, sophisticated, luxury, premium, refined, classic logo, gold accents, serif typography, timeless", 
            "bold": "bold, strong, impactful, powerful, dramatic, confident logo, striking design, high contrast, geometric shapes",
            "modern": "modern, contemporary, sleek, futuristic, innovative, tech logo, clean lines, geometric, asymmetrical"
        }
        
        style_guidance = style_prompts.get(style, "professional, modern, clean logo")
        
        # Build the final prompt
        base_prompt = f"Professional logo design for {brand_name}, {prompt}"
        enhanced_prompt = f"{base_prompt}, {style_guidance}, {industry_guidance}, vector logo, emblem, icon, professional branding, corporate identity, clean design, high resolution, 4K, detailed, professional, no text"
        
        # Remove any potentially problematic characters
        enhanced_prompt = re.sub(r'[^\w\s.,!?-]', '', enhanced_prompt)
        
        return enhanced_prompt
    
    def _generate_design_reason(self, brand_name, prompt, style, industry, enhanced_prompt):
        """Generate a detailed reason for the logo design choices"""
        reason_prompt = f"""
        Explain the design rationale for a logo for {brand_name} in the {industry} industry.
        
        Design Requirements: {prompt}
        Style: {style}
        Industry: {industry}
        AI Prompt Used: {enhanced_prompt}

        Provide a concise explanation (2-3 sentences) of how this logo design reflects:
        1. The brand identity and industry
        2. The chosen style and design approach
        3. The target audience and market positioning
        
        Focus on the symbolism, color choices, and overall design philosophy.
        """
        
        try:
            reason = self.llm_generator.generate_with_llm(reason_prompt, max_tokens=120, temperature=0.7)
            return reason
        except Exception as e:
            print(f"Design reason generation error: {str(e)}")
            return f"This {style} logo design for {brand_name} incorporates {industry}-appropriate symbolism and color psychology to create a memorable brand identity that resonates with your target audience."
    
    def _wait_for_generation(self, generation_id, max_wait=90):
        """Wait for logo generation to complete with progress tracking"""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        check_url = f"https://cloud.leonardo.ai/api/rest/v1/generations/{generation_id}"
        
        print("⏳ Waiting for logo generation...")
        for i in range(max_wait // 10):
            time.sleep(10)
            try:
                response = requests.get(check_url, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    generations = data.get('generations_by_pk', {})
                    status = generations.get('status')
                    
                    if status == 'COMPLETE':
                        images = generations.get('generated_images', [])
                        if images:
                            print("✅ Logo generation completed")
                            return images[0].get('url')
                    elif status == 'FAILED':
                        print("❌ Logo generation failed")
                        break
                    else:
                        print(f"🔄 Generation status: {status}")
            except Exception as e:
                print(f"⚠️ Status check error: {str(e)}")
        
        return None
    
    def _download_and_save_logo(self, image_url, brand_name):
        """Download and save the generated logo"""
        try:
            response = requests.get(image_url, timeout=30)
            if response.status_code == 200:
                # Create safe filename
                safe_brand_name = re.sub(r'[^\w]', '_', brand_name.lower())
                timestamp = int(time.time())
                filename = f"logo_ai_{safe_brand_name}_{timestamp}.png"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                # Verify the image was saved correctly
                if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                    return {
                        'filename': filename,
                        'path': f"/static/generated/{filename}",
                        'style': 'ai_generated',
                        'source': 'leonardo_ai',
                        'description': f'AI-generated professional logo for {brand_name}',
                        'timestamp': timestamp
                    }
        
        except Exception as e:
            print(f"❌ Logo download error: {str(e)}")
        
        return None
    
    def _generate_intelligent_fallback_logo(self, brand_name, prompt, style, industry):
        """Generate highly relevant fallback logos based on prompt and industry"""
        width, height = 400, 400
        
        # Analyze prompt for key elements and choose appropriate design
        design_function = self._analyze_prompt_for_design(prompt, industry)
        logo, design_name = design_function(width, height, brand_name, prompt, style, industry)
        
        # Generate design reason for fallback logo
        design_reason = self._generate_fallback_design_reason(brand_name, prompt, style, industry, design_name)
        
        # Save logo with timestamp
        safe_brand_name = re.sub(r'[^\w]', '_', brand_name.lower())
        timestamp = int(time.time())
        filename = f"logo_intelligent_{safe_brand_name}_{timestamp}.png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logo.save(filepath, 'PNG', quality=95)
        
        return {
            'filename': filename,
            'path': f"/static/generated/{filename}",
            'style': style,
            'source': f'intelligent_fallback_{design_name}',
            'description': f'Intelligent {design_name} logo for {brand_name} - {prompt}',
            'design_reason': design_reason,
            'timestamp': timestamp
        }
    
    def _generate_fallback_design_reason(self, brand_name, prompt, style, industry, design_name):
        """Generate design reason for fallback logos"""
        industry_descriptions = {
            "technology": "incorporating geometric tech elements and digital connectivity symbolism",
            "fashion": "featuring elegant fashion motifs and sophisticated styling",
            "food": "using appetizing culinary elements and warm, inviting colors",
            "health": "incorporating wellness symbols and balanced, natural elements",
            "finance": "featuring trust-based financial symbols and stable geometric forms",
            "education": "using knowledge and growth symbolism with academic elements",
            "eco": "incorporating natural, sustainable elements and environmental symbolism"
        }
        
        industry_desc = industry_descriptions.get(industry, "featuring professional business elements")
        
        return f"This {style} logo for {brand_name} uses {industry_desc} to create a distinctive brand identity that aligns with your '{prompt}' vision and appeals to your target market."
    
    def _analyze_prompt_for_design(self, prompt, industry):
        """Analyze prompt to determine the best design approach"""
        prompt_lower = prompt.lower()
        
        # Industry-specific designs
        industry_designs = {
            "technology": self._create_tech_logo,
            "fashion": self._create_fashion_logo,
            "food": self._create_food_logo,
            "health": self._create_health_logo,
            "finance": self._create_finance_logo,
            "education": self._create_education_logo,
            "eco": self._create_eco_logo
        }
        
        if industry in industry_designs:
            return industry_designs[industry]
        
        # Default based on style keywords in prompt
        if any(word in prompt_lower for word in ['tech', 'digital', 'software', 'app', 'ai', 'computer']):
            return self._create_tech_logo
        elif any(word in prompt_lower for word in ['food', 'restaurant', 'cafe', 'culinary', 'meal', 'cooking']):
            return self._create_food_logo
        elif any(word in prompt_lower for word in ['fashion', 'clothing', 'apparel', 'style', 'wear', 'boutique']):
            return self._create_fashion_logo
        elif any(word in prompt_lower for word in ['health', 'fitness', 'wellness', 'medical', 'care', 'therapy']):
            return self._create_health_logo
        elif any(word in prompt_lower for word in ['finance', 'bank', 'investment', 'money', 'wealth', 'financial']):
            return self._create_finance_logo
        elif any(word in prompt_lower for word in ['nature', 'eco', 'sustainable', 'green', 'environment']):
            return self._create_eco_logo
        else:
            return self._create_modern_logo
    
    def _create_tech_logo(self, width, height, brand_name, prompt, style, industry):
        """Create technology-focused logo"""
        logo = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(logo)
        
        colors = self._get_industry_color_scheme('technology')
        
        # Tech elements: circuits, nodes, connectivity
        center_x, center_y = width // 2, height // 2
        
        # Circuit board pattern with connected nodes
        for i in range(6):
            angle = 2 * math.pi * i / 6
            radius = 80
            node_x = center_x + radius * math.cos(angle)
            node_y = center_y + radius * math.sin(angle)
            
            # Draw node
            draw.ellipse([node_x-15, node_y-15, node_x+15, node_y+15],
                       fill=colors['primary'], outline=colors['secondary'], width=2)
            
            # Draw connecting lines
            next_angle = 2 * math.pi * (i + 1) / 6
            next_x = center_x + radius * math.cos(next_angle)
            next_y = center_y + radius * math.sin(next_angle)
            draw.line([(node_x, node_y), (next_x, next_y)], fill=colors['accent'], width=3)
        
        # Central processing unit element
        draw.rectangle([center_x-25, center_y-25, center_x+25, center_y+25],
                     fill=colors['secondary'], outline=colors['primary'], width=3)
        
        return logo, "technology"
    
    def _create_food_logo(self, width, height, brand_name, prompt, style, industry):
        """Create food industry logo"""
        logo = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(logo)
        
        colors = self._get_industry_color_scheme('food')
        
        # Food elements: circular plate with abstract food representation
        center_x, center_y = width // 2, height // 2
        
        # Plate
        draw.ellipse([center_x-80, center_y-80, center_x+80, center_y+80],
                   fill=colors['background'], outline=colors['primary'], width=4)
        
        # Abstract food elements (utensils and ingredients)
        # Fork
        draw.rectangle([center_x-30, center_y-60, center_x-10, center_y+40],
                     fill=colors['primary'])
        # Spoon
        draw.ellipse([center_x+5, center_y-50, center_x+25, center_y-30],
                   fill=colors['accent'])
        draw.rectangle([center_x+10, center_y-30, center_x+20, center_y+40],
                     fill=colors['accent'])
        
        return logo, "food"
    
    def _create_fashion_logo(self, width, height, brand_name, prompt, style, industry):
        """Create fashion industry logo"""
        logo = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(logo)
        
        colors = self._get_industry_color_scheme('fashion')
        
        # Fashion elements: hanger, clothing silhouette
        center_x, center_y = width // 2, height // 2
        
        # Clothing hanger
        draw.arc([center_x-40, center_y-60, center_x+40, center_y+20], 0, 180, fill=colors['primary'], width=6)
        draw.line([(center_x, center_y+20), (center_x, center_y+50)], fill=colors['primary'], width=6)
        
        # Dress/shirt silhouette
        draw.ellipse([center_x-25, center_y-35, center_x+25, center_y+15],
                   outline=colors['secondary'], width=4)
        
        return logo, "fashion"
    
    def _create_health_logo(self, width, height, brand_name, prompt, style, industry):
        """Create health/wellness logo"""
        logo = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(logo)
        
        colors = self._get_industry_color_scheme('health')
        
        # Health elements: cross, heart, leaves
        center_x, center_y = width // 2, height // 2
        
        # Medical cross
        draw.rectangle([center_x-40, center_y-15, center_x+40, center_y+15],
                     fill=colors['primary'])
        draw.rectangle([center_x-15, center_y-40, center_x+15, center_y+40],
                     fill=colors['primary'])
        
        # Heart pulse line
        pulse_points = [
            (center_x-30, center_y),
            (center_x-15, center_y-20),
            (center_x, center_y),
            (center_x+15, center_y+20),
            (center_x+30, center_y)
        ]
        draw.line(pulse_points, fill=colors['accent'], width=4)
        
        return logo, "health"
    
    def _create_finance_logo(self, width, height, brand_name, prompt, style, industry):
        """Create finance industry logo"""
        logo = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(logo)
        
        colors = self._get_industry_color_scheme('finance')
        
        # Finance elements: graphs, shields, growth arrows
        center_x, center_y = width // 2, height // 2
        
        # Shield shape for security
        points = [
            (center_x, center_y-50),
            (center_x-40, center_y-20),
            (center_x-40, center_y+30),
            (center_x, center_y+50),
            (center_x+40, center_y+30),
            (center_x+40, center_y-20)
        ]
        draw.polygon(points, fill=colors['primary'] + '80', outline=colors['secondary'], width=3)
        
        # Growth graph inside
        graph_points = [
            (center_x-25, center_y+20),
            (center_x-10, center_y-10),
            (center_x+5, center_y+5),
            (center_x+20, center_y-15)
        ]
        draw.line(graph_points, fill=colors['accent'], width=4)
        
        return logo, "finance"
    
    def _create_education_logo(self, width, height, brand_name, prompt, style, industry):
        """Create education logo"""
        logo = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(logo)
        
        colors = self._get_industry_color_scheme('education')
        
        # Education elements: books, graduation cap, light bulb
        center_x, center_y = width // 2, height // 2
        
        # Book stack
        for i in range(3):
            y_offset = i * 8
            draw.rectangle([center_x-40, center_y-25+y_offset, center_x+40, center_y+25+y_offset],
                         outline=colors['primary'], width=2)
        
        # Knowledge light rays
        for i in range(4):
            angle = 2 * math.pi * i / 4
            start_x = center_x + 45 * math.cos(angle)
            start_y = center_y + 45 * math.sin(angle)
            end_x = center_x + 70 * math.cos(angle)
            end_y = center_y + 70 * math.sin(angle)
            draw.line([(start_x, start_y), (end_x, end_y)], fill=colors['accent'], width=3)
        
        return logo, "education"
    
    def _create_eco_logo(self, width, height, brand_name, prompt, style, industry):
        """Create eco/nature logo"""
        logo = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(logo)
        
        colors = self._get_industry_color_scheme('eco')
        
        # Eco elements: leaves, trees, earth
        center_x, center_y = width // 2, height // 2
        
        # Tree/leaf motif
        # Trunk
        draw.rectangle([center_x-8, center_y-20, center_x+8, center_y+30],
                     fill=colors['secondary'])
        
        # Leaves in circular pattern
        for i in range(5):
            angle = 2 * math.pi * i / 5
            leaf_x = center_x + 40 * math.cos(angle)
            leaf_y = center_y - 10 + 40 * math.sin(angle)
            draw.ellipse([leaf_x-15, leaf_y-15, leaf_x+15, leaf_y+15],
                       fill=colors['primary'] + 'CC')
        
        return logo, "eco"
    
    def _create_modern_logo(self, width, height, brand_name, prompt, style, industry):
        """Create modern abstract logo"""
        logo = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(logo)
        
        colors = self._get_industry_color_scheme('general')
        
        # Modern abstract elements
        center_x, center_y = width // 2, height // 2
        
        # Layered geometric shapes
        # Outer circle
        draw.ellipse([center_x-80, center_y-80, center_x+80, center_y+80],
                   outline=colors['primary'], width=4)
        
        # Inner square rotated
        square_size = 100
        points = [
            (center_x, center_y-square_size//2),
            (center_x+square_size//2, center_y),
            (center_x, center_y+square_size//2),
            (center_x-square_size//2, center_y)
        ]
        draw.polygon(points, outline=colors['accent'], width=3)
        
        # Central dot
        draw.ellipse([center_x-10, center_y-10, center_x+10, center_y+10],
                   fill=colors['secondary'])
        
        return logo, "modern"
    
    def _get_industry_color_scheme(self, industry):
        """Get color scheme based on industry"""
        color_schemes = {
            "technology": {'primary': '#3498DB', 'secondary': '#2C3E50', 'accent': '#1ABC9C', 'neutral': '#34495E', 'background': '#FFFFFF'},
            "fashion": {'primary': '#E74C3C', 'secondary': '#C0392B', 'accent': '#9B59B6', 'neutral': '#2C3E50', 'background': '#FFFFFF'},
            "food": {'primary': '#E67E22', 'secondary': '#D35400', 'accent': '#F39C12', 'neutral': '#2C3E50', 'background': '#FFFFFF'},
            "health": {'primary': '#27AE60', 'secondary': '#2ECC71', 'accent': '#3498DB', 'neutral': '#2C3E50', 'background': '#FFFFFF'},
            "finance": {'primary': '#2C3E50', 'secondary': '#34495E', 'accent': '#F39C12', 'neutral': '#7F8C8D', 'background': '#FFFFFF'},
            "education": {'primary': '#9B59B6', 'secondary': '#8E44AD', 'accent': '#3498DB', 'neutral': '#2C3E50', 'background': '#FFFFFF'},
            "eco": {'primary': '#27AE60', 'secondary': '#2ECC71', 'accent': '#16A085', 'neutral': '#2C3E50', 'background': '#FFFFFF'},
            "general": {'primary': '#4A90E2', 'secondary': '#2C3E50', 'accent': '#FF6B6B', 'neutral': '#34495E', 'background': '#FFFFFF'}
        }
        return color_schemes.get(industry.lower(), color_schemes["general"])

class IntelligentScheduler:
    def __init__(self):
        self.platform_data = self._load_platform_data()
    
    def _load_platform_data(self):
        """Load optimal posting times for different platforms"""
        return {
            'instagram': {
                'best_times': ['09:00', '12:00', '15:00', '17:00', '19:00', '21:00'],
                'best_days': [1, 2, 3, 4, 5],
                'engagement_peak': '19:00',
                'content_types': ['Reels', 'Carousel', 'Single Image', 'Story']
            },
            'facebook': {
                'best_times': ['08:00', '11:00', '13:00', '15:00', '18:00', '20:00'],
                'best_days': [3, 4, 5, 6],
                'engagement_peak': '15:00',
                'content_types': ['Video', 'Image', 'Link', 'Carousel']
            },
            'twitter': {
                'best_times': ['07:00', '12:00', '15:00', '17:00', '20:00'],
                'best_days': [1, 2, 3, 4],
                'engagement_peak': '12:00',
                'content_types': ['Text', 'Image', 'Poll', 'Thread']
            },
            'linkedin': {
                'best_times': ['08:00', '10:00', '12:00', '14:00', '16:00'],
                'best_days': [1, 2, 3, 4],
                'engagement_peak': '10:00',
                'content_types': ['Article', 'Image', 'Video', 'Document']
            },
            'tiktok': {
                'best_times': ['09:00', '12:00', '16:00', '19:00', '21:00', '23:00'],
                'best_days': [2, 3, 4, 5, 6],
                'engagement_peak': '21:00',
                'content_types': ['Short Video', 'Duet', 'Story']
            }
        }
    
    def generate_optimized_schedule(self, target_audience, campaign_duration=7, platforms=None):
        """Generate intelligent posting schedule based on audience and platform data"""
        if platforms is None:
            platforms = ['instagram', 'facebook', 'twitter']
        
        audience_profile = self._analyze_audience(target_audience)
        schedule = []
        start_date = datetime.now()
        
        for day in range(campaign_duration):
            current_date = start_date + timedelta(days=day)
            weekday = current_date.weekday()
            
            for platform in platforms:
                if platform.lower() in self.platform_data:
                    platform_info = self.platform_data[platform.lower()]
                    
                    # Check if this is a good day for the platform
                    if weekday in platform_info['best_days']:
                        # Select 2-3 best times for this platform
                        best_times = random.sample(platform_info['best_times'], 
                                                 random.randint(2, 3))
                        
                        for time_str in best_times:
                            hour, minute = map(int, time_str.split(':'))
                            post_time = current_date.replace(hour=hour, minute=minute, second=0)
                            
                            # Calculate expected engagement
                            engagement = self._calculate_engagement(
                                platform, time_str, audience_profile, weekday
                            )
                            
                            content_type = random.choice(platform_info['content_types'])
                            
                            schedule.append({
                                'datetime': post_time.strftime('%Y-%m-%d %H:%M'),
                                'platform': platform.capitalize(),
                                'expected_engagement': f"{engagement}%",
                                'best_for': audience_profile['type'],
                                'content_type': content_type,
                                'optimal_timing': self._get_timing_quality(time_str, platform_info),
                                'audience_online': self._get_online_audience(time_str, audience_profile)
                            })
        
        # Sort by datetime and return top 15 posts
        schedule.sort(key=lambda x: x['datetime'])
        return schedule[:15]
    
    def _analyze_audience(self, audience_description):
        """Analyze audience for optimal scheduling"""
        audience_lower = audience_description.lower()
        
        profile = {
            'type': 'general',
            'age_group': 'adult',
            'lifestyle': 'balanced',
            'time_preference': 'evening'
        }
        
        # Age group detection
        if any(word in audience_lower for word in ['teen', 'student', 'young', 'gen z', 'youth']):
            profile['age_group'] = 'youth'
            profile['time_preference'] = 'evening'
        elif any(word in audience_lower for word in ['professional', 'business', 'corporate']):
            profile['age_group'] = 'professional'
            profile['time_preference'] = 'lunch_evening'
        elif any(word in audience_lower for word in ['family', 'parents', 'mother', 'father']):
            profile['age_group'] = 'family'
            profile['time_preference'] = 'morning_evening'
        elif any(word in audience_lower for word in ['senior', 'retired', 'elderly']):
            profile['age_group'] = 'senior'
            profile['time_preference'] = 'morning'
        
        # Audience type
        if any(word in audience_lower for word in ['tech', 'digital', 'gamer', 'stream']):
            profile['type'] = 'tech_savvy'
        elif any(word in audience_lower for word in ['creative', 'art', 'design', 'music']):
            profile['type'] = 'creative'
        elif any(word in audience_lower for word in ['fitness', 'health', 'wellness', 'sport']):
            profile['type'] = 'health_conscious'
        elif any(word in audience_lower for word in ['luxury', 'premium', 'executive']):
            profile['type'] = 'affluent'
        
        return profile
    
    def _calculate_engagement(self, platform, time_str, audience_profile, weekday):
        """Calculate expected engagement percentage"""
        base_engagement = 70
        
        # Platform modifiers
        platform_modifiers = {
            'instagram': 10,
            'tiktok': 15,
            'facebook': 5,
            'twitter': 8,
            'linkedin': 12
        }
        
        # Time modifiers
        hour = int(time_str.split(':')[0])
        if 7 <= hour <= 9:
            time_modifier = 5  # Morning
        elif 12 <= hour <= 14:
            time_modifier = 10  # Lunch
        elif 17 <= hour <= 19:
            time_modifier = 15  # Evening
        elif 20 <= hour <= 23:
            time_modifier = 12  # Night
        else:
            time_modifier = 0
        
        # Audience modifiers
        audience_modifiers = {
            'youth': 10,
            'professional': 8,
            'family': 6,
            'senior': 4
        }
        
        # Weekend modifier
        weekend_modifier = 5 if weekday >= 5 else 0
        
        engagement = (base_engagement + 
                     platform_modifiers.get(platform.lower(), 0) +
                     time_modifier +
                     audience_modifiers.get(audience_profile['age_group'], 0) +
                     weekend_modifier)
        
        return min(95, max(60, engagement))
    
    def _get_timing_quality(self, time_str, platform_info):
        """Get timing quality description"""
        if time_str == platform_info['engagement_peak']:
            return "Peak Time"
        elif time_str in platform_info['best_times'][:3]:
            return "Optimal"
        else:
            return "Good"
    
    def _get_online_audience(self, time_str, audience_profile):
        """Estimate online audience percentage"""
        base_audience = 60
        
        time_preference = audience_profile['time_preference']
        hour = int(time_str.split(':')[0])
        
        if time_preference == 'morning' and 6 <= hour <= 10:
            base_audience += 20
        elif time_preference == 'lunch_evening' and (12 <= hour <= 14 or 17 <= hour <= 20):
            base_audience += 25
        elif time_preference == 'evening' and 17 <= hour <= 23:
            base_audience += 30
        elif time_preference == 'morning_evening' and (7 <= hour <= 9 or 18 <= hour <= 21):
            base_audience += 15
        
        return f"{min(95, base_audience)}%"

class AdvancedCampaignGenerator:
    def __init__(self):
        self.llm_generator = LLMTextGenerator()
        self.logo_generator = AILogoGenerator()
        self.scheduler = IntelligentScheduler()
    
    def detect_industry(self, campaign_idea, brand_name):
        """Detect industry from campaign idea and brand name"""
        idea_lower = (campaign_idea + " " + brand_name).lower()
        
        industries = {
            "technology": ['tech', 'software', 'app', 'digital', 'ai', 'computer', 'internet', 'mobile', 'web', 'data', 'cloud'],
            "fashion": ['fashion', 'clothing', 'apparel', 'wear', 'style', 'outfit', 'dress', 'boutique', 'garment'],
            "food": ['food', 'restaurant', 'cafe', 'culinary', 'recipe', 'cooking', 'dining', 'meal', 'eat', 'beverage'],
            "health": ['health', 'fitness', 'wellness', 'medical', 'doctor', 'hospital', 'exercise', 'gym', 'therapy', 'care'],
            "finance": ['finance', 'bank', 'investment', 'money', 'wealth', 'financial', 'loan', 'insurance', 'capital'],
            "education": ['education', 'learning', 'school', 'university', 'course', 'training', 'knowledge', 'academy'],
            "realestate": ['real estate', 'property', 'home', 'house', 'apartment', 'construction', 'realty'],
            "automotive": ['car', 'auto', 'vehicle', 'motor', 'drive', 'transportation', 'automobile'],
            "eco": ['eco', 'sustainable', 'green', 'environment', 'nature', 'organic', 'renewable', 'planet']
        }
        
        for industry, keywords in industries.items():
            if any(keyword in idea_lower for keyword in keywords):
                return industry
        
        return "general"
    
    def generate_campaign_assets(self, campaign_data):
        """Generate all campaign assets using AI models"""
        try:
            # Detect industry for better logo generation
            industry = self.detect_industry(campaign_data['idea'], campaign_data['brand_name'])
            print(f"🎯 Detected industry: {industry}")
            
            # Generate tagline using LLM
            tagline_prompt = self._create_tagline_prompt(
                campaign_data['brand_name'],
                campaign_data['idea'],
                campaign_data['target_audience'],
                campaign_data.get('logo_prompt', ''),
                industry
            )
            tagline = self.llm_generator.generate_with_llm(tagline_prompt, max_tokens=50)
            
            # Generate ad copy using LLM
            ad_copy_prompt = self._create_ad_copy_prompt(
                campaign_data['brand_name'],
                campaign_data['idea'],
                campaign_data['target_audience'],
                tagline,
                industry
            )
            ad_copy_text = self.llm_generator.generate_with_llm(ad_copy_prompt, max_tokens=200)
            
            # Parse ad copy into structured format
            ad_copy = self._parse_ad_copy(ad_copy_text, tagline)
            
            # Generate logo using AI with industry context
            logo_data = self.logo_generator.generate_logo_with_ai(
                campaign_data['brand_name'],
                campaign_data.get('logo_prompt', 'professional modern logo'),
                campaign_data.get('style', 'creative'),
                industry
            )
            
            # Generate optimized schedule
            schedule = self.scheduler.generate_optimized_schedule(
                campaign_data['target_audience'],
                platforms=['instagram', 'facebook', 'twitter', 'linkedin']
            )
            
            # Generate color palette
            colors = self._generate_color_palette(campaign_data['brand_color'])
            
            return {
                'tagline': tagline,
                'ad_copy': ad_copy,
                'logo': logo_data,
                'schedule': schedule,
                'colors': colors,
                'campaign_data': campaign_data,
                'industry': industry
            }
            
        except Exception as e:
            print(f"Campaign Generation Error: {str(e)}")
            return self._generate_fallback_assets(campaign_data)
    
    def _create_tagline_prompt(self, brand_name, idea, audience, logo_prompt, industry):
        """Create prompt for tagline generation"""
        return f"""
        Create a compelling brand tagline for {brand_name} in the {industry} industry.
        
        Business Idea: {idea}
        Target Audience: {audience}
        Design Inspiration: {logo_prompt}
        Industry: {industry}
        
        Requirements:
        - Maximum 8 words
        - Memorable and catchy
        - Reflects brand personality and industry
        - Appeals to target audience
        - Include emotional appeal
        
        Return only the tagline without any explanation.
        """
    
    def _create_ad_copy_prompt(self, brand_name, idea, audience, tagline, industry):
        """Create prompt for ad copy generation"""
        return f"""
        Create engaging social media ad copy for {brand_name} in the {industry} industry.
        
        Brand Tagline: {tagline}
        Business Concept: {idea}
        Target Audience: {audience}
        Industry: {industry}
        
        Format your response as:
        HEADLINE: [catchy headline]
        DESCRIPTION: [compelling description, 2-3 sentences]
        CTA: [single call-to-action phrase]
        HASHTAGS: [3-5 relevant hashtags separated by commas]
        
        Make it engaging, persuasive, and platform-optimized for the {industry} industry.
        """
    
    def _parse_ad_copy(self, ad_copy_text, tagline):
        """Parse LLM-generated ad copy into structured format"""
        try:
            lines = ad_copy_text.split('\n')
            ad_copy = {
                'headline': tagline,
                'description': '',
                'cta': 'Learn More',
                'hashtags': ['Brand', 'New', 'Campaign']
            }
            
            for line in lines:
                if line.startswith('HEADLINE:'):
                    ad_copy['headline'] = line.replace('HEADLINE:', '').strip()
                elif line.startswith('DESCRIPTION:'):
                    ad_copy['description'] = line.replace('DESCRIPTION:', '').strip()
                elif line.startswith('CTA:'):
                    ad_copy['cta'] = line.replace('CTA:', '').strip()
                elif line.startswith('HASHTAGS:'):
                    hashtags_text = line.replace('HASHTAGS:', '').strip()
                    ad_copy['hashtags'] = [tag.strip() for tag in hashtags_text.split(',')]
            
            # Fallbacks if parsing fails
            if not ad_copy['description']:
                ad_copy['description'] = f"Discover {tagline}. Experience quality and innovation in every detail."
            
            return ad_copy
            
        except Exception as e:
            print(f"Ad Copy Parsing Error: {str(e)}")
            return {
                'headline': tagline,
                'description': f"Experience {tagline}. Quality and innovation for modern living.",
                'cta': 'Discover More',
                'hashtags': ['Brand', 'Quality', 'Innovation']
            }
    
    def _generate_color_palette(self, primary_color):
        """Generate complementary color palette"""
        def adjust_color(hex_color, factor):
            hex_color = hex_color.lstrip('#')
            r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
            r = min(255, int(r * factor))
            g = min(255, int(g * factor))
            b = min(255, int(b * factor))
            return '#{:02x}{:02x}{:02x}'.format(r, g, b)
        
        return {
            'primary': primary_color,
            'secondary': adjust_color(primary_color, 0.7),
            'accent': adjust_color(primary_color, 1.3),
            'neutral': '#2C3E50',
            'background': '#FFFFFF'
        }
    
    def _generate_fallback_assets(self, campaign_data):
        """Generate fallback assets when AI generation fails"""
        industry = self.detect_industry(campaign_data['idea'], campaign_data['brand_name'])
        
        return {
            'tagline': "Quality That Inspires",
            'ad_copy': {
                'headline': "Experience Excellence",
                'description': f"Discover {campaign_data['brand_name']}'s commitment to quality and innovation.",
                'cta': 'Learn More',
                'hashtags': ['Quality', 'Innovation', 'Excellence']
            },
            'logo': self.logo_generator._generate_intelligent_fallback_logo(
                campaign_data['brand_name'],
                campaign_data.get('logo_prompt', ''),
                campaign_data.get('style', 'creative'),
                industry
            ),
            'schedule': self.scheduler.generate_optimized_schedule(
                campaign_data['target_audience']
            ),
            'colors': self._generate_color_palette(campaign_data['brand_color']),
            'campaign_data': campaign_data,
            'industry': industry
        }

class CreativePosterCreator:
    def __init__(self):
        self.templates = {
            'creative': self._create_creative_poster,
            'minimal': self._create_minimal_poster,
            'elegant': self._create_elegant_poster,
            'bold': self._create_bold_poster,
            'modern': self._create_modern_poster
        }
    
    def create_poster(self, brand_name, tagline, ad_copy, logo_path, color_palette, style='creative'):
        """Create social media poster"""
        width, height = 1080, 1350
        poster = Image.new('RGB', (width, height), color_palette.get('background', '#FFFFFF'))
        draw = ImageDraw.Draw(poster)
        
        # Apply template
        template_func = self.templates.get(style, self._create_creative_poster)
        template_func(draw, width, height, brand_name, tagline, ad_copy, color_palette)
        
        # Add logo if available
        if logo_path and os.path.exists(logo_path.lstrip('/')):
            try:
                logo = Image.open(logo_path.lstrip('/')).convert('RGBA')
                logo = logo.resize((120, 120))
                poster.paste(logo, (width - 150, 50), logo)
            except Exception as e:
                print(f"Could not add logo to poster: {e}")
        
        # Save poster
        filename = f"poster_{brand_name.replace(' ', '_').lower()}_{int(time.time())}.png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        poster.save(filepath, quality=95)
        
        return {
            'filename': filename,
            'path': f"/static/generated/{filename}",
            'style': style,
            'dimensions': f"{width}x{height}"
        }
    
    def _create_creative_poster(self, draw, width, height, brand_name, tagline, ad_copy, colors):
        """Creative poster template"""
        # Background gradient
        for i in range(height):
            ratio = i / height
            r = int(int(colors['primary'][1:3], 16) * (1 - ratio) + 255 * ratio)
            g = int(int(colors['primary'][3:5], 16) * (1 - ratio) + 255 * ratio)
            b = int(int(colors['primary'][5:7], 16) * (1 - ratio) + 255 * ratio)
            draw.line([(0, i), (width, i)], fill=(r, g, b))
        
        # Content
        self._add_text_centered(draw, width, height//4, brand_name.upper(), colors['neutral'], 48)
        self._add_text_centered(draw, width, height//2, tagline, colors['secondary'], 36)
        self._add_text_centered(draw, width, height*3//4, ad_copy['description'][:100] + "...", colors['neutral'], 24)
        
        # CTA Button
        button_y = height * 4 // 5
        draw.rectangle([width//2-100, button_y-25, width//2+100, button_y+25], 
                      fill=colors['accent'], outline=colors['secondary'], width=2)
        self._add_text_centered(draw, width, button_y, ad_copy['cta'], '#FFFFFF', 20)
    
    def _create_minimal_poster(self, draw, width, height, brand_name, tagline, ad_copy, colors):
        """Minimal poster template"""
        # Clean background
        draw.rectangle([0, 0, width, height], fill=colors['background'])
        
        # Simple border
        draw.rectangle([50, 50, width-50, height-50], outline=colors['primary'], width=3)
        
        # Minimal text
        self._add_text_centered(draw, width, height//3, brand_name, colors['primary'], 42)
        self._add_text_centered(draw, width, height//2, tagline, colors['neutral'], 28)
        self._add_text_centered(draw, width, height*2//3, ad_copy['cta'], colors['accent'], 22)
    
    def _create_elegant_poster(self, draw, width, height, brand_name, tagline, ad_copy, colors):
        """Elegant poster template"""
        # Luxury background
        draw.rectangle([0, 0, width, height], fill=colors['primary'])
        
        # Gold accent line
        draw.rectangle([0, height//3, width, height//3 + 2], fill=colors['accent'])
        
        # Elegant typography
        self._add_text_centered(draw, width, height//4, brand_name.upper(), '#FFFFFF', 52)
        self._add_text_centered(draw, width, height//2, tagline, colors['accent'], 32)
        self._add_text_centered(draw, width, height*3//4, ad_copy['cta'], '#FFFFFF', 24)
    
    def _create_bold_poster(self, draw, width, height, brand_name, tagline, ad_copy, colors):
        """Bold poster template"""
        # Strong background
        draw.rectangle([0, 0, width, height], fill=colors['accent'])
        
        # Contrast elements
        draw.rectangle([0, height//2, width, height//2 + 100], fill=colors['primary'])
        
        # Bold typography
        self._add_text_centered(draw, width, height//3, brand_name.upper(), '#FFFFFF', 56)
        self._add_text_centered(draw, width, height//2 + 50, tagline, colors['accent'], 36)
        self._add_text_centered(draw, width, height*4//5, ad_copy['cta'], '#FFFFFF', 28)
    
    def _create_modern_poster(self, draw, width, height, brand_name, tagline, ad_copy, colors):
        """Modern poster template"""
        # Geometric background
        draw.rectangle([0, 0, width, height], fill=colors['background'])
        
        # Modern shapes
        draw.rectangle([0, 0, width//2, height], fill=colors['primary'])
        draw.ellipse([width//2, height//2, width, height], fill=colors['accent'])
        
        # Modern typography
        self._add_text_centered(draw, width//4, height//3, brand_name, '#FFFFFF', 44)
        self._add_text_centered(draw, width*3//4, height//2, tagline, colors['neutral'], 32)
        self._add_text_centered(draw, width//2, height*4//5, ad_copy['cta'], colors['accent'], 26)
    
    def _add_text_centered(self, draw, width, y, text, color, font_size):
        """Helper to add centered text"""
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2
        
        draw.text((x, y), text, fill=color, font=font)

# Initialize services
campaign_generator = AdvancedCampaignGenerator()
poster_creator = CreativePosterCreator()
logo_generator = AILogoGenerator()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api-status')
def api_status():
    """Check API status"""
    status = {
        'openai': bool(api_config.OPENAI_API_KEY),
        'leonardo_ai': bool(api_config.LEONARDO_API_KEY),
        'huggingface': bool(api_config.HF_API_KEY),
        'message': 'API status checked successfully'
    }
    return jsonify(status)

@app.route('/create-campaign', methods=['GET', 'POST'])
def create_campaign():
    if request.method == 'POST':
        try:
            campaign_data = {
                'idea': request.form.get('campaign_idea', '').strip(),
                'brand_name': request.form.get('brand_name', '').strip(),
                'target_audience': request.form.get('target_audience', '').strip(),
                'style': request.form.get('style', 'creative'),
                'logo_prompt': request.form.get('logo_prompt', '').strip(),
                'poster_prompt': request.form.get('poster_prompt', '').strip(),
                'brand_color': request.form.get('brand_color', '#4A90E2')
            }
            
            # Validate required fields
            if not campaign_data['idea'] or not campaign_data['brand_name']:
                return jsonify({'error': 'Campaign idea and brand name are required'}), 400
            
            print(f"🎯 Generating campaign for: {campaign_data['brand_name']}")
            
            # Generate campaign assets using AI
            campaign_assets = campaign_generator.generate_campaign_assets(campaign_data)
            
            # Create poster
            poster_data = poster_creator.create_poster(
                campaign_data['brand_name'],
                campaign_assets['tagline'],
                campaign_assets['ad_copy'],
                campaign_assets['logo']['path'],
                campaign_assets['colors'],
                campaign_data['style']
            )
            
            # Prepare final result
            result = {
                'ad_copy': campaign_assets['ad_copy'],
                'logo': campaign_assets['logo'],
                'poster': poster_data,
                'tagline': campaign_assets['tagline'],
                'schedule': campaign_assets['schedule'],
                'campaign_data': campaign_data,
                'colors': campaign_assets['colors'],
                'industry': campaign_assets.get('industry', 'general'),
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'api_used': {
                    'text_generation': campaign_generator.llm_generator.available,
                    'logo_generation': campaign_generator.logo_generator.available
                }
            }
            
            print(f"✅ Campaign generated successfully for {campaign_data['brand_name']}")
            return render_template('results.html', result=result)
            
        except Exception as e:
            import traceback
            print(f"❌ Campaign Creation Error: {str(e)}")
            print(traceback.format_exc())
            return jsonify({'error': f'Campaign generation failed: {str(e)}'}), 500
    
    return render_template('campaign.html')

@app.route('/api/redo-logo', methods=['POST'])
def redo_logo():
    """API endpoint to regenerate logo with industry context"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        brand_name = data.get('brand_name')
        logo_prompt = data.get('logo_prompt', 'professional modern logo')
        style = data.get('style', 'creative')
        industry = data.get('industry', 'general')
        
        if not brand_name:
            return jsonify({
                'success': False, 
                'error': 'Brand name is required'
            }), 400
        
        print(f"🔄 Regenerating logo for: {brand_name}")
        print(f"📝 Prompt: {logo_prompt}")
        print(f"🎨 Style: {style}")
        print(f"🏭 Industry: {industry}")
        
        # Generate new logo with industry context - FORCE NEW GENERATION
        logo_data = logo_generator.generate_logo_with_ai(brand_name, logo_prompt, style, industry)
        
        if not logo_data:
            return jsonify({
                'success': False,
                'error': 'Logo generation failed - no data returned'
            }), 500
        
        return jsonify({
            'success': True,
            'logo': logo_data,
            'message': 'Logo regenerated successfully'
        })
        
    except Exception as e:
        import traceback
        print(f"❌ Logo regeneration error: {str(e)}")
        print(f"🔍 Stack trace: {traceback.format_exc()}")
        
        return jsonify({
            'success': False,
            'error': f'Logo regeneration failed: {str(e)}'
        }), 500

@app.route('/api/suggest-colors', methods=['POST'])
def api_suggest_colors():
    """API endpoint for color suggestions"""
    try:
        data = request.json
        industry_colors = {
            'fashion': ['#E74C3C', '#9B59B6', '#3498DB', '#E67E22', '#1ABC9C'],
            'tech': ['#2C3E50', '#3498DB', '#1ABC9C', '#E74C3C', '#9B59B6'],
            'food': ['#E67E22', '#D35400', '#27AE60', '#C0392B', '#F39C12'],
            'health': ['#27AE60', '#2ECC71', '#3498DB', '#9B59B6', '#1ABC9C'],
            'luxury': ['#2C3E50', '#7F8C8D', '#BDC3C7', '#E74C3C', '#34495E']
        }
        
        idea = data.get('idea', '').lower()
        industry = 'fashion'  # Default
        
        if any(word in idea for word in ['tech', 'software', 'digital']):
            industry = 'tech'
        elif any(word in idea for word in ['food', 'restaurant', 'cafe']):
            industry = 'food'
        elif any(word in idea for word in ['health', 'fitness', 'wellness']):
            industry = 'health'
        elif any(word in idea for word in ['luxury', 'premium', 'exclusive']):
            industry = 'luxury'
        
        primary_color = random.choice(industry_colors[industry])
        
        palette = {
            'primary': primary_color,
            'secondary': campaign_generator._generate_color_palette(primary_color)['secondary'],
            'accent': campaign_generator._generate_color_palette(primary_color)['accent'],
            'neutral': '#2C3E50',
            'industry': industry,
            'palette_name': f'{industry.title()} Professional'
        }
        
        return jsonify([palette])
        
    except Exception as e:
        print(f"Color Suggestion Error: {str(e)}")
        fallback = [{
            'primary': '#4A90E2',
            'secondary': '#FF6B6B',
            'accent': '#4ECDC4',
            'neutral': '#2C3E50',
            'industry': 'general',
            'palette_name': 'Professional Blue'
        }]
        return jsonify(fallback)

@app.route('/api/generate-preview', methods=['POST'])
def generate_preview():
    """API endpoint for real-time preview"""
    data = request.json
    llm_generator = LLMTextGenerator()
    
    tagline_prompt = f"Create a short tagline for {data.get('brand_name', 'a brand')} about {data.get('idea', 'quality products')}"
    tagline = llm_generator.generate_with_llm(tagline_prompt, max_tokens=30)
    
    return jsonify({
        'headline': tagline,
        'description': f"Discover {tagline}. Experience quality and innovation.",
        'cta': 'Learn More'
    })

@app.route('/download-poster/<filename>')
def download_poster(filename):
    """Download generated poster"""
    return send_file(
        os.path.join('static', 'generated', filename),
        as_attachment=True
    )

@app.route('/download-logo/<filename>')
def download_logo(filename):
    """Download generated logo"""
    return send_file(
        os.path.join('static', 'generated', filename),
        as_attachment=True
    )

@app.route('/download-schedule')
def download_schedule():
    """Download schedule as JSON"""
    schedule_data = request.args.get('data', '[]')
    schedule = json.loads(schedule_data)
    
    filename = f"campaign_schedule_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    with open(filepath, 'w') as f:
        json.dump(schedule, f, indent=2)
    
    return send_file(filepath, as_attachment=True)

if __name__ == '__main__':
    print("🚀 Starting CampaignIQ Server...")
    print("📋 API Status:")
    api_config.validate_keys()
    app.run(debug=True, host='0.0.0.0', port=5000)