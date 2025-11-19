from flask import Flask, render_template, request, jsonify, send_file
import os
import json
import random
import math
import requests
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import io
import base64
import re
import time
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib
import pickle

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'campaigniq-fashion-secret-2024')
app.config['UPLOAD_FOLDER'] = 'static/generated'
app.config['MODEL_FOLDER'] = 'models'

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

# API Configuration
class APIConfig:
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_BASE_URL = "https://api.openai.com/v1/chat/completions"
    
    @classmethod
    def validate_keys(cls):
        missing_keys = []
        if not cls.OPENAI_API_KEY:
            missing_keys.append("OPENAI_API_KEY")
        
        if missing_keys:
            print(f"Warning: Missing API keys: {', '.join(missing_keys)}")
        else:
            print("All API keys are configured")
        
        return len(missing_keys) == 0

api_config = APIConfig()
api_config.validate_keys()

class LLMTextGenerator:
    def __init__(self):
        self.api_key = api_config.OPENAI_API_KEY
        self.base_url = api_config.OPENAI_BASE_URL
        self.available = bool(self.api_key)
    
    def generate_with_llm(self, prompt, max_tokens=150, temperature=0.7):
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
                    "content": """You are a creative fashion marketing expert specializing in psychedelic, trippy fashion designs. 
                    Create compelling, vibrant, and mind-bending fashion marketing content that resonates with alternative, 
                    artistic, and consciousness-expanding audiences."""
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
            else:
                return self._fallback_generation(prompt)
                
        except Exception as e:
            print(f"OpenAI API Exception: {str(e)}")
            return self._fallback_generation(prompt)
    
    def _fallback_generation(self, prompt):
        prompt_lower = prompt.lower()
        
        if "tagline" in prompt_lower or "psychedelic" in prompt_lower or "trippy" in prompt_lower:
            trippy_taglines = [
                "Expand Your Consciousness Through Fashion",
                "Where Reality Meets Imagination",
                "Trippy Vibes, Elevated Style",
                "Mind-Bending Fashion Journeys",
                "Psychedelic Patterns, Cosmic Style",
                "Fashion That Transcends Dimensions",
                "Wear Your Consciousness"
            ]
            return random.choice(trippy_taglines)
        
        elif "ad copy" in prompt_lower or "description" in prompt_lower:
            trippy_ad_copy = [
                "Dive into a kaleidoscopic world of fashion where colors swirl and patterns dance. Our psychedelic collection transforms ordinary reality into extraordinary visual journeys through vibrant, mind-altering designs.",
                "Experience fashion that transcends the mundane. Each piece is a portal to new dimensions of style, featuring swirling rainbows, cosmic patterns, and consciousness-expanding aesthetics.",
                "Step beyond conventional fashion into a realm of trippy elegance. Our designs blend psychedelic artistry with wearable art, creating pieces that transform your reality with every wear."
            ]
            return random.choice(trippy_ad_copy)
        
        return "Experience fashion that transforms your perspective. Discover psychedelic craftsmanship and mind-expanding design."

class FashionPostingTimePredictor:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_columns = [
            'platform', 'audience_age', 'audience_gender', 'content_type', 
            'brand_style', 'day_of_week', 'hour_of_day', 'season'
        ]
        self.model_path = os.path.join(app.config['MODEL_FOLDER'], 'fashion_posting_model.pkl')
        self.encoders_path = os.path.join(app.config['MODEL_FOLDER'], 'label_encoders.pkl')
        
    def generate_training_data(self):
        np.random.seed(42)
        n_samples = 5000
        
        data = []
        for i in range(n_samples):
            platform = np.random.choice(['instagram', 'facebook', 'tiktok', 'pinterest'], p=[0.4, 0.2, 0.3, 0.1])
            audience_age = np.random.choice(['18-24', '25-34', '35-44', '45-54', '55+'], p=[0.3, 0.4, 0.2, 0.08, 0.02])
            audience_gender = np.random.choice(['male', 'female', 'all'], p=[0.2, 0.7, 0.1])
            content_type = np.random.choice(['product_launch', 'style_inspiration', 'behind_scenes', 'user_content', 'promotion'])
            brand_style = np.random.choice(['psychedelic', 'luxury', 'minimal', 'vintage', 'modern', 'boho', 'streetwear', 'elegant'])
            
            engagement_data = self._calculate_engagement_score(
                platform, audience_age, audience_gender, content_type, brand_style
            )
            
            data.append({
                'platform': platform,
                'audience_age': audience_age,
                'audience_gender': audience_gender,
                'content_type': content_type,
                'brand_style': brand_style,
                'engagement_score': engagement_data['engagement_score'],
                'day_of_week': engagement_data['best_day'],
                'hour_of_day': engagement_data['best_hour'],
                'season': engagement_data['best_season']
            })
        
        return pd.DataFrame(data)
    
    def _calculate_engagement_score(self, platform, audience_age, audience_gender, content_type, brand_style):
        platform_patterns = {
            'instagram': {'best_hours': [9, 12, 17, 19, 21], 'best_days': [2, 3, 4, 5]},
            'facebook': {'best_hours': [9, 13, 18, 20], 'best_days': [3, 4, 5, 6]},
            'tiktok': {'best_hours': [7, 12, 17, 21, 23], 'best_days': [0, 1, 4, 5, 6]},
            'pinterest': {'best_hours': [14, 15, 20, 21], 'best_days': [5, 6]}
        }
        
        age_adjustments = {
            '18-24': {'hour_shift': 2, 'preferred_hours': [17, 18, 19, 20, 21, 22]},
            '25-34': {'hour_shift': 0, 'preferred_hours': [8, 12, 17, 18, 19, 20]},
            '35-44': {'hour_shift': -1, 'preferred_hours': [7, 8, 12, 17, 18, 19]},
            '45-54': {'hour_shift': -2, 'preferred_hours': [7, 8, 12, 17, 18]},
            '55+': {'hour_shift': -3, 'preferred_hours': [7, 8, 12, 13, 17]}
        }
        
        seasonal_preferences = {
            'psychedelic': [1, 2, 3],  # Popular in spring/summer/fall
            'luxury': [3, 0],
            'minimal': [1, 2, 3],
            'vintage': [0, 2],
            'modern': [1, 2],
            'boho': [1, 2],
            'streetwear': [1, 2],
            'elegant': [0, 3]
        }
        
        platform_data = platform_patterns.get(platform, platform_patterns['instagram'])
        age_data = age_adjustments.get(audience_age, age_adjustments['25-34'])
        
        base_hours = platform_data['best_hours']
        adjusted_hours = [(h + age_data['hour_shift']) % 24 for h in base_hours]
        best_hour = np.random.choice(adjusted_hours)
        
        best_day = np.random.choice(platform_data['best_days'])
        best_season = np.random.choice(seasonal_preferences.get(brand_style, [1, 2]))
        
        engagement_base = 60
        engagement_base += len(set(adjusted_hours) & set(age_data['preferred_hours'])) * 5
        engagement_base += random.randint(-10, 15)
        engagement_score = min(100, max(20, engagement_base))
        
        return {
            'engagement_score': engagement_score,
            'best_hour': best_hour,
            'best_day': best_day,
            'best_season': best_season
        }
    
    def train_model(self):
        print("Training XGBoost model for fashion posting times...")
        
        df = self.generate_training_data()
        
        X = df[self.feature_columns].copy()
        y = df['engagement_score']
        
        for column in self.feature_columns:
            if X[column].dtype == 'object':
                self.label_encoders[column] = LabelEncoder()
                X[column] = self.label_encoders[column].fit_transform(X[column].astype(str))
        
        X = X.values
        y = y.values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"Model trained - Train R²: {train_score:.3f}, Test R²: {test_score:.3f}")
        
        joblib.dump(self.model, self.model_path)
        with open(self.encoders_path, 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        return train_score, test_score
    
    def load_model(self):
        try:
            self.model = joblib.load(self.model_path)
            with open(self.encoders_path, 'rb') as f:
                self.label_encoders = pickle.load(f)
            print("XGBoost model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict_best_times(self, campaign_data, platforms=None):
        if self.model is None:
            if not self.load_model():
                print("Training new model...")
                self.train_model()
        
        if platforms is None:
            platforms = ['instagram', 'facebook', 'tiktok', 'pinterest']
        
        predictions = {}
        
        for platform in platforms:
            features = {
                'platform': platform,
                'audience_age': campaign_data.get('target_audience_age', '25-34'),
                'audience_gender': campaign_data.get('target_audience_gender', 'female'),
                'content_type': campaign_data.get('content_type', 'style_inspiration'),
                'brand_style': campaign_data.get('style', 'psychedelic'),
                'day_of_week': 0,
                'hour_of_day': 0,
                'season': self._get_current_season()
            }
            
            best_score = -1
            best_times = []
            
            for day in range(7):
                for hour in range(24):
                    features['day_of_week'] = day
                    features['hour_of_day'] = hour
                    
                    encoded_features = []
                    for col in self.feature_columns:
                        if col in self.label_encoders:
                            try:
                                value_str = str(features[col])
                                encoded_val = self.label_encoders[col].transform([value_str])[0]
                            except ValueError:
                                encoded_val = 0
                        else:
                            encoded_val = features[col]
                        encoded_features.append(encoded_val)
                    
                    encoded_features_array = np.array([encoded_features], dtype=np.float32)
                    
                    engagement_score = self.model.predict(encoded_features_array)[0]
                    
                    if engagement_score > best_score:
                        best_score = engagement_score
                        best_times = [{'day': day, 'hour': hour, 'score': engagement_score}]
                    elif engagement_score == best_score:
                        best_times.append({'day': day, 'hour': hour, 'score': engagement_score})
            
            top_times = sorted(best_times, key=lambda x: x['score'], reverse=True)[:5]
            
            readable_times = []
            for time_slot in top_times:
                day_name = self._get_day_name(time_slot['day'])
                hour_display = self._format_hour(time_slot['hour'])
                readable_times.append({
                    'day': day_name,
                    'hour': hour_display,
                    'score': round(time_slot['score'], 1),
                    'platform': platform
                })
            
            predictions[platform] = {
                'best_times': readable_times,
                'max_engagement': round(best_score, 1)
            }
        
        return predictions
    
    def _get_current_season(self):
        month = datetime.now().month
        if month in [12, 1, 2]:
            return 0
        elif month in [3, 4, 5]:
            return 1
        elif month in [6, 7, 8]:
            return 2
        else:
            return 3
    
    def _get_day_name(self, day_num):
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        return days[day_num]
    
    def _format_hour(self, hour):
        if hour == 0:
            return "12 AM"
        elif hour < 12:
            return f"{hour} AM"
        elif hour == 12:
            return "12 PM"
        else:
            return f"{hour-12} PM"

class FashionLogoGenerator:
    def __init__(self):
        self.llm_generator = LLMTextGenerator()
        self.fashion_styles = {
            'psychedelic': self._create_psychedelic_logo,
            'luxury': self._create_luxury_fashion_logo,
            'minimal': self._create_minimal_fashion_logo,
            'vintage': self._create_vintage_fashion_logo,
            'modern': self._create_modern_fashion_logo,
            'boho': self._create_boho_fashion_logo,
            'streetwear': self._create_streetwear_logo,
            'elegant': self._create_elegant_fashion_logo
        }
        self.logo_variations = {}  # Track variations for regeneration
    
    def generate_fashion_logo(self, brand_name, style="psychedelic", logo_prompt=""):
        print(f"Generating fashion logo for {brand_name} with prompt: {logo_prompt}")
        
        width, height = 400, 400
        
        if logo_prompt:
            style, design_elements = self._analyze_logo_prompt(logo_prompt, style)
        else:
            design_elements = self._get_default_design_elements(style)
        
        # Store variation count for this prompt to ensure different outputs
        prompt_key = f"{brand_name}_{style}_{logo_prompt}"
        if prompt_key not in self.logo_variations:
            self.logo_variations[prompt_key] = 0
        else:
            self.logo_variations[prompt_key] += 1
        
        variation = self.logo_variations[prompt_key] % 3  # Cycle through 3 variations
        
        design_function = self.fashion_styles.get(style, self._create_psychedelic_logo)
        logo, design_name = design_function(width, height, brand_name, design_elements, variation)
        
        design_reason = self._generate_fashion_design_reason(brand_name, style, design_name, logo_prompt, design_elements)
        
        safe_brand_name = re.sub(r'[^\w]', '_', brand_name.lower())
        safe_style = re.sub(r'[^\w]', '_', style.lower())
        timestamp = int(time.time() * 1000) + random.randint(1, 1000)
        filename = f"fashion_logo_{safe_brand_name}_{safe_style}_{timestamp}.png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        logo.save(filepath, 'PNG', quality=95)
        
        print(f"Fashion logo generated: {filename} (Variation: {variation})")
        
        return {
            'filename': filename,
            'path': f"/static/generated/{filename}",
            'style': style,
            'source': 'fashion_designer',
            'description': f'{style.title()} fashion logo for {brand_name}',
            'design_reason': design_reason,
            'timestamp': timestamp,
            'prompt_used': logo_prompt,
            'variation': variation
        }
    
    def _analyze_logo_prompt(self, prompt, default_style):
        prompt_lower = prompt.lower()
        
        style_keywords = {
            'psychedelic': ['psychedelic', 'trippy', 'swirling', 'rainbow', 'mushroom', 'melting', 'kaleidoscope', 'mind', 'consciousness', 'expanding'],
            'luxury': ['luxury', 'premium', 'exclusive', 'high-end', 'elegant', 'sophisticated', 'gold', 'premium', 'crown', 'jewel'],
            'minimal': ['minimal', 'simple', 'clean', 'modern', 'contemporary', 'sleek', 'basic', 'geometric'],
            'vintage': ['vintage', 'retro', 'classic', 'old', 'traditional', 'nostalgic', 'antique', 'ornate'],
            'modern': ['modern', 'contemporary', 'futuristic', 'innovative', 'tech', 'sleek', 'abstract'],
            'boho': ['boho', 'bohemian', 'hippie', 'natural', 'organic', 'flowing', 'earthy', 'mandala'],
            'streetwear': ['street', 'urban', 'edgy', 'bold', 'graphic', 'urban', 'hip-hop', 'graffiti'],
            'elegant': ['elegant', 'classy', 'refined', 'sophisticated', 'graceful', 'chic', 'sophisticated']
        }
        
        detected_style = default_style
        for style, keywords in style_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                detected_style = style
                break
        
        design_elements = {
            'colors': self._extract_colors_from_prompt(prompt),
            'shapes': self._extract_shapes_from_prompt(prompt),
            'elements': self._extract_elements_from_prompt(prompt),
            'mood': self._extract_mood_from_prompt(prompt),
            'specific_elements': self._extract_specific_elements(prompt)
        }
        
        print(f"Prompt analysis - Style: {detected_style}, Elements: {design_elements}")
        return detected_style, design_elements
    
    def _extract_colors_from_prompt(self, prompt):
        color_keywords = {
            'rainbow': ['rainbow', 'colorful', 'vibrant', 'kaleidoscope'],
            'purple': ['purple', 'violet', 'lavender', 'amethyst', 'psychedelic'],
            'pink': ['pink', 'magenta', 'fuchsia', 'hot pink'],
            'blue': ['blue', 'azure', 'cyan', 'turquoise'],
            'green': ['green', 'lime', 'emerald', 'neon'],
            'yellow': ['yellow', 'gold', 'sunshine'],
            'orange': ['orange', 'amber', 'sunset'],
            'red': ['red', 'crimson', 'scarlet']
        }
        
        colors = []
        prompt_lower = prompt.lower()
        
        # Check for rainbow/colorful first
        if any(keyword in prompt_lower for keyword in color_keywords['rainbow']):
            colors = ['rainbow', 'purple', 'pink', 'blue', 'green', 'yellow', 'orange', 'red']
        else:
            for color, keywords in color_keywords.items():
                if any(keyword in prompt_lower for keyword in keywords):
                    colors.append(color)
        
        return colors if colors else ['rainbow', 'purple', 'pink', 'blue']
    
    def _extract_shapes_from_prompt(self, prompt):
        shape_keywords = {
            'swirling': ['swirling', 'spiral', 'vortex', 'whirlpool', 'twisting'],
            'organic': ['organic', 'flowing', 'natural', 'curved', 'fluid', 'melting'],
            'geometric': ['geometric', 'angular', 'sharp', 'modern', 'polygon'],
            'circular': ['circular', 'round', 'circle', 'oval', 'sphere'],
            'complex': ['complex', 'detailed', 'intricate', 'ornate', 'elaborate'],
            'abstract': ['abstract', 'artistic', 'non-representational']
        }
        
        shapes = []
        prompt_lower = prompt.lower()
        for shape, keywords in shape_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                shapes.append(shape)
        
        return shapes if shapes else ['swirling', 'organic', 'abstract']
    
    def _extract_elements_from_prompt(self, prompt):
        element_keywords = {
            'mushroom': ['mushroom', 'fungus', 'psychedelic', 'magic', 'shroom'],
            'spiral': ['spiral', 'vortex', 'swirl', 'whirl'],
            'melting': ['melting', 'dripping', 'liquid', 'flowing'],
            'abstract': ['abstract', 'modern', 'artistic', 'non-figurative'],
            'typography': ['text', 'typography', 'letter', 'font', 'wordmark'],
            'cosmic': ['cosmic', 'galaxy', 'universe', 'stars', 'space'],
            'pattern': ['pattern', 'design', 'motif', 'texture']
        }
        
        elements = []
        prompt_lower = prompt.lower()
        for element, keywords in element_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                elements.append(element)
        
        return elements if elements else ['mushroom', 'spiral', 'melting']
    
    def _extract_specific_elements(self, prompt):
        """Extract very specific design elements from prompt"""
        specific_elements = []
        prompt_lower = prompt.lower()
        
        if 'mushroom' in prompt_lower:
            specific_elements.append('mushroom')
        if 'melting' in prompt_lower or 'dripping' in prompt_lower:
            specific_elements.append('melting')
        if 'spiral' in prompt_lower or 'swirl' in prompt_lower:
            specific_elements.append('spiral')
        if 'rainbow' in prompt_lower:
            specific_elements.append('rainbow')
            
        return specific_elements
    
    def _extract_mood_from_prompt(self, prompt):
        mood_keywords = {
            'trippy': ['trippy', 'psychedelic', 'mind-bending', 'hallucinogenic', 'consciousness'],
            'vibrant': ['vibrant', 'colorful', 'bright', 'lively', 'energetic'],
            'mystical': ['mystical', 'magical', 'spiritual', 'enchanting', 'ethereal'],
            'playful': ['playful', 'fun', 'youthful', 'whimsical'],
            'serious': ['serious', 'professional', 'formal', 'corporate'],
            'creative': ['creative', 'artistic', 'innovative', 'imaginative']
        }
        
        prompt_lower = prompt.lower()
        for mood, keywords in mood_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                return mood
        
        return 'trippy'
    
    def _get_default_design_elements(self, style):
        if style == 'psychedelic':
            return {
                'colors': ['rainbow', 'purple', 'pink', 'blue'],
                'shapes': ['swirling', 'organic', 'abstract'],
                'elements': ['mushroom', 'spiral', 'melting'],
                'mood': 'trippy',
                'specific_elements': ['mushroom', 'rainbow']
            }
        else:
            return {
                'colors': ['gold', 'black'],
                'shapes': ['geometric'],
                'elements': ['abstract'],
                'mood': 'elegant',
                'specific_elements': []
            }
    
    def _generate_fashion_design_reason(self, brand_name, style, design_name, logo_prompt, design_elements):
        if logo_prompt:
            reason_prompt = f"""
            Explain the design rationale for a {style} fashion logo for {brand_name} based on this prompt: "{logo_prompt}".
            
            Design elements used: {design_elements}
            
            Provide a professional design explanation (2-3 sentences) covering:
            - How the design elements reflect the brand identity and prompt requirements
            - The choice of visual elements and their symbolism in fashion context
            - How it appeals to the target fashion audience
            
            Keep it concise and professional.
            """
            
            try:
                reason = self.llm_generator.generate_with_llm(reason_prompt, max_tokens=120)
                return reason
            except:
                pass
        
        reasons = {
            'psychedelic': f"This psychedelic fashion logo for {brand_name} embodies mind-expanding aesthetics through swirling patterns, rainbow colors, and organic mushroom motifs, creating a trippy visual experience that resonates with alternative fashion enthusiasts.",
            'luxury': f"This luxury fashion logo for {brand_name} embodies sophistication through carefully crafted gold elements and elegant composition, conveying exclusivity and high-end craftsmanship suitable for premium fashion collections.",
            'minimal': f"Clean and contemporary, this minimalist logo for {brand_name} uses negative space and geometric precision to create a timeless identity, ideal for modern fashion brands focusing on essential design principles.",
            'vintage': f"Drawing inspiration from classic fashion eras, this vintage logo for {brand_name} incorporates ornate details and traditional motifs, evoking nostalgia and timeless style with historical authenticity.",
            'modern': f"This modern fashion logo for {brand_name} employs bold geometry and dynamic composition to represent contemporary style, perfect for forward-thinking fashion brands embracing innovation.",
            'boho': f"Bohemian-inspired with flowing lines and natural elements, this logo captures the free-spirited essence of {brand_name}, ideal for artisan collections and bohemian fashion aesthetics.",
            'streetwear': f"Urban and edgy, this streetwear logo for {brand_name} features bold typography and graphic elements, representing the dynamic energy of modern urban fashion culture and youth expression.",
            'elegant': f"Sophisticated and refined, this elegant logo for {brand_name} uses delicate lines and balanced composition, perfect for luxury fashion and sophisticated style statements."
        }
        return reasons.get(design_name, reasons['psychedelic'])
    
    def _create_psychedelic_logo(self, width, height, brand_name, design_elements, variation=0):
        logo = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(logo)
        
        colors = self._get_psychedelic_color_scheme(design_elements['colors'])
        center_x, center_y = width // 2, height // 2
        
        # Different variations for psychedelic logos
        if variation == 0:
            # Variation 1: Swirling mushroom with rainbow spiral
            self._draw_psychedelic_mushroom(draw, center_x, center_y, 80, colors, design_elements)
            
        elif variation == 1:
            # Variation 2: Abstract spiral mushroom
            self._draw_abstract_mushroom(draw, center_x, center_y, 70, colors, design_elements)
            
        else:
            # Variation 3: Geometric psychedelic pattern
            self._draw_geometric_psychedelic(draw, center_x, center_y, 75, colors, design_elements)
        
        return logo, "psychedelic"
    
    def _draw_psychedelic_mushroom(self, draw, center_x, center_y, size, colors, design_elements):
        # Draw mushroom cap
        cap_radius = size
        stem_width = size // 4
        stem_height = size
        
        # Rainbow spiral on cap
        spiral_points = []
        for i in range(0, 360, 5):
            angle = math.radians(i)
            spiral_radius = cap_radius * (i / 360)
            x = center_x + spiral_radius * math.cos(angle)
            y = center_y - spiral_radius * math.sin(angle) / 2
            spiral_points.append((x, y))
            
            if len(spiral_points) > 1:
                color_idx = (i // 15) % len(colors['rainbow'])
                draw.line([spiral_points[-2], spiral_points[-1]], 
                         fill=colors['rainbow'][color_idx], width=3)
        
        # Mushroom stem
        stem_points = [
            (center_x - stem_width, center_y),
            (center_x - stem_width//2, center_y + stem_height),
            (center_x + stem_width//2, center_y + stem_height),
            (center_x + stem_width, center_y)
        ]
        draw.polygon(stem_points, fill=colors['primary'])
        
        # Dots on cap (psychedelic pattern)
        for i in range(20):
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(0, cap_radius * 0.8)
            dot_x = center_x + radius * math.cos(angle)
            dot_y = center_y - radius * math.sin(angle) / 2
            dot_size = random.randint(2, 6)
            dot_color = random.choice(colors['rainbow'])
            draw.ellipse([dot_x-dot_size, dot_y-dot_size, dot_x+dot_size, dot_y+dot_size], 
                        fill=dot_color)
    
    def _draw_abstract_mushroom(self, draw, center_x, center_y, size, colors, design_elements):
        # Abstract mushroom with melting effect
        cap_points = []
        for i in range(0, 360, 10):
            angle = math.radians(i)
            base_radius = size
            # Add some random variation for melting effect
            variation = random.uniform(0.8, 1.2)
            radius = base_radius * variation
            
            x = center_x + radius * math.cos(angle)
            y = center_y - radius * math.sin(angle) / 2
            
            cap_points.append((x, y))
        
        # Draw cap with gradient colors
        for i in range(len(cap_points)):
            if i < len(cap_points) - 1:
                color_idx = i % len(colors['rainbow'])
                draw.line([cap_points[i], cap_points[i+1]], 
                         fill=colors['rainbow'][color_idx], width=4)
        
        # Melting stem
        stem_base_y = center_y + size // 2
        stem_drips = []
        for i in range(3):
            drip_x = center_x - size//4 + i * (size//4)
            drip_length = random.randint(size//3, size//2)
            drip_width = random.randint(3, 8)
            
            drip_points = [
                (drip_x, stem_base_y),
                (drip_x + random.randint(-5, 5), stem_base_y + drip_length//2),
                (drip_x + random.randint(-8, 8), stem_base_y + drip_length)
            ]
            
            draw.line(drip_points, fill=colors['primary'], width=drip_width)
            stem_drips.append(drip_points)
    
    def _draw_geometric_psychedelic(self, draw, center_x, center_y, size, colors, design_elements):
        # Geometric psychedelic pattern
        layers = 6
        for layer in range(layers):
            layer_size = size * (1 - layer * 0.15)
            sides = 6 + layer  # Increasing sides for each layer
            
            points = []
            for i in range(sides):
                angle = 2 * math.pi * i / sides
                x = center_x + layer_size * math.cos(angle)
                y = center_y + layer_size * math.sin(angle)
                points.append((x, y))
            
            # Draw polygon with rainbow colors
            if len(points) > 2:
                color_idx = layer % len(colors['rainbow'])
                draw.polygon(points, outline=colors['rainbow'][color_idx], width=3)
            
            # Add spiral lines inside
            if layer > 0:
                inner_points = []
                for i in range(sides * 2):
                    angle = 2 * math.pi * i / (sides * 2)
                    inner_radius = layer_size * 0.7
                    x = center_x + inner_radius * math.cos(angle)
                    y = center_y + inner_radius * math.sin(angle)
                    inner_points.append((x, y))
                    
                    if len(inner_points) > 1:
                        draw.line([inner_points[-2], inner_points[-1]], 
                                 fill=colors['rainbow'][(layer + 1) % len(colors['rainbow'])], 
                                 width=2)
        
        # Central mushroom silhouette
        mushroom_size = size // 3
        draw.ellipse([center_x-mushroom_size, center_y-mushroom_size, 
                     center_x+mushroom_size, center_y+mushroom_size//2], 
                    outline=colors['accent'], width=3)
    
    def _get_psychedelic_color_scheme(self, color_names):
        """Get psychedelic color scheme"""
        rainbow_colors = [
            '#FF0000', '#FF7F00', '#FFFF00', '#00FF00', 
            '#0000FF', '#4B0082', '#8B00FF'
        ]
        
        if 'rainbow' in color_names:
            return {
                'primary': '#8B00FF',  # Purple
                'secondary': '#FF00FF',  # Pink
                'accent': '#00FFFF',  # Cyan
                'rainbow': rainbow_colors,
                'background': '#000000'
            }
        else:
            # Use extracted colors or default psychedelic palette
            color_map = {
                'purple': '#8B00FF',
                'pink': '#FF00FF',
                'blue': '#0000FF',
                'green': '#00FF00',
                'yellow': '#FFFF00',
                'orange': '#FF7F00',
                'red': '#FF0000'
            }
            
            available_colors = [color_map.get(color, '#8B00FF') for color in color_names if color in color_map]
            if not available_colors:
                available_colors = ['#8B00FF', '#FF00FF', '#0000FF']
            
            return {
                'primary': available_colors[0],
                'secondary': available_colors[1] if len(available_colors) > 1 else '#FF00FF',
                'accent': available_colors[2] if len(available_colors) > 2 else '#00FFFF',
                'rainbow': available_colors * 2,  # Repeat if not enough colors
                'background': '#000000'
            }
    
    def _get_custom_color_scheme(self, color_names, style):
        """Get color scheme based on extracted colors or fallback to style defaults"""
        if style == 'psychedelic':
            return self._get_psychedelic_color_scheme(color_names)
            
        color_map = {
            'gold': '#D4AF37',
            'black': '#000000',
            'white': '#FFFFFF',
            'pink': '#E75480',
            'blue': '#3498DB',
            'green': '#27AE60',
            'purple': '#9B59B6',
            'red': '#E74C3C',
            'silver': '#C0C0C0',
            'bronze': '#CD7F32'
        }
        
        # Use extracted colors if available
        if color_names and len(color_names) >= 2:
            primary = color_map.get(color_names[0], '#D4AF37')
            secondary = color_map.get(color_names[1], '#000000')
            accent = color_map.get(color_names[0], '#D4AF37')  # Use first color as accent
        else:
            # Fallback to style defaults
            style_schemes = {
                "psychedelic": self._get_psychedelic_color_scheme([]),
                "luxury": {'primary': '#2C3E50', 'secondary': '#7F8C8D', 'accent': '#D4AF37', 'neutral': '#34495E', 'background': '#FFFFFF'},
                "minimal": {'primary': '#2C3E50', 'secondary': '#95A5A6', 'accent': '#E74C3C', 'neutral': '#34495E', 'background': '#FFFFFF'},
                "vintage": {'primary': '#8B4513', 'secondary': '#D2691E', 'accent': '#CD853F', 'neutral': '#2C3E50', 'background': '#FFFFFF'},
                "modern": {'primary': '#E74C3C', 'secondary': '#3498DB', 'accent': '#9B59B6', 'neutral': '#2C3E50', 'background': '#FFFFFF'},
                "boho": {'primary': '#27AE60', 'secondary': '#F39C12', 'accent': '#8E44AD', 'neutral': '#2C3E50', 'background': '#FFFFFF'},
                "streetwear": {'primary': '#2C3E50', 'secondary': '#E74C3C', 'accent': '#F39C12', 'neutral': '#34495E', 'background': '#FFFFFF'},
                "elegant": {'primary': '#9B59B6', 'secondary': '#E74C3C', 'accent': '#3498DB', 'neutral': '#2C3E50', 'background': '#FFFFFF'}
            }
            return style_schemes.get(style, style_schemes["psychedelic"])
        
        return {
            'primary': primary,
            'secondary': secondary,
            'accent': accent,
            'neutral': '#2C3E50',
            'background': '#FFFFFF'
        }

    def _create_luxury_fashion_logo(self, width, height, brand_name, design_elements, variation=0):
        logo = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(logo)
        colors = self._get_custom_color_scheme(design_elements['colors'], 'luxury')
        center_x, center_y = width // 2, height // 2
        
        # Simple luxury design - crown and jewels
        if variation == 0:
            # Crown design
            crown_points = [
                (center_x-40, center_y-20),
                (center_x-30, center_y-40),
                (center_x-15, center_y-30),
                (center_x, center_y-50),
                (center_x+15, center_y-30),
                (center_x+30, center_y-40),
                (center_x+40, center_y-20)
            ]
            draw.polygon(crown_points, fill=colors['accent'], outline=colors['primary'], width=3)
            
            # Jewels
            for i in range(3):
                jewel_x = center_x - 20 + i * 20
                draw.ellipse([jewel_x-6, center_y-25, jewel_x+6, center_y-13], 
                            fill=colors['secondary'])
            
        return logo, "luxury"

    def _create_minimal_fashion_logo(self, width, height, brand_name, design_elements, variation=0):
        logo = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(logo)
        colors = self._get_custom_color_scheme(design_elements['colors'], 'minimal')
        center_x, center_y = width // 2, height // 2
        
        # Simple geometric design
        draw.rectangle([center_x-30, center_y-30, center_x+30, center_y+30], 
                      outline=colors['primary'], width=3)
        draw.line([(center_x-20, center_y), (center_x+20, center_y)], 
                 fill=colors['accent'], width=2)
        
        return logo, "minimal"

    def _create_vintage_fashion_logo(self, width, height, brand_name, design_elements, variation=0):
        logo = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(logo)
        colors = self._get_custom_color_scheme(design_elements['colors'], 'vintage')
        center_x, center_y = width // 2, height // 2
        
        # Vintage ornate design
        draw.ellipse([center_x-50, center_y-50, center_x+50, center_y+50], 
                    outline=colors['primary'], width=4)
        for i in range(8):
            angle = 2 * math.pi * i / 8
            x = center_x + 35 * math.cos(angle)
            y = center_y + 35 * math.sin(angle)
            draw.ellipse([x-8, y-8, x+8, y+8], outline=colors['secondary'], width=2)
        
        return logo, "vintage"

    def _create_modern_fashion_logo(self, width, height, brand_name, design_elements, variation=0):
        logo = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(logo)
        colors = self._get_custom_color_scheme(design_elements['colors'], 'modern')
        center_x, center_y = width // 2, height // 2
        
        # Modern abstract design
        points = [
            (center_x, center_y-40),
            (center_x+40, center_y),
            (center_x, center_y+40),
            (center_x-40, center_y)
        ]
        draw.polygon(points, outline=colors['primary'], width=4)
        
        return logo, "modern"

    def _create_boho_fashion_logo(self, width, height, brand_name, design_elements, variation=0):
        logo = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(logo)
        colors = self._get_custom_color_scheme(design_elements['colors'], 'boho')
        center_x, center_y = width // 2, height // 2
        
        # Boho mandala design
        for i in range(12):
            angle = 2 * math.pi * i / 12
            start_x = center_x + 20 * math.cos(angle)
            start_y = center_y + 20 * math.sin(angle)
            end_x = center_x + 55 * math.cos(angle)
            end_y = center_y + 55 * math.sin(angle)
            draw.line([(start_x, start_y), (end_x, end_y)], 
                     fill=colors['primary'], width=3)
        
        return logo, "boho"

    def _create_streetwear_logo(self, width, height, brand_name, design_elements, variation=0):
        logo = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(logo)
        colors = self._get_custom_color_scheme(design_elements['colors'], 'streetwear')
        center_x, center_y = width // 2, height // 2
        
        # Streetwear urban design
        draw.rectangle([center_x-45, center_y-45, center_x+45, center_y+45], 
                      outline=colors['primary'], width=5)
        draw.line([(center_x-40, center_y-40), (center_x+40, center_y+40)], 
                 fill=colors['accent'], width=4)
        
        return logo, "streetwear"

    def _create_elegant_fashion_logo(self, width, height, brand_name, design_elements, variation=0):
        logo = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(logo)
        colors = self._get_custom_color_scheme(design_elements['colors'], 'elegant')
        center_x, center_y = width // 2, height // 2
        
        # Elegant swan design
        draw.ellipse([center_x-35, center_y-20, center_x-5, center_y+10], 
                    outline=colors['primary'], width=3)
        draw.ellipse([center_x+5, center_y-35, center_x+35, center_y-5], 
                    outline=colors['primary'], width=3)
        
        return logo, "elegant"

class FashionPosterCreator:
    def __init__(self):
        self.templates = {
            'psychedelic': self._create_psychedelic_poster,
            'luxury': self._create_luxury_poster,
            'minimal': self._create_minimal_poster,
            'vintage': self._create_vintage_poster,
            'modern': self._create_modern_poster,
            'boho': self._create_boho_poster,
            'streetwear': self._create_streetwear_poster,
            'elegant': self._create_elegant_poster
        }
    
    def create_fashion_poster(self, brand_name, tagline, ad_copy, logo_path, color_palette, style='psychedelic', custom_text=None, custom_colors=None):
        width, height = 1080, 1350
        
        # Use custom colors if provided
        effective_colors = custom_colors if custom_colors else color_palette
        
        poster = Image.new('RGB', (width, height), effective_colors.get('background', '#000000'))
        draw = ImageDraw.Draw(poster)
        
        template_func = self.templates.get(style, self._create_psychedelic_poster)
        template_func(draw, width, height, brand_name, tagline, ad_copy, effective_colors, custom_text)
        
        # Enhanced logo integration
        if logo_path:
            actual_logo_path = logo_path.lstrip('/')
            if not os.path.exists(actual_logo_path):
                actual_logo_path = os.path.join('static', 'generated', os.path.basename(logo_path))
            
            if os.path.exists(actual_logo_path):
                try:
                    logo = Image.open(actual_logo_path).convert('RGBA')
                    logo_size = 180
                    logo = logo.resize((logo_size, logo_size), Image.Resampling.LANCZOS)
                    
                    logo_x = (width - logo_size) // 2
                    logo_y = 80
                    
                    # Add glow effect for psychedelic style
                    if style == 'psychedelic':
                        glow_size = logo_size + 30
                        glow = Image.new('RGBA', (glow_size, glow_size), (0, 0, 0, 0))
                        glow_draw = ImageDraw.Draw(glow)
                        for i in range(5, 0, -1):
                            radius = glow_size // 2 - i * 3
                            alpha = 100 - i * 20
                            glow_draw.ellipse([glow_size//2 - radius, glow_size//2 - radius,
                                             glow_size//2 + radius, glow_size//2 + radius],
                                            outline=effective_colors['accent'] + hex(alpha)[2:].zfill(2))
                        poster.paste(glow, (logo_x-15, logo_y-15), glow)
                    
                    poster.paste(logo, (logo_x, logo_y), logo)
                    
                except Exception as e:
                    print(f"Could not add logo to fashion poster: {e}")
        
        safe_brand_name = re.sub(r'[^\w]', '_', brand_name.lower())
        safe_style = re.sub(r'[^\w]', '_', style.lower())
        timestamp = int(time.time() * 1000) + random.randint(1, 1000)
        filename = f"fashion_poster_{safe_brand_name}_{safe_style}_{timestamp}.png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        poster.save(filepath, quality=95)
        
        return {
            'filename': filename,
            'path': f"/static/generated/{filename}",
            'style': style,
            'dimensions': f"{width}x{height}",
            'customized': custom_text is not None or custom_colors is not None
        }
    
    def _create_psychedelic_poster(self, draw, width, height, brand_name, tagline, ad_copy, colors, custom_text):
        # Create psychedelic background with swirling patterns
        self._draw_psychedelic_background(draw, width, height, colors)
        
        text_content = custom_text if custom_text else {
            'brand': brand_name.upper(),
            'tagline': tagline,
            'description': ad_copy.get('description', 'Experience mind-expanding fashion')[:120] + "...",
            'cta': ad_copy.get('cta', 'Expand Your Consciousness')
        }
        
        # Draw central mushroom graphic
        self._draw_poster_mushroom(draw, width//2, height//2 - 50, 120, colors)
        
        # Add melting typography
        self._draw_melting_text(draw, width//2, height//3, text_content['brand'], colors['rainbow'][2], 64)
        self._draw_melting_text(draw, width//2, height//2 + 100, f'"{text_content["tagline"]}"', colors['rainbow'][4], 36)
        
        # Add description with glow effect
        self._draw_glowing_text(draw, width//2, height*3//4, text_content['description'], '#FFFFFF', 20)
        
        # Psychedelic CTA button
        button_y = height * 4 // 5
        self._draw_psychedelic_button(draw, width//2, button_y, text_content['cta'], colors)
    
    def _draw_psychedelic_background(self, draw, width, height, colors):
        # Create swirling rainbow background
        for y in range(0, height, 2):
            for x in range(0, width, 2):
                # Calculate color based on position and time-like function
                dx = x - width/2
                dy = y - height/2
                distance = math.sqrt(dx*dx + dy*dy)
                angle = math.atan2(dy, dx)
                
                # Create swirling pattern
                swirl = math.sin(distance * 0.02 + angle * 5) * 50
                color_index = int((distance + swirl) / 20) % len(colors['rainbow'])
                
                draw.point((x, y), fill=colors['rainbow'][color_index])
        
        # Add overlay for better text readability
        overlay = Image.new('RGBA', (width, height), (0, 0, 0, 128))
        draw.rectangle([0, 0, width, height], fill=(0, 0, 0, 128))
    
    def _draw_poster_mushroom(self, draw, center_x, center_y, size, colors):
        # Draw a prominent psychedelic mushroom
        cap_radius = size
        stem_width = size // 3
        stem_height = size
        
        # Rainbow spiral cap
        for i in range(0, 360, 2):
            angle = math.radians(i)
            spiral_radius = cap_radius * (i / 360)
            x = center_x + spiral_radius * math.cos(angle)
            y = center_y - spiral_radius * math.sin(angle) / 2
            
            if i > 0:
                prev_angle = math.radians(i-2)
                prev_radius = cap_radius * ((i-2) / 360)
                prev_x = center_x + prev_radius * math.cos(prev_angle)
                prev_y = center_y - prev_radius * math.sin(prev_angle) / 2
                
                color_idx = (i // 15) % len(colors['rainbow'])
                draw.line([(prev_x, prev_y), (x, y)], 
                         fill=colors['rainbow'][color_idx], width=4)
        
        # Stem with melting effect
        stem_points = [
            (center_x - stem_width, center_y),
            (center_x - stem_width//2, center_y + stem_height),
            (center_x + stem_width//2, center_y + stem_height),
            (center_x + stem_width, center_y)
        ]
        draw.polygon(stem_points, fill=colors['primary'])
        
        # Add dripping effect
        for i in range(3):
            drip_x = center_x - stem_width//2 + i * (stem_width//2)
            drip_length = random.randint(20, 40)
            drip_points = [
                (drip_x, center_y + stem_height),
                (drip_x + random.randint(-5, 5), center_y + stem_height + drip_length),
                (drip_x + random.randint(-8, 8), center_y + stem_height + drip_length + 10)
            ]
            draw.line(drip_points, fill=colors['primary'], width=4)
    
    def _draw_melting_text(self, draw, x, y, text, color, font_size):
        try:
            font = ImageFont.truetype("arialbd.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Draw main text
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        text_x = x - text_width // 2
        text_y = y - text_height // 2
        
        # Draw text with shadow
        draw.text((text_x+2, text_y+2), text, fill='#000000', font=font)
        draw.text((text_x, text_y), text, fill=color, font=font)
        
        # Add melting/dripping effect
        melt_height = text_height // 2
        for melt_x in range(text_x, text_x + text_width, 5):
            melt_length = random.randint(melt_height // 2, melt_height)
            melt_points = []
            
            for i in range(melt_length):
                melt_y = text_y + text_height + i
                wave = math.sin(melt_x * 0.1 + i * 0.2) * 3
                melt_points.append((melt_x + wave, melt_y))
            
            if len(melt_points) > 1:
                draw.line(melt_points, fill=color, width=2)
    
    def _draw_glowing_text(self, draw, x, y, text, color, font_size):
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        
        text_x = x - text_width // 2
        
        # Draw glow effect
        for i in range(3, 0, -1):
            glow_color = color if i == 1 else '#000000'
            glow_alpha = 200 - i * 50
            for offset_x, offset_y in [(-i,-i), (-i,i), (i,-i), (i,i)]:
                draw.text((text_x+offset_x, y+offset_y), text, fill=glow_color, font=font)
        
        draw.text((text_x, y), text, fill=color, font=font)
    
    def _draw_psychedelic_button(self, draw, x, y, text, colors):
        button_width, button_height = 240, 60
        
        # Draw pulsating button background
        for i in range(5):
            radius_x = button_width // 2 + i * 5
            radius_y = button_height // 2 + i * 3
            color_idx = i % len(colors['rainbow'])
            draw.ellipse([x-radius_x, y-radius_y, x+radius_x, y+radius_y], 
                        outline=colors['rainbow'][color_idx], width=2)
        
        # Main button
        draw.rectangle([x-button_width//2, y-button_height//2, 
                       x+button_width//2, y+button_height//2], 
                      fill=colors['primary'])
        
        # Button text
        try:
            font = ImageFont.truetype("arialbd.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = x - text_width // 2
        text_y = y - 10
        
        draw.text((text_x, text_y), text, fill='#FFFFFF', font=font)

    def _create_luxury_poster(self, draw, width, height, brand_name, tagline, ad_copy, colors, custom_text):
        # Luxury poster with gold accents
        draw.rectangle([0, 0, width, height], fill=colors['background'])
        
        # Gold gradient background
        for i in range(height):
            ratio = i / height
            r = int(212 * (1 - ratio) + 45 * ratio)
            g = int(175 * (1 - ratio) + 52 * ratio)
            b = int(55 * (1 - ratio) + 54 * ratio)
            draw.line([(0, i), (width, i)], fill=(r, g, b))
        
        text_content = custom_text if custom_text else {
            'brand': brand_name.upper(),
            'tagline': tagline,
            'description': ad_copy.get('description', 'Luxury fashion experience')[:100] + "...",
            'cta': ad_copy.get('cta', 'Discover Luxury')
        }
        
        # Luxury text styling
        self._add_fashion_text(draw, width//2, height//3, text_content['brand'], '#FFFFFF', 48, 'bold')
        self._add_fashion_text(draw, width//2, height//2, text_content['tagline'], colors['accent'], 32, 'italic')
        self._add_fashion_text(draw, width//2, height*2//3, text_content['description'], '#E8E8E8', 20, 'regular')
        
        # Luxury button
        button_y = height * 4 // 5
        draw.rectangle([width//2-120, button_y-25, width//2+120, button_y+25], 
                      fill=colors['accent'], outline='#FFFFFF', width=2)
        self._add_fashion_text(draw, width//2, button_y, text_content['cta'], '#FFFFFF', 20, 'bold')

    def _create_minimal_poster(self, draw, width, height, brand_name, tagline, ad_copy, colors, custom_text):
        # Minimal poster with clean design
        draw.rectangle([0, 0, width, height], fill=colors['background'])
        
        text_content = custom_text if custom_text else {
            'brand': brand_name,
            'tagline': tagline,
            'cta': ad_copy.get('cta', 'Explore')
        }
        
        # Minimal geometric elements
        draw.rectangle([width//2-150, height//3-1, width//2+150, height//3+1], fill=colors['primary'])
        draw.rectangle([width//2-1, height//3-50, width//2+1, height//3+50], fill=colors['primary'])
        
        self._add_fashion_text(draw, width//2, height//3-80, text_content['brand'], colors['primary'], 44, 'bold')
        self._add_fashion_text(draw, width//2, height//3+30, text_content['tagline'], colors['neutral'], 28, 'regular')
        
        # Simple CTA
        draw.rectangle([width//2-80, height*2//3, width//2+80, height*2//3+2], fill=colors['accent'])
        self._add_fashion_text(draw, width//2, height*2//3+20, text_content['cta'], colors['accent'], 26, 'bold')

    def _create_vintage_poster(self, draw, width, height, brand_name, tagline, ad_copy, colors, custom_text):
        # Vintage poster with sepia tones
        base_color = (245, 235, 220)
        draw.rectangle([0, 0, width, height], fill=base_color)
        
        text_content = custom_text if custom_text else {
            'brand': brand_name,
            'tagline': tagline,
            'cta': ad_copy.get('cta', 'Discover Vintage')
        }
        
        # Vintage border
        draw.rectangle([30, 30, width-30, height-30], outline=colors['primary'], width=4)
        
        self._add_fashion_text(draw, width//2, height//4, text_content['brand'], colors['primary'], 52, 'bold')
        self._add_fashion_text(draw, width//2, height//2, text_content['tagline'], colors['secondary'], 30, 'italic')
        self._add_fashion_text(draw, width//2, height*3//4, text_content['cta'], colors['accent'], 26, 'bold')

    def _create_modern_poster(self, draw, width, height, brand_name, tagline, ad_copy, colors, custom_text):
        # Modern poster with geometric design
        draw.rectangle([0, 0, width, height], fill=colors['background'])
        
        # Color blocks
        draw.rectangle([0, 0, width//2, height//2], fill=colors['primary'] + '80')
        draw.rectangle([width//2, height//2, width, height], fill=colors['accent'] + '80')
        
        text_content = custom_text if custom_text else {
            'brand': brand_name,
            'tagline': tagline,
            'cta': ad_copy.get('cta', 'Explore Modern')
        }
        
        self._add_fashion_text(draw, width//4, height//3, text_content['brand'], colors['primary'], 44, 'bold')
        self._add_fashion_text(draw, width*3//4, height//2, text_content['tagline'], colors['neutral'], 30, 'regular')
        
        # Modern CTA
        cta_bg_y = height*4//5
        draw.rectangle([width//2-120, cta_bg_y-30, width//2+120, cta_bg_y+30], 
                      fill=colors['accent'])
        self._add_fashion_text(draw, width//2, cta_bg_y, text_content['cta'], '#FFFFFF', 24, 'bold')

    def _create_boho_poster(self, draw, width, height, brand_name, tagline, ad_copy, colors, custom_text):
        # Boho poster with earthy tones
        for i in range(height):
            ratio = i / height
            r = int(139 * (1 - ratio) + 245 * ratio)
            g = int(69 * (1 - ratio) + 222 * ratio)
            b = int(19 * (1 - ratio) + 173 * ratio)
            draw.line([(0, i), (width, i)], fill=(r, g, b))
        
        text_content = custom_text if custom_text else {
            'brand': brand_name,
            'tagline': tagline,
            'cta': ad_copy.get('cta', 'Explore Boho')
        }
        
        # Boho pattern elements
        pattern_spacing = 80
        for i in range(0, width, pattern_spacing):
            for j in range(0, height, pattern_spacing):
                if (i + j) % (pattern_spacing * 2) == 0:
                    draw.ellipse([i-8, j-8, i+8, j+8], 
                                outline=colors['accent'], width=2)
        
        self._add_fashion_text(draw, width//2, height//3, text_content['brand'], colors['neutral'], 46, 'bold')
        self._add_fashion_text(draw, width//2, height//2, text_content['tagline'], colors['accent'], 32, 'italic')
        self._add_fashion_text(draw, width//2, height*2//3, text_content['cta'], colors['secondary'], 24, 'bold')

    def _create_streetwear_poster(self, draw, width, height, brand_name, tagline, ad_copy, colors, custom_text):
        # Streetwear poster with urban vibe
        draw.rectangle([0, 0, width, height], fill=colors['neutral'])
        
        # Urban grid
        for i in range(5):
            y_pos = height//4 + i * 40
            draw.line([(50, y_pos), (width-50, y_pos)], 
                     fill=colors['accent'], width=3)
        
        text_content = custom_text if custom_text else {
            'brand': brand_name.upper(),
            'tagline': tagline,
            'cta': ad_copy.get('cta', 'Explore Streetwear')
        }
        
        self._add_fashion_text(draw, width//2, height//3, text_content['brand'], colors['accent'], 56, 'bold')
        self._add_fashion_text(draw, width//2, height//2, text_content['tagline'], '#FFFFFF', 36, 'regular')
        self._add_fashion_text(draw, width//2, height*3//4, text_content['cta'], colors['secondary'], 28, 'bold')

    def _create_elegant_poster(self, draw, width, height, brand_name, tagline, ad_copy, colors, custom_text):
        # Elegant poster with soft gradient
        for i in range(height):
            ratio = i / height
            r = int(255 * (1 - ratio) + 245 * ratio)
            g = int(255 * (1 - ratio) + 230 * ratio)
            b = int(255 * (1 - ratio) + 250 * ratio)
            draw.line([(0, i), (width, i)], fill=(r, g, b))
        
        text_content = custom_text if custom_text else {
            'brand': brand_name,
            'tagline': tagline,
            'description': ad_copy.get('description', 'Elegant fashion experience')[:100] + "...",
            'cta': ad_copy.get('cta', 'Discover Elegance')
        }
        
        # Decorative border
        draw.rectangle([40, 40, width-40, height-40], outline=colors['primary'], width=2)
        
        self._add_fashion_text(draw, width//2, height//3, text_content['brand'], colors['primary'], 48, 'bold')
        self._add_fashion_text(draw, width//2, height//2, text_content['tagline'], colors['neutral'], 32, 'italic')
        self._add_fashion_text(draw, width//2, height*2//3, text_content['description'], colors['neutral'], 20, 'regular')
        
        # Elegant button
        button_y = height * 4 // 5
        draw.rectangle([width//2-100, button_y-25, width//2+100, button_y+25], 
                      fill=colors['accent'], outline=colors['primary'], width=2)
        self._add_fashion_text(draw, width//2, button_y, text_content['cta'], '#FFFFFF', 20, 'bold')

    def _add_fashion_text(self, draw, x, y, text, color, font_size, weight='regular'):
        try:
            if weight == 'bold':
                font = ImageFont.truetype("arialbd.ttf", font_size)
            elif weight == 'italic':
                font = ImageFont.truetype("ariali.ttf", font_size)
            else:
                font = ImageFont.truetype("arial.ttf", font_size)
        except:
            # Fallback to default font
            font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        
        if isinstance(x, int):
            x_pos = (x - text_width) // 2
        else:
            x_pos = x
        
        draw.text((x_pos, y), text, fill=color, font=font)

class FashionCampaignGenerator:
    def __init__(self):
        self.llm_generator = LLMTextGenerator()
        self.logo_generator = FashionLogoGenerator()
        self.poster_creator = FashionPosterCreator()
        self.posting_predictor = FashionPostingTimePredictor()
    
    def generate_fashion_campaign(self, campaign_data):
        try:
            # Generate fashion logo with user prompt
            logo_data = self.logo_generator.generate_fashion_logo(
                campaign_data['brand_name'],
                campaign_data.get('style', 'psychedelic'),
                campaign_data.get('logo_prompt', '')
            )
            
            # Generate tagline and ad copy using LLM
            tagline_prompt = f"""
            Create a compelling psychedelic fashion brand tagline for {campaign_data['brand_name']}.
            
            Campaign Idea: {campaign_data['idea']}
            Target Audience: {campaign_data['target_audience']}
            
            Requirements:
            - Maximum 6 words
            - Mind-expanding and trippy
            - Reflects psychedelic, consciousness-expanding style
            - Appeals to {campaign_data['target_audience']}
            
            Return only the tagline.
            """
            
            tagline = self.llm_generator.generate_with_llm(tagline_prompt, max_tokens=50)
            
            ad_copy_prompt = f"""
            Create engaging social media ad copy for {campaign_data['brand_name']} psychedelic fashion brand.
            
            Style: {campaign_data.get('style', 'psychedelic')}
            Tagline: {tagline}
            Campaign: {campaign_data['idea']}
            
            Format as:
            HEADLINE: [trippy, mind-bending headline]
            DESCRIPTION: [psychedelic description, 2 sentences about consciousness-expanding fashion]
            CTA: [action-oriented trippy phrase]
            HASHTAGS: [3-4 relevant psychedelic fashion hashtags]
            """
            
            ad_copy_text = self.llm_generator.generate_with_llm(ad_copy_prompt, max_tokens=200)
            ad_copy = self._parse_ad_copy(ad_copy_text, tagline)
            
            # Generate color palette
            colors = self._generate_psychedelic_color_palette(campaign_data.get('brand_color', '#8B00FF'))
            
            # Create fashion poster
            poster_data = self.poster_creator.create_fashion_poster(
                campaign_data['brand_name'],
                tagline,
                ad_copy,
                logo_data['path'],
                colors,
                campaign_data.get('style', 'psychedelic')
            )
            
            # Generate posting schedule using XGBoost
            posting_schedule = self.posting_predictor.predict_best_times(campaign_data)
            
            return {
                'tagline': tagline,
                'ad_copy': ad_copy,
                'logo': logo_data,
                'poster': poster_data,
                'colors': colors,
                'campaign_data': campaign_data,
                'posting_schedule': posting_schedule,
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            print(f"Fashion Campaign Generation Error: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return self._generate_fallback_campaign(campaign_data)
    
    def _parse_ad_copy(self, ad_copy_text, tagline):
        try:
            lines = ad_copy_text.split('\n')
            ad_copy = {
                'headline': tagline,
                'description': f"Discover {tagline}. Experience mind-expanding fashion and consciousness-altering design.",
                'cta': 'Expand Your Reality',
                'hashtags': ['PsychedelicFashion', 'TrippyStyle', 'ConsciousnessExpansion']
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
            
            return ad_copy
            
        except Exception as e:
            print(f"Ad Copy Parsing Error: {str(e)}")
            return {
                'headline': tagline,
                'description': f"Discover {tagline}. Experience mind-expanding fashion and consciousness-altering design.",
                'cta': 'Expand Your Reality',
                'hashtags': ['PsychedelicFashion', 'TrippyStyle', 'ConsciousnessExpansion']
            }
    
    def _generate_psychedelic_color_palette(self, primary_color):
        rainbow_colors = [
            '#FF0000', '#FF7F00', '#FFFF00', '#00FF00', 
            '#0000FF', '#4B0082', '#8B00FF'
        ]
        
        return {
            'primary': primary_color,
            'secondary': '#FF00FF',
            'accent': '#00FFFF',
            'rainbow': rainbow_colors,
            'neutral': '#2C3E50',
            'background': '#000000'
        }
    
    def _generate_fallback_campaign(self, campaign_data):
        return {
            'tagline': "Expand Your Consciousness",
            'ad_copy': {
                'headline': "Expand Your Consciousness",
                'description': "Discover our mind-bending collection of psychedelic fashion pieces designed for reality expansion.",
                'cta': 'Journey Beyond',
                'hashtags': ['PsychedelicFashion', 'TrippyStyle', 'MindExpansion']
            },
            'logo': self.logo_generator.generate_fashion_logo(
                campaign_data['brand_name'],
                'psychedelic',
                campaign_data.get('logo_prompt', '')
            ),
            'poster': {'filename': '', 'path': '', 'style': 'psychedelic', 'dimensions': '1080x1350'},
            'colors': self._generate_psychedelic_color_palette(campaign_data.get('brand_color', '#8B00FF')),
            'campaign_data': campaign_data,
            'posting_schedule': {},
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

# Initialize fashion services
fashion_generator = FashionCampaignGenerator()

# Train or load the XGBoost model on startup
print("Initializing Psychedelic Fashion CampaignIQ AI Services...")
print("Specialized in Psychedelic & Trippy Fashion Designs")
fashion_generator.posting_predictor.train_model()

@app.route('/')
def index():
    return render_template('fashion_index.html')

@app.route('/create-campaign', methods=['GET', 'POST'])
def create_campaign():
    if request.method == 'POST':
        try:
            campaign_data = {
                'idea': request.form.get('campaign_idea', '').strip(),
                'brand_name': request.form.get('brand_name', '').strip(),
                'target_audience': request.form.get('target_audience', '').strip(),
                'target_audience_age': request.form.get('target_audience_age', '18-24'),
                'target_audience_gender': request.form.get('target_audience_gender', 'all'),
                'content_type': request.form.get('content_type', 'style_inspiration'),
                'style': request.form.get('style', 'psychedelic'),
                'logo_prompt': request.form.get('logo_prompt', '').strip(),
                'poster_prompt': request.form.get('poster_prompt', '').strip(),
                'brand_color': request.form.get('brand_color', '#8B00FF')
            }
            
            if not campaign_data['idea'] or not campaign_data['brand_name']:
                return jsonify({'error': 'Campaign idea and brand name are required'}), 400
            
            print(f"Generating psychedelic fashion campaign for: {campaign_data['brand_name']}")
            
            # Generate fashion campaign
            campaign_result = fashion_generator.generate_fashion_campaign(campaign_data)
            
            return render_template('fashion_results.html', result=campaign_result)
            
        except Exception as e:
            import traceback
            print(f"Fashion Campaign Creation Error: {str(e)}")
            print(traceback.format_exc())
            return jsonify({'error': f'Campaign generation failed: {str(e)}'}), 500
    
    return render_template('fashion_campaign.html')

@app.route('/api/redo-logo', methods=['POST'])
def redo_logo():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        brand_name = data.get('brand_name')
        style = data.get('style', 'psychedelic')
        logo_prompt = data.get('logo_prompt', '')
        
        if not brand_name:
            return jsonify({
                'success': False, 
                'error': 'Brand name is required'
            }), 400
        
        print(f"REGENERATING psychedelic fashion logo for: {brand_name} with prompt: {logo_prompt}")
        
        logo_data = fashion_generator.logo_generator.generate_fashion_logo(
            brand_name, style, logo_prompt
        )
        
        return jsonify({
            'success': True,
            'logo': logo_data,
            'message': 'Psychedelic fashion logo regenerated successfully'
        })
        
    except Exception as e:
        import traceback
        print(f"Logo regeneration error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Logo regeneration failed: {str(e)}'
        }), 500

@app.route('/api/update-poster', methods=['POST'])
def update_poster():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        brand_name = data.get('brand_name')
        tagline = data.get('tagline', '')
        ad_copy = data.get('ad_copy', {})
        logo_path = data.get('logo_path', '')
        colors = data.get('colors', {})
        style = data.get('style', 'psychedelic')
        custom_text = data.get('custom_text')
        custom_colors = data.get('custom_colors')
        
        if not brand_name:
            return jsonify({
                'success': False, 
                'error': 'Brand name is required'
            }), 400
        
        print(f"UPDATING psychedelic poster for: {brand_name}")
        
        poster_data = fashion_generator.poster_creator.create_fashion_poster(
            brand_name,
            tagline,
            ad_copy,
            logo_path,
            colors,
            style,
            custom_text,
            custom_colors
        )
        
        return jsonify({
            'success': True,
            'poster': poster_data,
            'message': 'Psychedelic poster updated successfully'
        })
        
    except Exception as e:
        import traceback
        print(f"Poster update error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Poster update failed: {str(e)}'
        }), 500

@app.route('/api/predict-posting-times', methods=['POST'])
def predict_posting_times():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        campaign_data = {
            'target_audience_age': data.get('target_audience_age', '18-24'),
            'target_audience_gender': data.get('target_audience_gender', 'all'),
            'content_type': data.get('content_type', 'style_inspiration'),
            'style': data.get('style', 'psychedelic')
        }
        
        platforms = data.get('platforms', ['instagram', 'facebook', 'tiktok', 'pinterest'])
        
        print(f"Predicting posting times for psychedelic fashion: {campaign_data}")
        
        posting_schedule = fashion_generator.posting_predictor.predict_best_times(
            campaign_data, platforms
        )
        
        return jsonify({
            'success': True,
            'posting_schedule': posting_schedule,
            'message': 'Posting times predicted successfully'
        })
        
    except Exception as e:
        import traceback
        print(f"Posting time prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Posting time prediction failed: {str(e)}'
        }), 500

@app.route('/download-poster/<filename>')
def download_poster(filename):
    return send_file(
        os.path.join('static', 'generated', filename),
        as_attachment=True
    )

@app.route('/download-logo/<filename>')
def download_logo(filename):
    return send_file(
        os.path.join('static', 'generated', filename),
        as_attachment=True
    )

if __name__ == '__main__':
    print("Starting Psychedelic Fashion CampaignIQ Server...")
    print("Specialized in Psychedelic & Trippy Fashion Designs")
    print("XGBoost AI Posting Time Prediction Enabled")
    app.run(debug=True, host='0.0.0.0', port=5000)