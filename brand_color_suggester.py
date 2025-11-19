import random
import colorsys
import numpy as np
from sklearn.cluster import KMeans

class BrandColorSuggester:
    def __init__(self):
        self.color_palettes = {
            'fashion': [
                ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFE66D'],  # Vibrant
                ['#2C3E50', '#E74C3C', '#ECF0F1', '#3498DB'],  # Bold
                ['#F8B195', '#F67280', '#C06C84', '#6C5B7B']   # Romantic
            ],
            'accessories': [
                ['#1A1A1D', '#4E4E50', '#C3073F', '#950740'],  # Luxury
                ['#222831', '#393E46', '#00ADB5', '#EEEEEE'],  # Modern
                ['#2D4059', '#EA5455', '#F07B3F', '#FFD460']   # Energetic
            ],
            'lifestyle': [
                ['#556270', '#4ECDC4', '#C7F464', '#FF6B6B'],  # Creative
                ['#5D4157', '#838689', '#A8CABA', '#CAD7B2'],  # Natural
                ['#34314C', '#47B8E0', '#FF7473', '#FFC952']   # Playful
            ]
        }
        
        self.industry_colors = {
            'fashion': ['#E74C3C', '#9B59B6', '#3498DB', '#E67E22'],
            'tech': ['#2C3E50', '#3498DB', '#1ABC9C', '#E74C3C'],
            'food': ['#E67E22', '#D35400', '#27AE60', '#C0392B'],
            'health': ['#27AE60', '#2ECC71', '#3498DB', '#9B59B6'],
            'luxury': ['#2C3E50', '#7F8C8D', '#BDC3C7', '#E74C3C']
        }
    
    def suggest_colors(self, campaign_idea, brand_name, industry=None):
        """Suggest brand colors based on campaign context"""
        industry = industry or self._detect_industry(campaign_idea)
        
        # Base palette based on industry
        base_palette = self.industry_colors.get(industry, self.industry_colors['fashion'])
        
        # Generate complementary colors
        suggestions = self._generate_color_schemes(base_palette)
        
        return {
            'primary': suggestions[0],
            'secondary': suggestions[1],
            'accent': suggestions[2],
            'neutral': suggestions[3],
            'industry': industry
        }
    
    def _detect_industry(self, campaign_idea):
        """Detect industry from campaign idea"""
        idea_lower = campaign_idea.lower()
        
        if any(word in idea_lower for word in ['tech', 'software', 'app', 'digital']):
            return 'tech'
        elif any(word in idea_lower for word in ['food', 'restaurant', 'recipe', 'cafe']):
            return 'food'
        elif any(word in idea_lower for word in ['health', 'fitness', 'wellness', 'yoga']):
            return 'health'
        elif any(word in idea_lower for word in ['luxury', 'premium', 'exclusive', 'high-end']):
            return 'luxury'
        else:
            return 'fashion'
    
    def _generate_color_schemes(self, base_palette):
        """Generate complementary color schemes"""
        schemes = []
        
        for base_color in base_palette[:2]:  # Use first two base colors
            rgb = self.hex_to_rgb(base_color)
            hsv = colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)
            
            # Generate complementary colors
            complementary_h = (hsv[0] + 0.5) % 1.0
            analogous_1_h = (hsv[0] + 0.1) % 1.0
            analogous_2_h = (hsv[0] - 0.1) % 1.0
            
            schemes.extend([
                base_color,  # Primary
                self.hsv_to_hex(complementary_h, hsv[1], hsv[2]),  # Complementary
                self.hsv_to_hex(analogous_1_h, hsv[1] * 0.8, hsv[2]),  # Analogous 1
                self.hsv_to_hex(analogous_2_h, hsv[1] * 0.6, hsv[2] * 0.9)  # Analogous 2
            ])
        
        return list(dict.fromkeys(schemes))[:4]  # Remove duplicates and limit to 4
    
    def hex_to_rgb(self, hex_color):
        """Convert hex color to RGB"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def hsv_to_hex(self, h, s, v):
        """Convert HSV to hex color"""
        rgb = colorsys.hsv_to_rgb(h, s, v)
        return '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
    
    def get_color_names(self, hex_colors):
        """Get human-readable color names"""
        color_names = {
            '#FF6B6B': 'Coral Red', '#4ECDC4': 'Turquoise', '#45B7D1': 'Sky Blue',
            '#FFE66D': 'Sun Yellow', '#2C3E50': 'Midnight Blue', '#E74C3C': 'Alizarin',
            '#ECF0F1': 'Cloud White', '#3498DB': 'Peter River', '#F8B195': 'Peach',
            '#F67280': 'Rose Pink', '#C06C84': 'Dusty Rose', '#6C5B7B': 'Muted Purple',
            '#1A1A1D': 'Rich Black', '#4E4E50': 'Charcoal', '#C3073F': 'Ruby Red',
            '#950740': 'Wine Red', '#222831': 'Dark Gunmetal', '#393E46': 'Charcoal Blue',
            '#00ADB5': 'Blue Green', '#EEEEEE': 'Bright Gray', '#2D4059': 'Dark Blue',
            '#EA5455': 'Soft Red', '#F07B3F': 'Orange', '#FFD460': 'Maize'
        }
        
        return [color_names.get(color, f'Color {i+1}') for i, color in enumerate(hex_colors)]