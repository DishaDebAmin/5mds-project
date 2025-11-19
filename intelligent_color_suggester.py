import random
import colorsys
import numpy as np
from sklearn.cluster import KMeans

class IntelligentColorSuggester:
    def __init__(self):
        self.color_psychology = {
            'red': {'emotions': ['passion', 'energy', 'excitement'], 'industries': ['food', 'entertainment', 'sports']},
            'blue': {'emotions': ['trust', 'calm', 'professionalism'], 'industries': ['tech', 'finance', 'healthcare']},
            'green': {'emotions': ['growth', 'health', 'nature'], 'industries': ['eco', 'health', 'finance']},
            'yellow': {'emotions': ['happiness', 'optimism', 'creativity'], 'industries': ['creative', 'children', 'food']},
            'purple': {'emotions': ['luxury', 'creativity', 'wisdom'], 'industries': ['beauty', 'luxury', 'creative']},
            'orange': {'emotions': ['enthusiasm', 'fun', 'confidence'], 'industries': ['entertainment', 'food', 'sports']},
            'pink': {'emotions': ['compassion', 'playfulness', 'love'], 'industries': ['beauty', 'fashion', 'kids']},
            'black': {'emotions': ['power', 'sophistication', 'elegance'], 'industries': ['luxury', 'fashion', 'automotive']}
        }
        
        self.industry_palettes = {
            'fashion': {
                'description': 'Trendy and expressive colors for fashion brands',
                'palettes': [
                    ['#E74C3C', '#ECF0F1', '#2C3E50', '#3498DB'],  # Bold & Modern
                    ['#F8B195', '#F67280', '#C06C84', '#6C5B7B'],  # Romantic & Soft
                    ['#2C3E50', '#E74C3C', '#ECF0F1', '#3498DB']   # Classic & Elegant
                ]
            },
            'tech': {
                'description': 'Trustworthy and innovative tech colors',
                'palettes': [
                    ['#2C3E50', '#3498DB', '#1ABC9C', '#ECF0F1'],  # Professional Tech
                    ['#2980B9', '#6DD5FA', '#FFFFFF', '#2C3E50'],   # Modern & Clean
                    ['#8E44AD', '#3498DB', '#2C3E50', '#ECF0F1']   # Creative Tech
                ]
            },
            'food': {
                'description': 'Appetizing and fresh food industry colors',
                'palettes': [
                    ['#E67E22', '#D35400', '#27AE60', '#F1C40F'],  # Warm & Natural
                    ['#C0392B', '#E74C3C', '#F39C12', '#27AE60'],   # Vibrant & Fresh
                    ['#F39C12', '#D35400', '#27AE60', '#16A085']   # Organic & Healthy
                ]
            },
            'health': {
                'description': 'Calming and trustworthy health colors',
                'palettes': [
                    ['#27AE60', '#2ECC71', '#3498DB', '#ECF0F1'],  # Fresh & Clean
                    ['#3498DB', '#2980B9', '#1ABC9C', '#16A085'],  # Professional & Calm
                    ['#9B59B6', '#8E44AD', '#3498DB', '#ECF0F1']   # Innovative Health
                ]
            },
            'luxury': {
                'description': 'Sophisticated and exclusive luxury colors',
                'palettes': [
                    ['#2C3E50', '#7F8C8D', '#BDC3C7', '#E74C3C'],  # Classic Luxury
                    ['#34495E', '#16A085', '#2C3E50', '#ECF0F1'],   # Modern Luxury
                    ['#8E44AD', '#2C3E50', '#E74C3C', '#F39C12']   # Bold Luxury
                ]
            }
        }
    
    def suggest_colors_with_reasoning(self, campaign_idea, brand_name, target_audience, previous_suggestions=[]):
        """Suggest brand colors with detailed reasoning"""
        industry = self._detect_industry(campaign_idea)
        audience_type = self._detect_audience_type(target_audience)
        
        # Get multiple palette options
        palette_options = []
        for i in range(3):  # Generate 3 different options
            palette = self._generate_intelligent_palette(industry, audience_type, campaign_idea)
            reasoning = self._generate_color_reasoning(palette, industry, audience_type, campaign_idea)
            
            palette_options.append({
                'primary': palette[0],
                'secondary': palette[1],
                'accent': palette[2],
                'neutral': palette[3],
                'industry': industry,
                'reasoning': reasoning,
                'palette_name': self._generate_palette_name(industry, i)
            })
        
        return palette_options
    
    def _detect_industry(self, campaign_idea):
        """Detect industry from campaign idea with more granularity"""
        idea_lower = campaign_idea.lower()
        
        if any(word in idea_lower for word in ['tech', 'software', 'app', 'digital', 'ai', 'technology']):
            return 'tech'
        elif any(word in idea_lower for word in ['food', 'restaurant', 'recipe', 'cafe', 'cooking', 'bakery']):
            return 'food'
        elif any(word in idea_lower for word in ['health', 'fitness', 'wellness', 'yoga', 'medical', 'clinic']):
            return 'health'
        elif any(word in idea_lower for word in ['luxury', 'premium', 'exclusive', 'high-end', 'boutique']):
            return 'luxury'
        elif any(word in idea_lower for word in ['fashion', 'clothing', 'apparel', 'wear', 'style', 'outfit']):
            return 'fashion'
        else:
            return 'fashion'  # Default to fashion
    
    def _detect_audience_type(self, audience_description):
        """Detect audience type with more granularity"""
        audience_lower = audience_description.lower()
        
        if any(word in audience_lower for word in ['teen', 'student', 'school', 'young', 'gen z']):
            return 'youth'
        elif any(word in audience_lower for word in ['professional', 'business', 'corporate', 'executive', 'enterprise']):
            return 'professional'
        elif any(word in audience_lower for word in ['family', 'parents', 'mother', 'father', 'home']):
            return 'family'
        elif any(word in audience_lower for word in ['creative', 'artist', 'designer', 'maker', 'innovator']):
            return 'creative'
        else:
            return 'general'
    
    def _generate_intelligent_palette(self, industry, audience_type, campaign_idea):
        """Generate intelligent color palette based on context"""
        base_palettes = self.industry_palettes[industry]['palettes']
        base_palette = random.choice(base_palettes)
        
        # Adjust palette based on audience
        adjusted_palette = self._adjust_for_audience(base_palette, audience_type)
        
        return adjusted_palette
    
    def _adjust_for_audience(self, palette, audience_type):
        """Adjust colors based on target audience preferences"""
        if audience_type == 'youth':
            # More vibrant, saturated colors
            return [self._increase_saturation(color, 1.3) for color in palette]
        elif audience_type == 'professional':
            # More muted, sophisticated colors
            return [self._decrease_saturation(color, 0.8) for color in palette]
        elif audience_type == 'family':
            # Warmer, friendlier colors
            return [self._adjust_temperature(color, 'warmer') for color in palette]
        elif audience_type == 'creative':
            # More unique, contrasting colors
            return self._create_contrasting_palette(palette)
        else:
            return palette
    
    def _generate_color_reasoning(self, palette, industry, audience_type, campaign_idea):
        """Generate detailed reasoning for color choices"""
        primary_color = palette[0]
        color_name = self._get_color_family(primary_color)
        
        reasoning = {
            'primary': f"{color_name.capitalize()} evokes {self._get_emotion_for_color(color_name)},"
                       f" perfect for {industry} brands targeting {audience_type} audiences.",
            'secondary': f"Complements the primary color while maintaining brand consistency"
                         f" and visual hierarchy.",
            'accent': f"Adds visual interest and calls attention to key elements"
                      f" without overwhelming the design.",
            'overall': f"This palette combines professionalism with approachability,"
                       f" making it ideal for {campaign_idea[:30]}..."
        }
        
        return reasoning
    
    def _get_color_family(self, hex_color):
        """Get the color family name from hex"""
        color_families = {
            'red': ['#FF6B6B', '#E74C3C', '#C0392B', '#FF4757'],
            'blue': ['#3498DB', '#2980B9', '#1ABC9C', '#74B9FF'],
            'green': ['#27AE60', '#2ECC71', '#1ABC9C', '#55E6C1'],
            'yellow': ['#F1C40F', '#F39C12', '#E67E22', '#FF9FF3'],
            'purple': ['#9B59B6', '#8E44AD', '#6C5CE7', '#A29BFE'],
            'orange': ['#E67E22', '#D35400', '#F39C12', '#FF7979'],
            'pink': ['#F8B195', '#F67280', '#C06C84', '#FD79A8'],
            'black': ['#2C3E50', '#34495E', '#2D3436', '#636E72']
        }
        
        for family, colors in color_families.items():
            if hex_color.upper() in [c.upper() for c in colors]:
                return family
        
        return 'professional'
    
    def _get_emotion_for_color(self, color_family):
        """Get emotions associated with color family"""
        emotions = {
            'red': 'passion, energy, and excitement',
            'blue': 'trust, calmness, and professionalism', 
            'green': 'growth, health, and nature',
            'yellow': 'happiness, optimism, and creativity',
            'purple': 'luxury, creativity, and wisdom',
            'orange': 'enthusiasm, fun, and confidence',
            'pink': 'compassion, playfulness, and love',
            'black': 'power, sophistication, and elegance'
        }
        return emotions.get(color_family, 'professionalism and trust')
    
    def _generate_palette_name(self, industry, index):
        """Generate creative names for color palettes"""
        names = {
            'fashion': ['Runway Elegance', 'Urban Chic', 'Designer Dreams'],
            'tech': ['Digital Future', 'Innovation Core', 'Tech Precision'],
            'food': ['Fresh Harvest', 'Culinary Art', 'Natural Taste'],
            'health': ['Wellness Harmony', 'Pure Balance', 'Vitality Boost'],
            'luxury': ['Opulent Gold', 'Executive Class', 'Premium Elite']
        }
        return names.get(industry, ['Professional', 'Modern', 'Creative'])[index]
    
    def _increase_saturation(self, hex_color, factor):
        """Increase color saturation"""
        return self._adjust_hsv(hex_color, s_factor=factor)
    
    def _decrease_saturation(self, hex_color, factor):
        """Decrease color saturation"""
        return self._adjust_hsv(hex_color, s_factor=factor)
    
    def _adjust_temperature(self, hex_color, temperature):
        """Make color warmer or cooler"""
        if temperature == 'warmer':
            return self._adjust_hsv(hex_color, h_shift=0.02)  # Shift toward red/orange
        else:
            return self._adjust_hsv(hex_color, h_shift=-0.02)  # Shift toward blue
    
    def _create_contrasting_palette(self, palette):
        """Create a more contrasting palette for creative audiences"""
        # Ensure good contrast between colors
        new_palette = [palette[0]]
        for color in palette[1:]:
            if self._color_distance(new_palette[-1], color) < 100:
                # Colors too similar, adjust
                adjusted = self._adjust_hsv(color, h_shift=0.3)
                new_palette.append(adjusted)
            else:
                new_palette.append(color)
        return new_palette
    
    def _color_distance(self, hex1, hex2):
        """Calculate color distance in RGB space"""
        r1, g1, b1 = self.hex_to_rgb(hex1)
        r2, g2, b2 = self.hex_to_rgb(hex2)
        return ((r1-r2)**2 + (g1-g2)**2 + (b1-b2)**2) ** 0.5
    
    def _adjust_hsv(self, hex_color, h_shift=0, s_factor=1, v_factor=1):
        """Adjust color in HSV space"""
        r, g, b = self.hex_to_rgb(hex_color)
        h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
        
        h = (h + h_shift) % 1.0
        s = min(max(s * s_factor, 0), 1)
        v = min(max(v * v_factor, 0), 1)
        
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))
    
    def hex_to_rgb(self, hex_color):
        """Convert hex color to RGB"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))