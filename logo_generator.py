from PIL import Image, ImageDraw, ImageFont
import os
import random

class LogoGenerator:
    def __init__(self):
        self.shapes = ['circle', 'square', 'rounded_square', 'triangle', 'hexagon']
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFE66D', '#2C3E50', '#E74C3C']
        self.icons = ['★', '◆', '●', '▲', '■', '♠', '♥', '♣', '♦', '⭐', '✨']
    
    def generate_logo(self, brand_name, primary_color, style='modern'):
        """Generate a logo for the brand"""
        width, height = 300, 300
        logo = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(logo)
        
        # Choose style-based elements
        if style == 'modern':
            shape = random.choice(['circle', 'square'])
            icon = random.choice(['●', '■', '◆'])
        elif style == 'creative':
            shape = random.choice(['rounded_square', 'hexagon'])
            icon = random.choice(['★', '⭐', '✨'])
        else:  # classic
            shape = random.choice(['circle', 'triangle'])
            icon = random.choice(['◆', '♠', '♥'])
        
        # Draw background shape
        shape_color = primary_color
        if shape == 'circle':
            draw.ellipse([50, 50, 250, 250], fill=shape_color)
        elif shape == 'square':
            draw.rectangle([50, 50, 250, 250], fill=shape_color)
        elif shape == 'rounded_square':
            self._draw_rounded_rectangle(draw, 50, 50, 250, 250, 30, shape_color)
        elif shape == 'triangle':
            draw.polygon([(150, 50), (250, 250), (50, 250)], fill=shape_color)
        elif shape == 'hexagon':
            self._draw_hexagon(draw, 150, 150, 100, shape_color)
        
        # Draw icon
        try:
            font = ImageFont.truetype("arial.ttf", 80)
        except:
            font = ImageFont.load_default()
        
        # Use white or black text based on color brightness
        text_color = self._get_contrast_color(primary_color)
        bbox = draw.textbbox((0, 0), icon, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        draw.text(
            (width//2 - text_width//2, height//2 - text_height//2),
            icon, fill=text_color, font=font
        )
        
        # Save logo
        filename = f"logo_{brand_name.replace(' ', '_').lower()}_{random.randint(1000,9999)}.png"
        filepath = os.path.join('static', 'generated', filename)
        logo.save(filepath)
        
        return {
            'filename': filename,
            'path': f"/static/generated/{filename}",
            'style': style,
            'primary_color': primary_color
        }
    
    def _draw_rounded_rectangle(self, draw, x1, y1, x2, y2, radius, color):
        """Draw a rounded rectangle"""
        draw.rectangle([x1, y1+radius, x2, y2-radius], fill=color)
        draw.rectangle([x1+radius, y1, x2-radius, y2], fill=color)
        draw.pieslice([x1, y1, x1+radius*2, y1+radius*2], 180, 270, fill=color)
        draw.pieslice([x2-radius*2, y1, x2, y1+radius*2], 270, 360, fill=color)
        draw.pieslice([x1, y2-radius*2, x1+radius*2, y2], 90, 180, fill=color)
        draw.pieslice([x2-radius*2, y2-radius*2, x2, y2], 0, 90, fill=color)
    
    def _draw_hexagon(self, draw, center_x, center_y, size, color):
        """Draw a hexagon"""
        points = []
        for i in range(6):
            angle = 2 * 3.14159 * i / 6
            x = center_x + size * (0.8 * (1 if i % 2 == 0 else 0.9)) * (1 if i < 3 else -1) * 0.5
            y = center_y + size * (0.866 if i % 2 == 0 else 0.5) * (1 if i < 2 else -1 if i > 3 else 0)
            points.append((x, y))
        draw.polygon(points, fill=color)
    
    def _get_contrast_color(self, hex_color):
        """Get contrasting color (white or black) based on brightness"""
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        brightness = (r * 299 + g * 587 + b * 114) / 1000
        return '#000000' if brightness > 128 else '#FFFFFF'

    def generate_tagline(self, brand_name, campaign_idea, industry):
        """Generate a brand tagline"""
        taglines = {
            'fashion': [
                f"Redefining Style with {brand_name}",
                f"Where Fashion Meets Personality",
                f"Your Style, Our Passion - {brand_name}",
                f"Elevate Your Everyday with {brand_name}"
            ],
            'tech': [
                f"Innovating Tomorrow with {brand_name}",
                f"Technology that Understands You",
                f"Smart Solutions for Modern Life - {brand_name}",
                f"The Future is {brand_name}"
            ],
            'food': [
                f"Taste the Difference with {brand_name}",
                f"Where Flavor Meets Passion",
                f"Fresh. Authentic. {brand_name}.",
                f"Culinary Excellence with {brand_name}"
            ],
            'health': [
                f"Your Wellness Journey Starts with {brand_name}",
                f"Healthy Living, Simplified",
                f"Empowering Your Health - {brand_name}",
                f"Feel the Difference with {brand_name}"
            ],
            'luxury': [
                f"Exclusive Elegance by {brand_name}",
                f"Where Luxury Becomes Legacy",
                f"Timeless Sophistication - {brand_name}",
                f"The Art of Luxury Living"
            ]
        }
        
        industry_taglines = taglines.get(industry, taglines['fashion'])
        return random.choice(industry_taglines)